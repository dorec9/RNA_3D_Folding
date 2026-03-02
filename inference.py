import time
import pandas as pd
import numpy as np
from typing import List

from rna_folding.data_loader import load_msa
from rna_folding.template_search import TemplateSearcher
from rna_folding.protenix_runner import ProtenixRunner
from rna_folding.drfold2_runner import DRFold2Runner
from rna_folding.prediction_selector import PredictionSelector

class RNAFoldingPipeline:
    """
    End-to-End Orchestrator for the Kaggle Stanford RNA 3D Folding pipeline.
    Responsible for initializing all sub-modules, distributing the compute budget 
    across targets, and assembling final coordinate dictionaries for CSV submission.
    """
    def __init__(self, config: dict):
        self.config = config
        
        self.tbm = TemplateSearcher(
            pdb_db_path=config.get('pdb_db_path', ''),
            mmseqs_bin=config.get('mmseqs_bin', 'mmseqs')
        )
        
        self.protenix = ProtenixRunner(
            protenix_weights_path=config.get('protenix_weights', ''),
            aido_weights_path=config.get('aido_weights', '')
        )
        
        self.drfold2 = DRFold2Runner(
            model_dir=config.get('drfold2_weights', '')
        )
        
        self.selector = PredictionSelector(
            alpha_diversity=config.get('alpha_diversity', 0.5)
        )

    def check_time_budget(self, start_time: float, target_idx: int, total_targets: int, sequence: str) -> dict:
        """
        Dynamically decide which branches of the pipeline to run based on the sequence
        length and the remaining time before the 8-hour Kaggle cutoff.
        """
        elapsed = time.time() - start_time
        avg_time_per_target = elapsed / max(1, target_idx)
        remaining_time = (8 * 3600) - elapsed
        
        # Heuristics: if sequence is too long, we might skip DRFold2 Hybrid 
        # to save memory and time
        seq_len = len(sequence)
        run_protenix = True
        run_drfold2_hybrid = True
        
        if seq_len > 1000 and remaining_time < (avg_time_per_target * (total_targets - target_idx) * 1.5):
            run_drfold2_hybrid = False # Fast fallback
            
        if remaining_time < 3600: # Less than 1 hour left, panic mode
            run_drfold2_hybrid = False
            run_protenix = False # Fall back completely to DRFold2 Single or TBM
            
        return {
            "run_protenix": run_protenix,
            "run_drfold2_hybrid": run_drfold2_hybrid,
            "panic_mode": remaining_time < 3600
        }

    def process_target(self, target_id: str, sequence: str, budget: dict) -> List[np.ndarray]:
        """
        Generate 5 submission structures for a single target sequence.
        """
        candidates = []
        
        # 1. TBM execution
        tbm_coords = self.tbm.get_tbm_candidates(sequence, top_k=2)
        for i, coords in enumerate(tbm_coords):
            # Give high arbitrary confidence to TBM to force it as an anchor
            candidates.append({'source': 'TBM', 'coords': coords, 'conf': 100.0 - i})
            
        protenix_coords = []
        if budget.get('run_protenix', True):
            # Mock MSA load
            msa_str = load_msa(f"/mock/path/{target_id}.a3m")
            protenix_coords = self.protenix.predict_structures(sequence, msa_str, num_return_sequences=3)
            for i, coords in enumerate(protenix_coords):
                candidates.append({'source': 'Protenix', 'coords': coords, 'conf': 90.0 - (i * 5)})
                
        if budget.get('run_drfold2_hybrid', True) and protenix_coords:
            # We use the top Protenix structure as the Reference Potential
            top_ref = protenix_coords[0]
            hybrid_coords = self.drfold2.predict_hybrid(sequence, top_ref, num_return=3)
            for i, coords in enumerate(hybrid_coords):
                # DRFold2 Hybrid receives baseline confidence points
                candidates.append({'source': 'DRFold2_Hybrid', 'coords': coords, 'conf': 80.0 - i})
        elif budget.get('panic_mode', False):
             single_coords = self.drfold2.predict_single(sequence, num_return=5)
             for i, coords in enumerate(single_coords):
                 candidates.append({'source': 'DRFold2_Single', 'coords': coords, 'conf': 50.0 - i})
                 
        # 4. Selection (Agentic Tree Search)
        best_5_coords = self.selector.select_best_of_5(candidates)
        return best_5_coords

    def serialize_for_submission(self, target_id: str, best_5_coords: List[np.ndarray]) -> List[dict]:
        """
        Convert the list of 5 Nx3 C1' coordinate arrays into the long tabular format
        expected by the Kaggle submission rules.
        Format: id (sequence_id), pred_model_1_x, pred_model_1_y, ... pred_model_5_z
        """
        # Note: Actual Kaggle Stanford RNA 3D formatting is typically 1 row per nucleotide position per target
        # Here we mock the shape serialization
        rows = []
        
        # For demonstration: assuming uniform length matches `best_5_coords[0]`
        num_residues = len(best_5_coords[0])
        
        # Simple fallback representation if we don't have exactly 5 or coordinates are malformed
        while len(best_5_coords) < 5:
            best_5_coords.append(np.zeros((num_residues, 3)))
            
        for res_idx in range(num_residues):
            row_dict = {"id": f"{target_id}_{res_idx}"}
            for model_idx in range(5):
                coords = best_5_coords[model_idx][res_idx]
                row_dict[f"res_id_{model_idx+1}"] = res_idx + 1 # 1-indexed
                row_dict[f"x_{model_idx+1}"] = coords[0]
                row_dict[f"y_{model_idx+1}"] = coords[1]
                row_dict[f"z_{model_idx+1}"] = coords[2]
            rows.append(row_dict)
            
        return rows
