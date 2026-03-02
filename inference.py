"""
End-to-end orchestrator for the Kaggle Stanford RNA 3D Folding pipeline.

Architecture (ordered by expected impact):
1. Template-Based Modeling (TBM) via MMseqs2 --search-type 3
   - Zero GPU cost; runs on CPU in minutes
   - Directly implements the 1st-place solution from Part 1
2. Protenix deep learning inference (highest impact for template-poor targets)
   - FP16 precision for T4 GPU; RNA MSA integration
3. DRFold2 as lightweight fallback / hybrid refiner
   - Single-sequence, no MSA needed; fast on CPU or GPU
4. Agentic tree search best-of-5 selection
   - Balances structural diversity against model-reliability weights

References:
    1st-place solution (jaejohn): MMseqs2 TBM, mean TM-align 0.593
    Post-competition tree search (arunodhayan/d4t4): mean TM-align 0.635
    RNAPro model ablation: templates are the single most important input feature
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from rna_folding.data_loader import find_msa_path, load_msa
from rna_folding.template_search import TemplateSearcher
from rna_folding.protenix_runner import ProtenixRunner
from rna_folding.drfold2_runner import DRFold2Runner
from rna_folding.prediction_selector import PredictionSelector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters (all overridable via config dict)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # TBM
    "pdb_db_path": "",
    "cif_dir": "",
    "mmseqs_bin": "mmseqs",
    "mmseqs_tmp_dir": "/tmp/mmseqs_rna",
    "tbm_top_k": 5,
    # Protenix
    "protenix_weights": "",
    "use_rna_msa": True,
    "protenix_recycling_steps": 3,
    "protenix_samples": 3,
    # DRFold2
    "drfold2_weights": "",
    # MSA
    "msa_root": "",
    # Selection
    "w_div": 0.40,
    "w_dist": 0.25,
    # Runtime
    "vram_gb": 16,
    "time_limit_hours": 8.0,
}

# Confidence scores assigned when models don't return an explicit pLDDT.
# Higher → higher priority in the selection stage.
_CONF_TBM = 100.0       # TBM anchor: always highest raw conf
_CONF_PROTENIX = 90.0   # Protenix base conf (decrements per sample)
_CONF_DRFOLD2 = 70.0    # DRFold2 base conf (decrements per sample)


class RNAFoldingPipeline:
    """
    End-to-end orchestrator for generating best-of-5 RNA 3D structure predictions.

    Initialise once per Kaggle session, then call :meth:`run` (bulk) or
    :meth:`process_target` (per-target) in a loop.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = {**_DEFAULTS, **config}

        self.tbm = TemplateSearcher(
            pdb_db_path=cfg["pdb_db_path"],
            cif_dir=cfg["cif_dir"],
            mmseqs_bin=cfg["mmseqs_bin"],
            tmp_dir=cfg["mmseqs_tmp_dir"],
        )
        self.tbm_top_k: int = cfg["tbm_top_k"]

        self.protenix = ProtenixRunner(
            protenix_weights_path=cfg["protenix_weights"],
            use_rna_msa=cfg["use_rna_msa"],
            num_recycling_steps=cfg["protenix_recycling_steps"],
            num_diffusion_samples=cfg["protenix_samples"],
            vram_gb=cfg["vram_gb"],
        )

        self.drfold2 = DRFold2Runner(model_dir=cfg["drfold2_weights"])

        self.selector = PredictionSelector(
            w_div=cfg["w_div"],
            w_dist=cfg["w_dist"],
        )

        self.msa_root: str = cfg["msa_root"]
        self.vram_gb: int = cfg["vram_gb"]
        self.time_limit_seconds: float = cfg["time_limit_hours"] * 3600.0

    # ------------------------------------------------------------------
    # Time budget
    # ------------------------------------------------------------------

    def compute_budget(
        self, start_time: float, target_idx: int, total_targets: int, seq_len: int
    ) -> Dict[str, bool]:
        """
        Decide which pipeline branches to run for the current target.

        Heuristics:
        - *panic mode* (< 1 h remaining): only TBM + DRFold2 single
        - *constrained mode* (< 2 h OR long sequence): skip DRFold2 hybrid
        - *full mode*: TBM + Protenix + DRFold2 hybrid

        Args:
            start_time: ``time.time()`` at pipeline start.
            target_idx: 0-based index of the current target.
            total_targets: Total number of targets to process.
            seq_len: Length of the current target sequence.

        Returns:
            Dict with boolean keys ``run_protenix``, ``run_drfold2_hybrid``,
            ``panic_mode``.
        """
        elapsed = time.time() - start_time
        remaining = self.time_limit_seconds - elapsed
        avg_per_target = elapsed / max(1, target_idx)
        projected_remaining_targets = total_targets - target_idx
        projected_needed = avg_per_target * projected_remaining_targets

        panic = remaining < 3600.0
        constrained = (
            remaining < 7200.0
            or seq_len > 1000
            or projected_needed > remaining * 1.5
        )

        return {
            "run_protenix": not panic,
            "run_drfold2_hybrid": not panic and not constrained,
            "panic_mode": panic,
        }

    # ------------------------------------------------------------------
    # Per-target processing
    # ------------------------------------------------------------------

    def process_target(
        self,
        target_id: str,
        sequence: str,
        budget: Dict[str, bool],
    ) -> List[np.ndarray]:
        """
        Generate 5 candidate structures for a single target.

        Args:
            target_id: Competition target identifier.
            sequence: RNA sequence string.
            budget: Dict from :meth:`compute_budget`.

        Returns:
            List of exactly 5 ``(len(sequence), 3)`` float32 arrays.
        """
        candidates: List[Dict[str, Any]] = []

        # 1. Template-Based Modeling ----------------------------------------
        # TBM runs on CPU and is always fast — run unconditionally.
        tbm_results = self.tbm.get_tbm_candidates_with_evalue(
            sequence, top_k=self.tbm_top_k, target_id=target_id
        )
        for rank, res in enumerate(tbm_results):
            candidates.append({
                "source": "TBM",
                "coords": res["coords"],
                # Higher e-value rank = lower confidence
                "conf": _CONF_TBM - rank * 5.0,
                "evalue": res["evalue"],
            })

        # 2. Protenix deep learning -----------------------------------------
        protenix_coords: List[np.ndarray] = []
        if budget.get("run_protenix"):
            msa_str = ""
            if self.msa_root:
                msa_path = find_msa_path(target_id, self.msa_root)
                if msa_path:
                    msa_str = load_msa(msa_path)

            # Pass the best TBM structure as a template hint (if available)
            template_hint = tbm_results[0]["coords"] if tbm_results else None

            protenix_coords = self.protenix.predict_structures(
                sequence=sequence,
                msa_str=msa_str,
                template_coords=template_hint,
                num_return_sequences=3,
            )
            for rank, coords in enumerate(protenix_coords):
                candidates.append({
                    "source": "Protenix",
                    "coords": coords,
                    "conf": _CONF_PROTENIX - rank * 5.0,
                })

        # 3. DRFold2 ----------------------------------------------------------
        if budget.get("run_drfold2_hybrid") and protenix_coords:
            # Hybrid: use top Protenix structure as reference potential
            hybrid_coords = self.drfold2.predict_hybrid(
                sequence=sequence,
                ref_coords=protenix_coords[0],
                num_return=2,
            )
            for rank, coords in enumerate(hybrid_coords):
                candidates.append({
                    "source": "DRFold2_Hybrid",
                    "coords": coords,
                    "conf": _CONF_DRFOLD2 - rank * 3.0,
                })

        elif budget.get("panic_mode"):
            # Panic: single-sequence DRFold2 only (no GPU needed)
            single_coords = self.drfold2.predict_single(sequence, num_return=3)
            for rank, coords in enumerate(single_coords):
                candidates.append({
                    "source": "DRFold2_Single",
                    "coords": coords,
                    "conf": _CONF_DRFOLD2 - rank * 3.0,
                })

        # 4. Agentic tree search selection ------------------------------------
        return self.selector.select_best_of_5(candidates)

    # ------------------------------------------------------------------
    # Submission serialization
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_for_submission(
        target_id: str,
        best_5_coords: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Convert 5 coordinate arrays into the Kaggle submission row format.

        Each row represents one nucleotide position and carries the C1'
        coordinates from all 5 predictions::

            id, x_1, y_1, z_1, x_2, y_2, z_2, ..., x_5, y_5, z_5

        Args:
            target_id: Competition target identifier.
            best_5_coords: List of exactly 5 ``(N, 3)`` arrays.

        Returns:
            List of dicts — one per nucleotide position.
        """
        # Pad to exactly 5 if needed
        while len(best_5_coords) < 5:
            best_5_coords.append(best_5_coords[-1].copy() if best_5_coords
                                 else np.zeros((1, 3), dtype=np.float32))

        num_residues = best_5_coords[0].shape[0]
        rows: List[Dict[str, Any]] = []

        for res_idx in range(num_residues):
            row: Dict[str, Any] = {"id": f"{target_id}_{res_idx + 1}"}
            for model_idx in range(5):
                c = best_5_coords[model_idx]
                xyz = c[res_idx] if res_idx < len(c) else np.zeros(3)
                row[f"x_{model_idx + 1}"] = float(xyz[0])
                row[f"y_{model_idx + 1}"] = float(xyz[1])
                row[f"z_{model_idx + 1}"] = float(xyz[2])
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Bulk runner
    # ------------------------------------------------------------------

    def run(
        self,
        test_df: pd.DataFrame,
        output_path: str = "submission.csv",
    ) -> pd.DataFrame:
        """
        Process all targets in *test_df* and write submission CSV.

        Args:
            test_df: DataFrame with columns ``target_id`` and ``sequence``.
            output_path: Destination path for the submission CSV.

        Returns:
            The submission DataFrame.
        """
        total = len(test_df)
        all_rows: List[Dict[str, Any]] = []
        start_time = time.time()

        for idx, row in test_df.iterrows():
            target_id = str(row["target_id"])
            sequence = str(row["sequence"])
            seq_len = len(sequence)

            logger.info(
                "[%d/%d] %s  len=%d", idx + 1, total, target_id, seq_len
            )

            budget = self.compute_budget(start_time, idx, total, seq_len)
            best_5 = self.process_target(target_id, sequence, budget)
            all_rows.extend(self.serialize_for_submission(target_id, best_5))

        submission = pd.DataFrame(all_rows)
        submission.to_csv(output_path, index=False)
        logger.info("Submission saved to %s (%d rows)", output_path, len(submission))
        return submission
