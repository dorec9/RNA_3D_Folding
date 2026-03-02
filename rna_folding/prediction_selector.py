import numpy as np
from typing import List, Dict, Any

class PredictionSelector:
    """
    Agentic Tree Search Algorithm implementation.
    Responsible for selecting the optimal 'Best-of-5' submissions from a large pool of 
    candidate models (TBM, Protenix, DRFold2), balancing physical energy, prediction 
    confidence, and structural diversity to maximize the multi-hit TM-score probability.
    """
    def __init__(self, alpha_diversity: float = 0.5):
        """
        Args:
            alpha_diversity (float): Weight parameter for the diversity penalty. A higher 
                                     value penalizes candidates that are structurally too similar 
                                     to previously chosen anchors.
        """
        self.alpha_diversity = alpha_diversity

    def _calculate_pairwise_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        Helper method to calculate Root Mean Square Deviation between two aligned C1' coordinate arrays.
        *(Placeholder for actual exact implementation needing Kabsch alignment)*
        """
        diff = coords1 - coords2
        return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    def score_candidate(self, candidate_conf: float, candidate_coords: np.ndarray, anchors: List[np.ndarray]) -> float:
        """
        Calculates a selection score combining raw confidence and an RMSD-based diversity bonus 
        relative to the already chosen anchors.
        
        Args:
            candidate_conf (float): The base confidence or inverted Energy score (higher is better).
            candidate_coords (np.ndarray): Candidate's C1' coordinates.
            anchors (List[np.ndarray]): The coordinates of the structures already selected.
            
        Returns:
            float: The final adjusted selection score.
        """
        if not anchors:
            return candidate_conf
            
        # Diversity score: the minimum RMSD distance to any existing anchor
        min_rmsd_to_anchors = min(self._calculate_pairwise_rmsd(candidate_coords, anc) for anc in anchors)
        
        # We boost the score if the structure explores a slightly different conformation space.
        # But if it diverges completely (creating nonsense), confidence drops so overall score lowers.
        adjusted_score = candidate_conf + (self.alpha_diversity * min_rmsd_to_anchors)
        return adjusted_score

    def select_best_of_5(self, candidates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Execute the greedy tree search.

        Expects `candidates` to be a list of dicts with keys:
            - 'source': str (e.g., 'TBM', 'Protenix', 'DRFold2_Hybrid')
            - 'coords': np.ndarray (Nx3)
            - 'conf': float (e.g., ARES pLDDT score or negative FARFAR2 energy)
        
        Returns:
            List[np.ndarray]: Exactly 5 C1' coordinate arrays.
        """
        if len(candidates) <= 5:
            # If we generated 5 or fewer via fallback, just return all
            return [c['coords'] for c in candidates]
            
        selected_structures = []
        selected_anchors_coords = []
        
        # 1. First Anchor: Highest confidence Structure from TBM (Local geometry preservation)
        tbm_candidates = [c for c in candidates if c['source'] == 'TBM']
        if tbm_candidates:
            best_tbm = max(tbm_candidates, key=lambda x: x['conf'])
            selected_structures.append(best_tbm['coords'])
            selected_anchors_coords.append(best_tbm['coords'])
            candidates.remove(best_tbm)

        # 2. Second Anchor: Highest confidence Structure from Protenix (Global form preservation)
        protenix_cands = [c for c in candidates if c['source'] == 'Protenix']
        if protenix_cands:
            best_protenix = max(protenix_cands, key=lambda x: x['conf'])
            selected_structures.append(best_protenix['coords'])
            selected_anchors_coords.append(best_protenix['coords'])
            candidates.remove(best_protenix)
            
        # 3. Fill the remaining spots using greedy approach balancing conf and diversity
        while len(selected_structures) < 5 and len(candidates) > 0:
            best_score = -float('inf')
            best_candidate = None
            
            for cand in candidates:
                current_score = self.score_candidate(cand['conf'], cand['coords'], selected_anchors_coords)
                if current_score > best_score:
                    best_score = current_score
                    best_candidate = cand
                    
            selected_structures.append(best_candidate['coords'])
            selected_anchors_coords.append(best_candidate['coords'])
            candidates.remove(best_candidate)
            
        # In a rare edge case where we still lack 5, pad with the last known structure
        while len(selected_structures) < 5:
            selected_structures.append(selected_structures[-1])
            
        return selected_structures
