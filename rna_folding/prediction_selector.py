"""
Agentic tree search for best-of-5 submission selection.

Implements the post-competition winning strategy described in the competition
analysis:

    score = w_model × (w_div × diversity − w_dist × distance_to_priors)

where:
- ``w_model``  — source reliability weight  (Protenix: 0.75, TBM: 0.15, DRFold2: 0.10)
- ``w_div``    — diversity bonus weight      (mean pairwise Kabsch-aligned RMSD)
- ``w_dist``   — anchor distance penalty    (distance to selected anchors)
- Anchors: TBM conformation 1 (slot 0) and Protenix conformation 2 (slot 1)

Reference:
    arunodhayan (d4t4 team) post-competition analysis, Kaggle discussion thread
    Mean TM-align 0.635 — highest individual score achieved after Part 1
"""

import logging
from typing import Any, Dict, List

import numpy as np

from rna_folding.coordinate_utils import align_kabsch, calculate_rmsd

logger = logging.getLogger(__name__)

# Model reliability weights from the competition analysis
MODEL_WEIGHTS: Dict[str, float] = {
    "TBM": 0.15,
    "Protenix": 0.75,
    "DRFold2": 0.10,
    "DRFold2_Hybrid": 0.10,
    "DRFold2_Single": 0.10,
}

# Scoring hyper-parameters (diversity bonus / anchor penalty)
W_DIV = 0.40    # reward for structural diversity
W_DIST = 0.25   # penalty for being too close to an anchor


class PredictionSelector:
    """
    Select the optimal 5-structure submission from a pool of candidates.

    The algorithm mirrors the agentic tree search that achieved TM-align 0.635:
    1. Slot 0 → best TBM structure (local geometry anchor)
    2. Slot 1 → best Protenix structure (global form anchor)
    3. Slots 2–4 → greedy fill using the weighted diversity-quality score

    The selector handles edge cases (fewer than 5 candidates, shape mismatches,
    missing sources) gracefully by padding with the last selected structure.
    """

    def __init__(
        self,
        w_div: float = W_DIV,
        w_dist: float = W_DIST,
    ) -> None:
        """
        Args:
            w_div: Weight applied to the mean pairwise RMSD diversity bonus.
            w_dist: Weight applied to the anchor distance penalty.
        """
        self.w_div = w_div
        self.w_dist = w_dist

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
        """Kabsch-aligned RMSD; returns inf on shape mismatch."""
        if a.shape != b.shape or a.shape[0] == 0:
            return float("inf")
        try:
            return calculate_rmsd(a, b)
        except Exception:
            return float("inf")

    def _diversity_score(
        self, coords: np.ndarray, pool: List[np.ndarray]
    ) -> float:
        """
        Mean pairwise Kabsch-aligned RMSD between *coords* and all structures
        in *pool*.  Higher → more diverse → better.
        """
        if not pool:
            return 0.0
        rmsds = [self._kabsch_rmsd(coords, p) for p in pool]
        finite = [r for r in rmsds if r < float("inf")]
        return float(np.mean(finite)) if finite else 0.0

    def _anchor_distance(
        self, coords: np.ndarray, anchors: List[np.ndarray]
    ) -> float:
        """
        Minimum Kabsch-aligned RMSD between *coords* and any anchor.
        Lower → geometrically closer to anchors → potential penalty.
        """
        if not anchors:
            return 0.0
        dists = [self._kabsch_rmsd(coords, a) for a in anchors]
        return float(min(d for d in dists if d < float("inf")) or 0.0)

    def _score(
        self,
        candidate: Dict[str, Any],
        selected_coords: List[np.ndarray],
        anchors: List[np.ndarray],
    ) -> float:
        """
        Compute the weighted selection score for a single candidate.

        Formula:
            score = w_model × (w_div × diversity − w_dist × dist_to_anchors)

        where diversity is the mean RMSD to all already-selected structures.
        """
        source = candidate.get("source", "DRFold2")
        w_model = MODEL_WEIGHTS.get(source, 0.10)
        coords = candidate["coords"]

        diversity = self._diversity_score(coords, selected_coords)
        anchor_dist = self._anchor_distance(coords, anchors)

        return w_model * (self.w_div * diversity - self.w_dist * anchor_dist)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best_of_5(
        self, candidates: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Select exactly 5 structures from *candidates*.

        Each candidate dict must have:
        - ``'source'``: str — one of 'TBM', 'Protenix', 'DRFold2', etc.
        - ``'coords'``: np.ndarray of shape (N, 3)
        - ``'conf'``: float — model confidence (higher is better); used only
          when the diversity score is zero (tie-breaking at the start).

        Args:
            candidates: Pool of candidate structures (any size ≥ 0).

        Returns:
            List of exactly 5 ``(N, 3)`` coordinate arrays.  If the pool has
            fewer than 5 entries, the last entry is repeated to pad to 5.
        """
        if not candidates:
            logger.warning("No candidates provided; returning zero coordinates.")
            return [np.zeros((1, 3), dtype=np.float32)] * 5

        # Work on a shallow copy to avoid mutating caller's list
        pool = list(candidates)

        selected: List[np.ndarray] = []
        anchors: List[np.ndarray] = []

        # ----------------------------------------------------------
        # Slot 0: best TBM structure (local geometry anchor)
        # ----------------------------------------------------------
        tbm_pool = [c for c in pool if c.get("source") == "TBM"]
        if tbm_pool:
            best_tbm = max(tbm_pool, key=lambda c: c["conf"])
            selected.append(best_tbm["coords"])
            anchors.append(best_tbm["coords"])
            pool.remove(best_tbm)
            logger.debug("Slot 0 → TBM (conf=%.2f)", best_tbm["conf"])

        # ----------------------------------------------------------
        # Slot 1: best Protenix structure (global form anchor)
        # ----------------------------------------------------------
        prot_pool = [c for c in pool if c.get("source") == "Protenix"]
        if prot_pool:
            best_prot = max(prot_pool, key=lambda c: c["conf"])
            selected.append(best_prot["coords"])
            anchors.append(best_prot["coords"])
            pool.remove(best_prot)
            logger.debug("Slot 1 → Protenix (conf=%.2f)", best_prot["conf"])

        # ----------------------------------------------------------
        # Slots 2–4: greedy diversity-quality fill
        # ----------------------------------------------------------
        while len(selected) < 5 and pool:
            scored = [
                (c, self._score(c, selected, anchors))
                for c in pool
            ]
            best_cand, best_s = max(scored, key=lambda x: x[1])
            selected.append(best_cand["coords"])
            pool.remove(best_cand)
            logger.debug(
                "Slot %d → %s (score=%.4f)",
                len(selected) - 1,
                best_cand.get("source", "?"),
                best_s,
            )

        # ----------------------------------------------------------
        # Pad to exactly 5 if the pool was too small
        # ----------------------------------------------------------
        while len(selected) < 5:
            selected.append(selected[-1].copy())

        return selected[:5]
