"""
DRFold2 inference wrapper for RNA 3D structure prediction.

DRFold2 is a single-sequence model (no MSA required), making it valuable as a
lightweight fallback when templates are absent and GPU memory is insufficient
for Protenix.  It also supports a hybrid mode that incorporates an external
reference coordinate set as a structural prior.

References:
    DRFold2 GitHub: github.com/leeyang/DRfold2
    Competition analysis: 3rd-place team's Protenix + DRFold2 ensemble
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from rna_folding.memory_utils import clear_vram

logger = logging.getLogger(__name__)

# Weight of the reference potential in hybrid coordinate blending.
# Mirrors the 70 % reference / 30 % DRFold2 heuristic from competition analysis.
HYBRID_REF_WEIGHT = 0.70


class DRFold2Runner:
    """
    Wrapper for DRFold2 single-sequence RNA structure prediction.

    Exposes two inference modes:

    * **Single mode** — standard DRFold2 inference from sequence alone.
    * **Hybrid mode** — blends DRFold2 output with a TBM or Protenix reference.
      Serves as the SCOR (Structural Context and Reference) component that
      combines diverse modelling philosophies without requiring a second neural
      network call.
    """

    def __init__(self, model_dir: str) -> None:
        """
        Args:
            model_dir: Directory containing DRFold2 model weights and scripts.
                       The runner expects either a ``predict.py`` entry point or
                       a ``drfold2`` binary on PATH.
        """
        self.model_dir = model_dir
        self._model = None  # Lazy-loaded Python API handle

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self):
        """Attempt to import DRFold2 Python API; return None on failure."""
        if self._model is not None:
            return self._model
        try:
            import sys
            if self.model_dir and self.model_dir not in sys.path:
                sys.path.insert(0, self.model_dir)
            from drfold2 import DRFold2  # noqa: F401
            self._model = DRFold2(model_dir=self.model_dir)
            logger.info("DRFold2 loaded from %s", self.model_dir)
            return self._model
        except Exception as exc:
            logger.warning("DRFold2 Python API unavailable (%s). Will use subprocess.", exc)
            return None

    # ------------------------------------------------------------------
    # Subprocess runner
    # ------------------------------------------------------------------

    def _predict_subprocess(
        self,
        sequence: str,
        output_dir: str,
        num_return: int,
        ref_coords: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Run DRFold2 via its command-line ``predict.py`` script.

        Looks for a ``predict.py`` in ``model_dir``, or falls back to the
        ``drfold2`` binary if available.
        """
        fasta_path = os.path.join(output_dir, "query.fasta")
        with open(fasta_path, "w") as fh:
            fh.write(f">query\n{sequence}\n")

        predict_script = os.path.join(self.model_dir, "predict.py")
        if os.path.exists(predict_script):
            cmd = [
                "python", predict_script,
                "--fasta", fasta_path,
                "--output", output_dir,
                "--num_models", str(num_return),
            ]
        else:
            cmd = [
                "drfold2",
                "--fasta", fasta_path,
                "--output", output_dir,
                "--num_models", str(num_return),
            ]

        logger.debug("Running DRFold2: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("DRFold2 subprocess failed: %s", exc)
            return []

        coords_list: List[np.ndarray] = []
        for fname in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, fname)
            if fname.endswith(".npy"):
                arr = np.load(path).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    coords_list.append(arr)
        return coords_list[:num_return]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, sequence: str, num_return: int = 5) -> List[np.ndarray]:
        """
        Standard DRFold2 inference from sequence alone (no MSA, no reference).

        Args:
            sequence: Target RNA sequence.
            num_return: Number of structure samples to generate.

        Returns:
            List of ``(len(sequence), 3)`` float32 arrays.
        """
        model = self._try_load_model()
        if model is not None:
            try:
                results = model.predict(sequence, num_models=num_return)
                coords_list = []
                for r in results[:num_return]:
                    arr = np.array(r, dtype=np.float32)
                    if arr.shape == (len(sequence), 3):
                        coords_list.append(arr)
                clear_vram()
                if coords_list:
                    return coords_list
            except Exception as exc:
                logger.warning("DRFold2 API predict failed: %s", exc)
                clear_vram()

        with tempfile.TemporaryDirectory() as tmpdir:
            return self._predict_subprocess(sequence, tmpdir, num_return)

    def predict_hybrid(
        self,
        sequence: str,
        ref_coords: np.ndarray,
        num_return: int = 3,
    ) -> List[np.ndarray]:
        """
        Hybrid DRFold2 inference guided by a reference coordinate set.

        The reference (from TBM or Protenix) is blended with DRFold2 predictions
        using ``HYBRID_REF_WEIGHT`` (0.70 reference, 0.30 DRFold2 by default).
        This implements the SCOR-style structural prior injection without
        requiring a dedicated module.

        Args:
            sequence: Target RNA sequence.
            ref_coords: ``(N, 3)`` reference C1' coordinates from TBM or Protenix.
            num_return: Number of hybrid structures to produce.

        Returns:
            List of ``(len(sequence), 3)`` float32 blended coordinate arrays.
        """
        raw_predictions = self.predict_single(sequence, num_return=num_return)
        if not raw_predictions:
            return []

        hybrid_list: List[np.ndarray] = []
        for drfold_coords in raw_predictions:
            if drfold_coords.shape != ref_coords.shape:
                # Shape mismatch — fall back to raw DRFold2 output
                hybrid_list.append(drfold_coords)
                continue
            blended = HYBRID_REF_WEIGHT * ref_coords + (1.0 - HYBRID_REF_WEIGHT) * drfold_coords
            hybrid_list.append(blended.astype(np.float32))

        return hybrid_list
