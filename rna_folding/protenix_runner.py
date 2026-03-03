"""
Protenix inference wrapper for RNA 3D structure prediction.

Protenix is ByteDance's open-source reproduction of AlphaFold 3.  This
module wraps the command-line entry point and, when the library is installed,
the Python API.

Key optimisations for the Kaggle T4 GPU (16 GB VRAM):
- FP16 precision (T4 has FP16 Tensor Cores; BF16 / TF32 are not supported)
- Chunked pair-representation attention (``chunk_size`` parameter)
- CPU offloading of model weights between runs via ``memory_utils``

References:
    Protenix GitHub: github.com/bytedance/Protenix
    d4t4 RNA-MSA integration: Kaggle discussion
    RNAPro post-competition model: Kaggle notebook by theoviel
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from rna_folding.memory_utils import (
    clear_vram,
    estimate_max_sequence_length,
    get_chunk_size,
)

logger = logging.getLogger(__name__)


class ProtenixRunner:
    """
    Wrapper for Protenix v1.0.0+ inference on RNA targets.

    When ``protenix_weights_path`` is a valid directory the runner attempts to
    use the Protenix Python API directly.  If the import fails (weights not
    present, library not installed) it falls back to subprocess invocation.
    Both paths produce a list of C1' coordinate arrays.
    """

    def __init__(
        self,
        protenix_weights_path: str,
        use_rna_msa: bool = True,
        num_recycling_steps: int = 3,
        num_diffusion_samples: int = 5,
        vram_gb: int = 16,
    ) -> None:
        """
        Args:
            protenix_weights_path: Path to Protenix model weights directory.
            use_rna_msa: Enable RNA MSA integration (d4t4 innovation that
                         significantly boosts template-poor target accuracy).
            num_recycling_steps: Evoformer recycling iterations.  3 is the
                                 AlphaFold 3 default; reduce to 1 for speed.
            num_diffusion_samples: Number of structure samples drawn during
                                   the diffusion step.  Maps directly to the
                                   number of returned structures.
            vram_gb: Available VRAM — used to set safe chunk sizes.
        """
        self.protenix_weights_path = protenix_weights_path
        self.use_rna_msa = use_rna_msa
        self.num_recycling_steps = num_recycling_steps
        self.num_diffusion_samples = num_diffusion_samples
        self.vram_gb = vram_gb
        self._model = None  # Lazy-initialised Python API handle

    # ------------------------------------------------------------------
    # Model loading (lazy, to avoid VRAM consumption until needed)
    # ------------------------------------------------------------------

    def _try_load_model(self):
        """Attempt to import and load Protenix; return None on failure."""
        if self._model is not None:
            return self._model
        try:
            from protenix.model.protenix import Protenix  # noqa: F401
            import torch

            weights_path = self.protenix_weights_path
            if os.path.isfile(weights_path) and weights_path.endswith(".pt"):
                # Direct checkpoint file (e.g. 1599_ema_0.999.pt from Kaggle dataset)
                model = Protenix()
                state = torch.load(weights_path, map_location="cpu")
                # Checkpoint may be nested under a 'model' or 'state_dict' key
                if isinstance(state, dict):
                    sd = state.get("model", state.get("state_dict", state))
                else:
                    sd = state
                model.load_state_dict(sd, strict=False)
            else:
                # Directory with from_pretrained support
                model = Protenix.from_pretrained(weights_path)

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda().half()  # FP16 — correct for T4
            self._model = model
            logger.info("Protenix loaded from %s", weights_path)
            return model
        except Exception as exc:
            logger.warning("Protenix Python API unavailable (%s). Will use subprocess.", exc)
            return None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict_via_api(
        self,
        sequence: str,
        msa_str: str,
        template_coords: Optional[np.ndarray],
        num_return: int,
    ) -> List[np.ndarray]:
        """Run Protenix through the Python API (fast path)."""
        import torch

        model = self._try_load_model()
        if model is None:
            return []

        seq_len = len(sequence)
        chunk_size = get_chunk_size(seq_len, self.vram_gb)

        try:
            with torch.no_grad():
                results = model.predict(
                    sequence=sequence,
                    msa=msa_str if self.use_rna_msa else "",
                    template_coords=template_coords,
                    num_recycling_steps=self.num_recycling_steps,
                    num_diffusion_samples=num_return,
                    chunk_size=chunk_size,
                )
            # Expected output: list of dicts with key 'c1_prime_coords' → (N, 3)
            coords_list = []
            for r in results[:num_return]:
                if isinstance(r, dict) and "c1_prime_coords" in r:
                    coords_list.append(np.array(r["c1_prime_coords"], dtype=np.float32))
                elif isinstance(r, np.ndarray) and r.shape == (seq_len, 3):
                    coords_list.append(r.astype(np.float32))
            return coords_list
        except Exception as exc:
            logger.warning("Protenix API inference failed: %s", exc)
            return []
        finally:
            clear_vram()

    def _predict_via_subprocess(
        self,
        sequence: str,
        msa_path: str,
        output_dir: str,
        num_return: int,
    ) -> List[np.ndarray]:
        """
        Run Protenix via its command-line entry point.

        Expects the ``protenix`` binary to be on PATH or the weights directory
        to contain a ``run.sh`` launcher.
        """
        fasta_path = os.path.join(output_dir, "query.fasta")
        with open(fasta_path, "w") as fh:
            fh.write(f">query\n{sequence}\n")

        cmd = [
            "protenix", "predict",
            "--fasta", fasta_path,
            "--output_dir", output_dir,
            "--weights", self.protenix_weights_path,
            "--num_diffusion_samples", str(num_return),
            "--num_recycling_steps", str(self.num_recycling_steps),
            "--precision", "fp16",           # T4 requires FP16, not BF16
        ]
        if self.use_rna_msa and msa_path and os.path.exists(msa_path):
            cmd += ["--msa", msa_path]

        logger.debug("Running Protenix subprocess: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Protenix subprocess failed: %s", exc)
            return []

        # Collect output coordinate files (expected: query_sample_N.npy or .cif)
        coords_list: List[np.ndarray] = []
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith(".npy") and "c1_prime" in fname:
                arr = np.load(os.path.join(output_dir, fname)).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    coords_list.append(arr)
        return coords_list[:num_return]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_structures(
        self,
        sequence: str,
        msa_str: str = "",
        msa_path: str = "",
        template_coords: Optional[np.ndarray] = None,
        num_return_sequences: int = 3,
    ) -> List[np.ndarray]:
        """
        Generate RNA 3D structure candidates using Protenix.

        Tries the Python API first; falls back to subprocess if unavailable.
        Returns an empty list rather than raising if both paths fail, so the
        pipeline can continue with TBM-only predictions.

        Args:
            sequence: Target RNA sequence.
            msa_str: Raw A3M MSA content (used with Python API).
            msa_path: Path to .a3m file (used with subprocess).
            template_coords: Optional ``(N, 3)`` C1' coordinate array from TBM
                             to provide as a structural template hint.
            num_return_sequences: Number of structure samples to generate.

        Returns:
            List of ``(len(sequence), 3)`` float32 arrays, up to
            ``num_return_sequences`` entries.
        """
        max_safe = estimate_max_sequence_length(self.vram_gb)
        if len(sequence) > max_safe:
            logger.warning(
                "Sequence length %d exceeds safe limit %d for %d GB VRAM. "
                "Using aggressive chunking.",
                len(sequence), max_safe, self.vram_gb,
            )

        # Try Python API first (lower overhead)
        results = self._predict_via_api(sequence, msa_str, template_coords, num_return_sequences)
        if results:
            return results

        # Fallback: subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self._predict_via_subprocess(
                sequence, msa_path, tmpdir, num_return_sequences
            )

        return results
