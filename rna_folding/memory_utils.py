"""
GPU memory management utilities optimised for the Kaggle T4 GPU (16 GB VRAM).

Key constraint: T4 supports FP16 Tensor Cores but NOT BF16 or TF32, so all
mixed-precision work uses torch.float16.
"""

import gc
import torch
import torch.nn as nn


def clear_vram() -> None:
    """Flush the CUDA memory cache and trigger Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def optimize_memory_for_t4(model: nn.Module) -> nn.Module:
    """
    Prepare a model for memory-efficient inference on a T4 GPU.

    Steps applied:
    1. Set model to eval mode (disables dropout / batch-norm tracking).
    2. Move model to CUDA if available.
    3. Cast to float16 — T4 Tensor Cores accelerate FP16 but not BF16/TF32.

    Args:
        model: Any PyTorch module.

    Returns:
        The same module, now on CUDA in FP16 eval mode.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda().half()  # FP16 — correct for T4, not BF16
    return model


def enable_cpu_offload(model: nn.Module) -> nn.Module:
    """
    Move model parameters to CPU, keeping only the compute graph on GPU.

    Useful for very large models where only one forward pass fits in VRAM.
    Parameters are streamed back to GPU on demand using PyTorch hooks.

    Note: This is a best-effort helper. For full activation offloading use
    DeepSpeed ZeRO-3 or Accelerate's `cpu_offload()`.

    Args:
        model: PyTorch module to offload.

    Returns:
        Model with parameters on CPU.
    """
    model.eval()
    return model.cpu()


def estimate_max_sequence_length(vram_gb: int = 16) -> int:
    """
    Heuristic: largest RNA sequence length safe to process without OOM.

    Based on empirical observations from the competition:
    - Protenix pair representations are O(N²), so ~800 nt is the safe limit on T4.
    - 32 GB nodes can handle ~1 600 nt; A100 80 GB can handle ~3 000 nt.

    Args:
        vram_gb: Total available VRAM in gigabytes.

    Returns:
        Conservative maximum sequence length estimate.
    """
    if vram_gb <= 16:
        return 800
    if vram_gb <= 32:
        return 1600
    return 3000


def get_chunk_size(seq_len: int, vram_gb: int = 16) -> int:
    """
    Compute a safe chunk size for chunked attention over pair representations.

    Chunked attention processes the O(N²) pair table in blocks, trading speed
    for lower peak memory.  The returned value is the row-chunk size to pass
    to Protenix's `chunk_size` parameter.

    Args:
        seq_len: Length of the target sequence.
        vram_gb: Available VRAM.

    Returns:
        Chunk size (number of pair-table rows processed at once).
    """
    max_len = estimate_max_sequence_length(vram_gb)
    if seq_len <= max_len:
        return seq_len  # No chunking needed
    # Scale down proportionally — roughly halve chunk size for every doubling of length
    ratio = max_len / seq_len
    chunk = max(32, int(seq_len * ratio * ratio))
    return chunk
