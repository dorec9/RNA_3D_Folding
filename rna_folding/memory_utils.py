import torch
import gc

def clear_vram() -> None:
    """
    Clears CUDA VRAM and forces garbage collection.
    Crucial for Kaggle environments running multiple large models sequentially
    on restricted hardware (T4 16GB).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def optimize_memory_for_t4(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply mixed precision, gradient checkpointing (if fine-tuning), 
    and set parameters for optimal memory footprint on a T4 GPU.
    
    Args:
        model (torch.nn.Module): The PyTorch model to optimize.
        
    Returns:
        torch.nn.Module: The optimized model.
    """
    model.eval()
    if torch.cuda.is_available():
        # Move model to CUDA and cast to float16 or bfloat16 for efficiency
        model = model.half().cuda()
        
    # If the model has gradient checkpointing attributes, enable them here
    # model.gradient_checkpointing_enable()
    return model

def estimate_max_length(vram_gb: int = 16) -> int:
    """
    Heuristic to estimate the maximum sequence length that can be processed
    without raising an OOM error on the current hardware setup.
    
    Args:
        vram_gb (int): Total available VRAM in GB.
        
    Returns:
        int: Maximum safe sequence length limit.
    """
    if vram_gb <= 16:
        return 800  # Fallback for 16GB (Protenix or DRFold2 alone might struggle over 800-1000)
    return 2000 # Fallback for A100 / larger setups
