"""
Reproducibility utilities for setting random seeds.

This module ensures deterministic behavior across all random number generators
used in the project (Python, NumPy, PyTorch).
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch backends (cuDNN)
    
    Args:
        seed: Random seed value. Default is 42.
    
    Example:
        >>> from src.utils.reproducibility import set_seed
        >>> set_seed(42)
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")


def get_seed_from_config(config: dict, default: int = 42) -> int:
    """
    Extract seed from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        default: Default seed if not found in config
    
    Returns:
        Seed value
    """
    return config.get('seed', default)


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    Initialize worker processes for DataLoader with different seeds.
    
    This ensures that each DataLoader worker has a different random seed,
    preventing identical data augmentation across workers.
    
    Args:
        worker_id: Worker ID (automatically passed by DataLoader)
        seed: Base seed. If None, uses PyTorch's current seed.
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn
        ... )
    """
    if seed is None:
        seed = torch.initial_seed() % 2**32
    
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
