"""
Reproducibility utilities for setting random seeds.

This module ensures deterministic behavior across all random number generators
used in the project (Python, NumPy, PyTorch).
"""

import os
import random
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

