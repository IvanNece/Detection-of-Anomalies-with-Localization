"""
Data package for MVTec AD dataset handling.

This package provides utilities for:
- Dataset loading and preprocessing (dataset.py)
- Data splitting (splitter.py)
- Image transformations (transforms.py)
"""

from .dataset import MVTecDataset, create_dataloaders
from .splitter import (
    create_clean_split,
    create_all_clean_splits,
    save_splits,
    load_splits,
    verify_split
)
from .transforms import (
    get_clean_transforms,
    get_shift_transforms,
    CleanDomainTransform,
    ShiftDomainTransform
)

__all__ = [
    # Dataset
    'MVTecDataset',
    'create_dataloaders',
    # Splitter
    'create_clean_split',
    'create_all_clean_splits',
    'save_splits',
    'load_splits',
    'verify_split',
    # Transforms
    'get_clean_transforms',
    'get_shift_transforms',
    'CleanDomainTransform',
    'ShiftDomainTransform',
]
