"""
Adapter utilities for anomalib library integration.

This module provides efficient conversion between our MVTecDataset format
and anomalib's expected data structure, enabling seamless use of anomalib models
while maintaining consistency with our data pipeline.

NOTE: This adapter is shared infrastructure that will be used by both PaDiM (Phase 4)
      and PatchCore (Phase 5). When merging branches, coordinate with PatchCore team.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


class AnomalibDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper that converts our MVTec format to anomalib format.
    
    anomalib expects dictionaries with keys:
    - 'image': Tensor (C, H, W)
    - 'mask': Tensor (H, W) or None
    - 'label': int (0=normal, 1=anomalous)
    - 'image_path': str
    - 'mask_path': str or ""
    
    This adapter ensures compatibility while using our existing transforms.
    """
    
    def __init__(
        self,
        images: List[str],
        labels: List[int],
        masks: Optional[List[Optional[str]]] = None,
        transform=None
    ):
        """
        Initialize anomalib-compatible dataset.
        
        Args:
            images: List of image file paths
            labels: List of labels (0=normal, 1=anomalous)
            masks: Optional list of mask file paths (None for normal images)
            transform: Transform callable (should be our MVTecTransform)
        """
        self.images = images
        self.labels = labels
        self.masks = masks if masks is not None else [None] * len(images)
        self.transform = transform
        
        # Validate
        assert len(self.images) == len(self.labels) == len(self.masks), \
            "Images, labels, and masks must have same length"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item in anomalib format.
        
        Returns:
            Dictionary with keys: image, mask, label, image_path, mask_path
        """
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask if available
        mask_path = self.masks[idx]
        mask = None
        if mask_path is not None and Path(mask_path).exists():
            mask = Image.open(mask_path).convert('L')
        
        # Apply transforms
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            # Fallback: basic conversion
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask_tensor = None if mask is None else torch.from_numpy(np.array(mask)).float() / 255.0
        
        # Prepare mask in anomalib format (H, W) not (1, H, W)
        if mask_tensor is not None and mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)  # (1, H, W) → (H, W)
        
        # Prepare output dictionary
        output = {
            'image': image_tensor,  # (3, H, W)
            'label': self.labels[idx],  # 0 or 1
            'image_path': str(image_path),
            'mask_path': str(mask_path) if mask_path is not None else ""
        }
        
        # Add mask only if available
        if mask_tensor is not None:
            output['mask'] = mask_tensor  # (H, W)
        
        return output


def load_split_data(split_json_path: Path, class_name: str, domain: str = 'clean') -> Dict:
    """
    Load split data from JSON file for a specific class.
    
    Args:
        split_json_path: Path to split JSON file (e.g., 'data/processed/clean_splits.json')
        class_name: Name of the class ('hazelnut', 'carpet', 'zipper')
        domain: Domain type ('clean' or 'shifted')
    
    Returns:
        Dictionary with keys: train, val, test, each containing images/masks/labels
        
    Example:
        >>> split_data = load_split_data(
        ...     Path('data/processed/clean_splits.json'),
        ...     'hazelnut',
        ...     'clean'
        ... )
        >>> print(split_data['train']['images'][:2])
    """
    with open(split_json_path, 'r') as f:
        splits = json.load(f)
    
    if class_name not in splits:
        raise ValueError(f"Class '{class_name}' not found in {split_json_path}")
    
    return splits[class_name]


def create_anomalib_dataloader(
    split_data: Dict,
    split_type: str,
    transform,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader from split data in anomalib format.
    
    Args:
        split_data: Dictionary from load_split_data (for one class)
        split_type: 'train', 'val', or 'test'
        transform: Transform to apply (e.g., CleanDomainTransform)
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for CUDA
    
    Returns:
        DataLoader with anomalib-compatible batches
        
    Example:
        >>> from src.data.transforms import CleanDomainTransform
        >>> split_data = load_split_data(Path('data/processed/clean_splits.json'), 'hazelnut')
        >>> transform = CleanDomainTransform()
        >>> train_loader = create_anomalib_dataloader(
        ...     split_data, 'train', transform, batch_size=32, shuffle=False
        ... )
    """
    if split_type not in split_data:
        raise ValueError(f"Split type '{split_type}' not found in split_data")
    
    data = split_data[split_type]
    
    # Extract lists
    images = data['images']
    labels = data['labels']
    masks = data.get('masks', [None] * len(images))  # Default to None if not present
    
    # Create dataset
    dataset = AnomalibDataset(
        images=images,
        labels=labels,
        masks=masks,
        transform=transform
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return dataloader


def collate_fn_anomalib(batch: List[Dict]) -> Dict:
    """
    Custom collate function for anomalib batches.
    
    Handles variable mask availability across samples in a batch.
    
    Args:
        batch: List of dictionaries from AnomalibDataset
    
    Returns:
        Batched dictionary with stacked tensors
    """
    # Stack images and labels
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    
    # Handle masks (may be None for some samples)
    masks = []
    has_masks = False
    for item in batch:
        if 'mask' in item and item['mask'] is not None:
            masks.append(item['mask'])
            has_masks = True
        else:
            # Placeholder: create zero mask with same spatial size as image
            h, w = item['image'].shape[1], item['image'].shape[2]
            masks.append(torch.zeros(h, w))
    
    masks = torch.stack(masks) if has_masks else None
    
    # Collect paths
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'label': labels,
        'image_path': image_paths,
        'mask_path': mask_paths
    }


# NOTE: When PatchCore team implements their feature extraction, coordinate on this utility
def get_split_statistics(split_data: Dict, split_type: str) -> Dict:
    """
    Get statistics for a split (useful for debugging and logging).
    
    Args:
        split_data: Dictionary from load_split_data
        split_type: 'train', 'val', or 'test'
    
    Returns:
        Dictionary with statistics (num_samples, num_normal, num_anomalous)
    """
    data = split_data[split_type]
    labels = data['labels']
    
    return {
        'num_samples': len(labels),
        'num_normal': sum(1 for l in labels if l == 0),
        'num_anomalous': sum(1 for l in labels if l == 1),
        'normal_ratio': sum(1 for l in labels if l == 0) / len(labels) if len(labels) > 0 else 0
    }
