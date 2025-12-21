"""
PyTorch Dataset class for MVTec AD.

This module provides the dataset implementation for loading MVTec AD images
with proper handling of normal/anomalous images and ground truth masks.
"""
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec AD anomaly detection.
    
    Supports loading images with optional ground truth masks for both
    normal and anomalous samples across train/val/test splits.
    
    Attributes:
        images: List of image file paths
        masks: List of mask file paths (None for normal images)
        labels: List of labels (0=normal, 1=anomalous)
        transform: Callable transform to apply to images and masks
        phase: Dataset phase ('train', 'val', or 'test')
    """
    
    def __init__(
        self,
        images: List[str],
        masks: List[Optional[str]],
        labels: List[int],
        transform: Optional[Callable] = None,
        phase: str = 'train'
    ):
        """
        Initialize MVTec dataset.
        
        Args:
            images: List of image file paths (as strings)
            masks: List of mask file paths (None for normal images)
            labels: List of labels (0=normal, 1=anomalous)
            transform: Transform to apply (should handle image+mask)
            phase: Dataset phase ('train', 'val', 'test')
            
        Raises:
            ValueError: If lists have different lengths or invalid phase
        """
        # Validate inputs
        if not (len(images) == len(masks) == len(labels)):
            raise ValueError(
                f"Images, masks, and labels must have same length. "
                f"Got {len(images)}, {len(masks)}, {len(labels)}"
            )
        
        if phase not in ['train', 'val', 'test']:
            raise ValueError(f"Phase must be 'train', 'val', or 'test'. Got '{phase}'")
        
        # Validate train phase has only normal samples
        if phase == 'train' and any(label != 0 for label in labels):
            raise ValueError("Train phase should only contain normal samples (label=0)")
        
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transform = transform
        self.phase = phase
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask, label, image_path):
            - image: Tensor of shape (3, H, W), normalized
            - mask: Tensor of shape (1, H, W), binary [0, 1], or None
            - label: Integer label (0=normal, 1=anomalous)
            - image_path: String path to the image file
            
        Example:
            >>> dataset = MVTecDataset(...)
            >>> image, mask, label, path = dataset[0]
            >>> print(image.shape)  # torch.Size([3, 224, 224])
            >>> print(label)        # 0 or 1
        """
        # Load image
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Load mask (if available)
        mask_path = self.masks[idx]
        mask = None
        
        if mask_path is not None:
            try:
                # Load mask as grayscale
                mask = Image.open(mask_path).convert('L')
            except Exception as e:
                raise RuntimeError(f"Failed to load mask {mask_path}: {e}")
        
        # Apply transforms
        if self.transform is not None:
            try:
                image, mask = self.transform(image, mask)
            except Exception as e:
                raise RuntimeError(f"Transform failed for {image_path}: {e}")
        
        # Validate that image is not None
        if image is None:
            raise RuntimeError(f"Transform returned None for image {image_path}")
        
        # Get label
        label = self.labels[idx]
        
        return image, mask, label, image_path
    
    def get_stats(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        normal_count = sum(1 for l in self.labels if l == 0)
        anomalous_count = sum(1 for l in self.labels if l == 1)
        mask_count = sum(1 for m in self.masks if m is not None)
        
        return {
            'phase': self.phase,
            'total_samples': len(self),
            'normal_samples': normal_count,
            'anomalous_samples': anomalous_count,
            'anomaly_ratio': anomalous_count / len(self) if len(self) > 0 else 0,
            'masks_available': mask_count
        }
    
    @classmethod
    def from_split(
        cls,
        split_dict: dict,
        transform: Optional[Callable] = None,
        phase: str = 'train'
    ) -> 'MVTecDataset':
        """
        Create dataset from split dictionary.
        
        Constructor for creating dataset directly from
        the split dictionary returned by splitter functions.
        
        Args:
            split_dict: Dictionary with 'images', 'masks', 'labels' keys
            transform: Transform to apply
            phase: Dataset phase
            
        Returns:
            MVTecDataset instance
        """
        return cls(
            images=split_dict['images'],
            masks=split_dict['masks'],
            labels=split_dict['labels'],
            transform=transform,
            phase=phase
        )


def create_dataloaders(
    splits: dict,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, val, test datasets for a single class.
    
    Args:
        splits: Split dictionary for a class (with 'train', 'val', 'test')
        transform: Transform to apply
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Example:
        >>> from src.data.splitter import load_splits
        >>> from src.data.transforms import get_clean_transforms
        >>> 
        >>> splits = load_splits(Path('data/processed/clean_splits.json'))
        >>> train_ds, val_ds, test_ds = create_dataloaders(
        ...     splits['hazelnut'],
        ...     transform=get_clean_transforms(),
        ...     batch_size=32
        ... )
    """
    train_dataset = MVTecDataset.from_split(
        splits['train'],
        transform=transform,
        phase='train'
    )
    
    val_dataset = MVTecDataset.from_split(
        splits['val'],
        transform=transform,
        phase='val'
    )
    
    test_dataset = MVTecDataset.from_split(
        splits['test'],
        transform=transform,
        phase='test'
    )
    
    return train_dataset, val_dataset, test_dataset
