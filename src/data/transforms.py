"""
Data transformation utilities for MVTec AD dataset.

This module provides transform pipelines for both clean and shifted domains,
handling image preprocessing and augmentation with proper mask handling.
"""

from typing import Optional, Tuple

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np


class MVTecTransform:
    """
    Base transform class for MVTec AD dataset.
    
    Handles synchronized transforms for both images and masks (when available).
    """
    
    def __init__(
        self,
        image_size: int = 224,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize transform pipeline.
        
        Args:
            image_size: Target size for images (square resize)
            normalize_mean: ImageNet mean for normalization
            normalize_std: ImageNet std for normalization
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
    
    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transforms to image and mask.
        
        Args:
            image: Input PIL Image
            mask: Optional ground truth mask PIL Image
            
        Returns:
            Tuple of (transformed_image, transformed_mask)
            - image: Tensor of shape (3, H, W), normalized
            - mask: Tensor of shape (1, H, W), binary [0, 1], or None
        """
        raise NotImplementedError("Subclasses must implement __call__")


class CleanDomainTransform(MVTecTransform):
    """
    Transform pipeline for clean domain (Phase 1).
    
    Applies only basic preprocessing:
    - Resize to target size
    - Convert to tensor
    - Normalize with ImageNet statistics
    
    No augmentation applied (deterministic preprocessing).
    """
    
    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply clean domain transforms.
        
        Args:
            image: RGB PIL Image
            mask: Optional grayscale PIL Image (ground truth mask)
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image: (3, 224, 224) normalized float tensor
            - mask: (1, 224, 224) binary float tensor [0, 1] or None
        """
        # Transform image
        image = TF.resize(image, [self.image_size, self.image_size], 
                         interpolation=TF.InterpolationMode.BILINEAR)
        image = TF.to_tensor(image)  # Converts to [0, 1] and (C, H, W)
        image = TF.normalize(image, mean=self.normalize_mean, std=self.normalize_std)
        
        # Transform mask (if provided)
        if mask is not None:
            mask = TF.resize(mask, [self.image_size, self.image_size],
                           interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.to_tensor(mask)  # (1, H, W), values in [0, 1]
            # Binarize mask: threshold at 0.5
            mask = (mask > 0.5).float()
        
        return image, mask


def get_clean_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> CleanDomainTransform:
    """
    Factory function to get clean domain transforms.
    
    This is the transform used for Phase 1 (clean domain) for all splits:
    train, validation, and test.
    
    Args:
        image_size: Target image size (default: 224 for ResNet)
        normalize_mean: ImageNet mean values
        normalize_std: ImageNet std values
        
    Returns:
        CleanDomainTransform instance
        
    Example:
        >>> from src.data.transforms import get_clean_transforms
        >>> transform = get_clean_transforms()
        >>> image_tensor, mask_tensor = transform(pil_image, pil_mask)
    """
    return CleanDomainTransform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )


# Placeholder for future shift transforms (Phase 2)
class ShiftDomainTransform(MVTecTransform):
    """
    Transform pipeline for shifted domain (Phase 2).
    
    Will be implemented in Step 2.1 with:
    - Geometric transforms (rotation, crop, translation)
    - Photometric transforms (color jitter, blur, noise)
    
    TODO: Implement in PHASE 2
    """
    
    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply shift domain transforms."""
        raise NotImplementedError("Shift transforms will be implemented in Phase 2")


def get_shift_transforms(**kwargs) -> ShiftDomainTransform:
    """
    Factory function for shift domain transforms.
    
    TODO: Implement in PHASE 2 (Step 2.1)
    """
    raise NotImplementedError("Shift transforms will be implemented in Phase 2")
