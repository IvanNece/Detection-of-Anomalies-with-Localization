"""
Data transformation utilities for MVTec AD dataset.

This module provides transform pipelines for both clean and shifted domains,
handling image preprocessing and augmentation with proper mask handling.
"""

import random
from typing import Optional, Tuple, Dict, Any

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


class ShiftDomainTransform:
    """
    Transform pipeline for shifted domain (Phase 2).
    
    Applies realistic domain shift transformations to simulate variations in
    acquisition conditions (illumination, sensor noise, geometric perturbations).
    
    This class handles synchronized geometric transforms for both images and masks,
    and photometric transforms applied only to images.
    
    Attributes:
        image_size: Target image size
        normalize_mean: ImageNet normalization mean
        normalize_std: ImageNet normalization std
        geometric_config: Configuration for geometric transforms
        photometric_config: Configuration for photometric transforms
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        image_size: int = 224,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        geometric_config: Optional[Dict[str, Any]] = None,
        photometric_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize shift domain transform pipeline.
        
        Args:
            image_size: Target size for images (square resize)
            normalize_mean: ImageNet mean for normalization
            normalize_std: ImageNet std for normalization
            geometric_config: Dict with rotation_range, scale_range, etc.
            photometric_config: Dict with brightness_range, blur params, etc.
            seed: Random seed for reproducibility (optional)
            
        Example:
            >>> transform = ShiftDomainTransform(
            ...     geometric_config={'rotation_range': [-10, 10]},
            ...     photometric_config={'brightness_range': [0.7, 1.3]},
            ...     seed=42
            ... )
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.seed = seed
        
        # Default geometric config
        self.geometric_config = geometric_config or {
            'rotation_range': [-10, 10],
            'scale_range': [0.9, 1.0],
            'aspect_ratio_range': [0.9, 1.1],
            'translate_range': 0.1
        }
        
        # Default photometric config
        self.photometric_config = photometric_config or {
            'brightness_range': [0.7, 1.3],
            'contrast_range': [0.7, 1.3],
            'saturation_range': [0.7, 1.3],
            'gaussian_blur': {'kernel_size': [3, 5], 'sigma_range': [0.1, 2.0]},
            'gaussian_noise': {'sigma_range': [0.01, 0.05]}
        }
        
    def _build_geometric_transform(self, rotation: Optional[float] = None, 
                                   translate_x: Optional[float] = None,
                                   translate_y: Optional[float] = None,
                                   scale_x: Optional[float] = None,
                                   scale_y: Optional[float] = None) -> A.Compose:
        """
        Build geometric transform pipeline with deterministic parameters.
        
        Args:
            rotation: Fixed rotation angle in degrees (if None, uses random from config)
            translate_x: Fixed x translation percentage (if None, uses random from config)
            translate_y: Fixed y translation percentage (if None, uses random from config)
            scale_x: Fixed x scale factor (if None, uses random from config)
            scale_y: Fixed y scale factor (if None, uses random from config)
        
        Returns:
            Albumentations Compose pipeline for geometric transforms
        """
        # If parameters provided (deterministic mode), use them directly
        if rotation is not None and translate_x is not None and scale_x is not None:
            return A.Compose([
                A.LongestMaxSize(max_size=int(self.image_size * 1.2)),
                A.PadIfNeeded(
                    min_height=int(self.image_size * 1.2),
                    min_width=int(self.image_size * 1.2),
                    border_mode=cv2.BORDER_REFLECT_101
                ),
                # Deterministic Affine with fixed parameters
                A.Affine(
                    translate_percent={"x": translate_x, "y": translate_y},
                    rotate=rotation,
                    scale={"x": scale_x, "y": scale_y},
                    shear=0,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0
                ),
                A.CenterCrop(
                    height=self.image_size,
                    width=self.image_size
                )
            ])
        else:
            # Random mode - albumentations will sample parameters
            return A.Compose([
                A.LongestMaxSize(max_size=int(self.image_size * 1.2)),
                A.PadIfNeeded(
                    min_height=int(self.image_size * 1.2),
                    min_width=int(self.image_size * 1.2),
                    border_mode=cv2.BORDER_REFLECT_101
                ),
                A.Affine(
                    translate_percent={"x": (-self.geometric_config['translate_range'], self.geometric_config['translate_range']),
                                       "y": (-self.geometric_config['translate_range'], self.geometric_config['translate_range'])},
                    rotate=self.geometric_config['rotation_range'],
                    scale={"x": (self.geometric_config['scale_range'][0], self.geometric_config['scale_range'][1]),
                           "y": (self.geometric_config['scale_range'][0], self.geometric_config['scale_range'][1])},
                    shear=0,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0
                ),
                A.CenterCrop(
                    height=self.image_size,
                    width=self.image_size
                )
            ])
    
    def _build_photometric_transform(self, brightness: Optional[float] = None,
                                     contrast: Optional[float] = None,
                                     saturation: Optional[float] = None,
                                     blur_kernel: Optional[int] = None,
                                     blur_sigma: Optional[float] = None,
                                     apply_blur: bool = True) -> A.Compose:
        """
        Build photometric transform pipeline with optional deterministic parameters.
        
        Args:
            brightness: Fixed brightness factor (if None, uses random from config)
            contrast: Fixed contrast factor (if None, uses random from config)
            saturation: Fixed saturation factor (if None, uses random from config)
            blur_kernel: Fixed blur kernel size (if None, uses random from config)
            blur_sigma: Fixed blur sigma (if None, uses random from config)
            apply_blur: Whether to apply blur (for deterministic control)
        
        Returns:
            Albumentations Compose pipeline for photometric transforms
        """
        if brightness is not None and contrast is not None:
            # Deterministic mode with fixed parameters
            return A.Compose([
                A.ColorJitter(
                    brightness=(brightness, brightness),
                    contrast=(contrast, contrast),
                    saturation=(saturation, saturation),
                    hue=0.0,
                    p=1.0
                ),
                A.GaussianBlur(
                    blur_limit=(blur_kernel, blur_kernel),
                    sigma_limit=(blur_sigma, blur_sigma),
                    p=1.0 if apply_blur else 0.0
                )
            ])
        else:
            # Random mode
            return A.Compose([
                A.ColorJitter(
                    brightness=self.photometric_config['brightness_range'],
                    contrast=self.photometric_config['contrast_range'],
                    saturation=self.photometric_config['saturation_range'],
                    hue=0.0,
                    p=0.8
                ),
                A.GaussianBlur(
                    blur_limit=tuple(self.photometric_config['gaussian_blur']['kernel_size']),
                    sigma_limit=tuple(self.photometric_config['gaussian_blur']['sigma_range']),
                    p=0.5
                )
            ])
    
    def _apply_photometric_transforms(self, image: np.ndarray,
                                      brightness: Optional[float] = None,
                                      contrast: Optional[float] = None,
                                      saturation: Optional[float] = None,
                                      blur_kernel: Optional[int] = None,
                                      blur_sigma: Optional[float] = None,
                                      apply_blur: bool = True,
                                      noise_sigma: Optional[float] = None,
                                      apply_noise: bool = True) -> np.ndarray:
        """
        Apply photometric transforms to image only (not mask).
        
        Args:
            image: RGB image as numpy array [0, 1] float32
            brightness, contrast, saturation: Fixed color jitter params (None = random)
            blur_kernel, blur_sigma: Fixed blur params (None = random)
            apply_blur: Whether to apply blur
            noise_sigma: Fixed noise sigma (None = random)
            apply_noise: Whether to apply noise
            
        Returns:
            Transformed image as numpy array [0, 1] float32
        """
        # Build pipeline with deterministic or random params
        photometric = self._build_photometric_transform(
            brightness, contrast, saturation, blur_kernel, blur_sigma, apply_blur
        )
        
        transformed = photometric(image=image)
        image = transformed['image']
        
        # Add Gaussian noise
        if apply_noise:
            if noise_sigma is None:
                # Random mode
                sigma = random.uniform(*self.photometric_config['gaussian_noise']['sigma_range'])
            else:
                # Deterministic mode
                sigma = noise_sigma
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply shift domain transforms to image and mask.
        
        Pipeline:
        1. Convert PIL to numpy
        2. Apply geometric transforms (image + mask synchronized)
        3. Apply photometric transforms (image only)
        4. Convert to tensor and normalize
        
        Args:
            image: RGB PIL Image
            mask: Optional grayscale PIL Image (ground truth mask)
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image: (3, H, W) normalized float tensor
            - mask: (1, H, W) binary float tensor [0, 1] or None
            
        Example:
            >>> transform = ShiftDomainTransform(seed=42)
            >>> image_shifted, mask_shifted = transform(pil_image, pil_mask)
        """
        # Set seed for reproducibility if provided
        # When seed is set, sample transform parameters ONCE to ensure reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
            # Sample geometric parameters deterministically
            rotation = random.uniform(*self.geometric_config['rotation_range'])
            translate_x = random.uniform(-self.geometric_config['translate_range'], 
                                        self.geometric_config['translate_range'])
            translate_y = random.uniform(-self.geometric_config['translate_range'],
                                        self.geometric_config['translate_range'])
            scale_x = random.uniform(*self.geometric_config['scale_range'])
            scale_y = random.uniform(*self.geometric_config['scale_range'])
            
            # Sample photometric parameters deterministically
            brightness = random.uniform(*self.photometric_config['brightness_range'])
            contrast = random.uniform(*self.photometric_config['contrast_range'])
            saturation = random.uniform(*self.photometric_config['saturation_range'])
            blur_kernel = random.choice(self.photometric_config['gaussian_blur']['kernel_size'])
            blur_sigma = random.uniform(*self.photometric_config['gaussian_blur']['sigma_range'])
            apply_blur = random.random() < 0.5
            noise_sigma = random.uniform(*self.photometric_config['gaussian_noise']['sigma_range'])
            apply_noise = random.random() < 0.5
            
            # Build deterministic transforms
            geometric_transform = self._build_geometric_transform(
                rotation, translate_x, translate_y, scale_x, scale_y
            )
        else:
            # Random mode - albumentations will sample parameters
            rotation = None
            geometric_transform = self._build_geometric_transform()
            brightness = contrast = saturation = None
            blur_kernel = blur_sigma = noise_sigma = None
            apply_blur = apply_noise = True
        
        # Convert PIL to numpy
        image_np = np.array(image).astype(np.float32) / 255.0  # [0, 1]
        
        if mask is not None:
            mask_np = np.array(mask).astype(np.float32)
            # Ensure mask is 2D (H, W) for albumentations
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            # Normalize to [0, 1] and binarize
            mask_np = (mask_np / 255.0 > 0.5).astype(np.float32)
        else:
            mask_np = None
        
        # Apply geometric transforms (synchronized for image and mask)
        if mask_np is not None:
            transformed = geometric_transform(image=image_np, mask=mask_np)
            image_np = transformed['image']
            mask_np = transformed['mask']
        else:
            transformed = geometric_transform(image=image_np)
            image_np = transformed['image']
        
        # Apply photometric transforms (image only)
        if self.seed is not None:
            # Deterministic mode
            image_np = self._apply_photometric_transforms(
                image_np, brightness, contrast, saturation,
                blur_kernel, blur_sigma, apply_blur,
                noise_sigma, apply_noise
            )
        else:
            # Random mode
            image_np = self._apply_photometric_transforms(image_np)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Normalize with ImageNet stats
        image_tensor = TF.normalize(image_tensor, mean=self.normalize_mean, std=self.normalize_std)
        
        # Process mask
        if mask_np is not None:
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (H, W) -> (1, H, W)
            # Ensure binary [0, 1]
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = None
        
        return image_tensor, mask_tensor


def get_shift_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    geometric_config: Optional[Dict[str, Any]] = None,
    photometric_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> ShiftDomainTransform:
    """
    Factory function to get shift domain transforms.
    
    This is the transform used for Phase 2 (domain shift) to simulate
    realistic variations in acquisition conditions.
    
    Args:
        image_size: Target image size (default: 224 for ResNet)
        normalize_mean: ImageNet mean values
        normalize_std: ImageNet std values
        geometric_config: Configuration dict for geometric transforms
        photometric_config: Configuration dict for photometric transforms
        seed: Random seed for reproducibility
        
    Returns:
        ShiftDomainTransform instance
        
    Example:
        >>> from src.data.transforms import get_shift_transforms
        >>> from src.utils.config import Config
        >>> 
        >>> config = Config.load('configs/experiment_config.yaml')
        >>> transform = get_shift_transforms(
        ...     image_size=config.dataset.image_size,
        ...     geometric_config=config.domain_shift.geometric,
        ...     photometric_config=config.domain_shift.photometric,
        ...     seed=config.seed
        ... )
        >>> image_shifted, mask_shifted = transform(pil_image, pil_mask)
    """
    return ShiftDomainTransform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        geometric_config=geometric_config,
        photometric_config=photometric_config,
        seed=seed
    )
