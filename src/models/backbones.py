"""
Feature extraction backbones for PatchCore.

Implements ResNet-50 feature extractor with multi-scale local aggregation
following the PatchCore paper (Roth et al., CVPR 2022).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Tuple


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet-50 Feature Extractor for PatchCore.
    
    Extracts multi-scale features from intermediate layers with local aggregation
    to create locally-aware patch representations.
    
    Args:
        layers: Layer names to extract features from (e.g., ['layer2', 'layer3'])
        patch_size: Neighborhood size for local aggregation (default: 3)
        pretrained: Whether to use ImageNet pre-trained weights
        
    Attributes:
        backbone: Frozen ResNet-50 model
        feature_shapes: Dictionary mapping layer names to feature shapes
        output_dim: Total dimensionality of concatenated features
    """
    
    def __init__(
        self,
        layers: List[str] = ['layer2', 'layer3'],
        patch_size: int = 3,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.layers = layers
        self.patch_size = patch_size
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze all parameters - no training needed
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        
        # Storage for intermediate feature maps
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        
        # Compute output dimensions
        self._compute_output_dims()
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate feature maps."""
        
        def hook_fn(name: str):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(hook_fn(layer_name))
    
    def _compute_output_dims(self) -> None:
        """Compute output feature dimensions via dummy forward pass."""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = self.backbone(dummy_input)
        
        # Store feature shapes (C, H, W)
        self.feature_shapes = {
            name: tuple(feat.shape[1:])
            for name, feat in self.feature_maps.items()
        }
        
        # Total output dimension is sum of channels
        self.output_dim = sum(
            shape[0] for shape in self.feature_shapes.values()
        )
    
    def _apply_local_aggregation(
        self,
        features: torch.Tensor,
        patch_size: int
    ) -> torch.Tensor:
        """
        Apply local aggregation via average pooling over neighborhood.
        
        Implements Eq. 2 from PatchCore paper: pools features from p x p
        neighborhood around each spatial position.
        
        Args:
            features: Feature map (B, C, H, W)
            patch_size: Neighborhood size for pooling
            
        Returns:
            Aggregated features with same spatial dimensions (B, C, H, W)
        """
        # Average pool with stride=1 and padding to preserve spatial dimensions
        pooled = F.avg_pool2d(
            features,
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2
        )
        
        return pooled
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale locally-aware features.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Concatenated multi-scale features (B, C_total, H_out, W_out)
        """
        self.feature_maps.clear()
        
        # Forward pass through backbone (populates feature_maps via hooks)
        with torch.no_grad():
            _ = self.backbone(images)
        
        aggregated_features = []
        target_size = None
        
        # Process each layer
        for layer_name in self.layers:
            feat = self.feature_maps[layer_name]
            
            # Apply local aggregation
            feat_agg = self._apply_local_aggregation(feat, self.patch_size)
            
            # Set target size from highest resolution layer
            if target_size is None:
                target_size = feat_agg.shape[2:]
            
            # Upsample to match target size if needed
            if feat_agg.shape[2:] != target_size:
                feat_agg = F.interpolate(
                    feat_agg,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            aggregated_features.append(feat_agg)
        
        # Concatenate along channel dimension
        output = torch.cat(aggregated_features, dim=1)
        
        return output
    
    def get_patch_features(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patch features as 1D vectors for memory bank construction.
        
        Converts feature maps (B, C, H, W) to patch vectors (B*H*W, C).
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Tuple of:
            - patch_features: Flattened patch vectors (N_patches, C)
            - spatial_dims: Spatial dimensions (H, W)
        """
        features = self.forward(images)  # (B, C, H, W)
        
        B, C, H, W = features.shape
        
        # Reshape: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        patch_features = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        return patch_features, (H, W)


def get_resnet50_feature_extractor(
    layers: List[str] = ['layer2', 'layer3'],
    patch_size: int = 3,
    pretrained: bool = True
) -> ResNet50FeatureExtractor:
    """
    Factory function to create ResNet50FeatureExtractor.
    
    Args:
        layers: Layer names to extract
        patch_size: Neighborhood size for local aggregation
        pretrained: Use ImageNet pre-trained weights
        
    Returns:
        Configured ResNet50FeatureExtractor instance
    """
    return ResNet50FeatureExtractor(
        layers=layers,
        patch_size=patch_size,
        pretrained=pretrained
    )
