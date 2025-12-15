"""
PatchCore anomaly detection model.

Complete implementation of PatchCore following "Towards Total Recall in 
Industrial Anomaly Detection" (Roth et al., CVPR 2022).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbones import ResNet50FeatureExtractor
from .memory_bank import MemoryBank, GreedyCoresetSubsampling


class PatchCore(nn.Module):
    """
    PatchCore for anomaly detection and localization.
    
    Non-parametric anomaly detection using memory bank of nominal patch features
    with greedy coreset subsampling for efficiency.
    
    Args:
        backbone_layers: Layers to extract (default: ['layer2', 'layer3'])
        patch_size: Neighborhood size for local aggregation
        coreset_ratio: Fraction of patches to retain (0.01 = 1%)
        n_neighbors: Number of neighbors for density reweighting
        device: Device for computation ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        backbone_layers: list = ['layer2', 'layer3'],
        patch_size: int = 3,
        coreset_ratio: float = 0.01,
        n_neighbors: int = 9,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        
        # Feature extractor
        self.backbone = ResNet50FeatureExtractor(
            layers=backbone_layers,
            patch_size=patch_size,
            pretrained=True
        ).to(device)
        
        # Memory bank (initially empty)
        self.memory_bank: Optional[MemoryBank] = None
        
        # Spatial dimensions for heatmap reconstruction
        self.spatial_dims: Optional[Tuple[int, int]] = None
    
    def fit(
        self,
        train_loader: DataLoader,
        apply_coreset: bool = True
    ) -> None:
        """
        Fit PatchCore on normal training data.
        
        Steps:
        1. Extract patch features from all training images
        2. Apply coreset subsampling if requested
        3. Build memory bank
        
        Args:
            train_loader: DataLoader with only normal images
            apply_coreset: Whether to apply coreset subsampling
        """
        self.backbone.eval()
        
        all_features = []
        
        with torch.no_grad():
            for images, _, labels, _ in tqdm(
                train_loader, desc="Extracting features"
            ):
                # Verify only normal images
                assert torch.all(labels == 0), "Training must contain only normal images"
                
                images = images.to(self.device)
                
                # Extract patch features
                patch_features, spatial_dims = self.backbone.get_patch_features(images)
                
                # Save spatial dimensions for heatmap reconstruction
                if self.spatial_dims is None:
                    self.spatial_dims = spatial_dims
                
                all_features.append(patch_features.cpu())
        
        # Concatenate all patches
        all_features = torch.cat(all_features, dim=0)
        
        # Apply coreset subsampling
        if apply_coreset:
            sampler = GreedyCoresetSubsampling(target_ratio=self.coreset_ratio)
            coreset_indices = sampler.sample(all_features)
            memory_features = all_features[coreset_indices]
        else:
            memory_features = all_features
        
        # Build memory bank
        self.memory_bank = MemoryBank(
            features=memory_features.numpy(),
            use_faiss=True,
            n_neighbors=self.n_neighbors
        )
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        return_heatmaps: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict anomaly scores for batch of images.
        
        Args:
            images: Input images (B, 3, H, W)
            return_heatmaps: Whether to return spatial heatmaps
            
        Returns:
            Tuple of (image_scores, heatmaps):
            - image_scores: Per-image scores (B,)
            - heatmaps: Spatial anomaly maps (B, H, W) or None
        """
        assert self.memory_bank is not None, "Must call fit() before predict()"
        
        self.backbone.eval()
        images = images.to(self.device)
        
        B = images.shape[0]
        
        # Extract patch features
        patch_features, (H, W) = self.backbone.get_patch_features(images)
        
        # Compute anomaly scores for each patch
        patch_scores = self.memory_bank.compute_anomaly_scores(
            patch_features,
            apply_reweighting=True
        )
        
        # Reshape to patch-level map: (B, H, W)
        patch_scores_map = patch_scores.reshape(B, H, W)
        
        # Image-level score: maximum over all patches (Eq. 6)
        image_scores = np.max(patch_scores_map.reshape(B, -1), axis=1)
        
        heatmaps = None
        if return_heatmaps:
            # Upsample heatmaps to original resolution
            heatmaps_tensor = torch.from_numpy(patch_scores_map).unsqueeze(1).float()
            heatmaps_tensor = torch.nn.functional.interpolate(
                heatmaps_tensor,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            heatmaps = heatmaps_tensor.squeeze(1).numpy()
        
        return image_scores, heatmaps
    
    def save(self, save_dir: Path, class_name: str, domain: str = 'clean') -> None:
        """
        Save model to disk.
        
        Only memory bank is saved (backbone is frozen pre-trained).
        
        Args:
            save_dir: Directory to save in
            class_name: Class name
            domain: 'clean' or 'shift'
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / f"patchcore_{class_name}_{domain}.npy"
        self.memory_bank.save(str(filepath))
        
        # Save configuration
        config_path = save_dir / f"patchcore_{class_name}_{domain}_config.pth"
        torch.save({
            'spatial_dims': self.spatial_dims,
            'coreset_ratio': self.coreset_ratio,
            'n_neighbors': self.n_neighbors
        }, config_path)
    
    def load(self, save_dir: Path, class_name: str, domain: str = 'clean') -> None:
        """
        Load model from disk.
        
        Args:
            save_dir: Directory to load from
            class_name: Class name
            domain: 'clean' or 'shift'
        """
        save_dir = Path(save_dir)
        
        filepath = save_dir / f"patchcore_{class_name}_{domain}.npy"
        config_path = save_dir / f"patchcore_{class_name}_{domain}_config.pth"
        
        # Load memory bank
        self.memory_bank = MemoryBank.load(
            str(filepath),
            use_faiss=True,
            n_neighbors=self.n_neighbors
        )
        
        # Load configuration
        config = torch.load(config_path)
        self.spatial_dims = config['spatial_dims']
