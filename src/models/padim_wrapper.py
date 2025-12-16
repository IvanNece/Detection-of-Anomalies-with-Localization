"""
PaDiM (Probabilistic Anomaly Detection with Multi-scale features) wrapper.

This module provides a clean interface for anomalib's PaDiM implementation,
maintaining consistency with our project structure (PatchCore interface) and 
enabling easy comparison between methods.

PaDiM Overview:
- Extracts multi-scale features from pre-trained CNN (ResNet)
- Models normal appearance using multivariate Gaussian distributions per spatial location
- Computes Mahalanobis distance for anomaly scoring
- No gradient-based training required (statistical approach like PatchCore)

Key Implementation Notes:
- Interface aligned with PatchCore for fair comparison
- Uses anomalib's Padim implementation under the hood
- Batch processing for efficiency
- Consistent save/load format

Reference:
    Defard et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection
    and Localization" (ICPR 2021)
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# anomalib imports (dynamically import to avoid dependency issues)
try:
    from anomalib.models.image.padim import Padim
    from anomalib.models.image.padim.torch_model import PadimModel
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    print("Warning: anomalib not available. Install with: pip install anomalib")


class PadimWrapper(nn.Module):
    """
    Wrapper for anomalib PaDiM with PatchCore-compatible interface.
    
    This wrapper provides:
    - Consistent interface with PatchCore for fair comparison
    - Batch processing capabilities
    - Easy configuration from experiment_config.yaml
    - Training and inference methods aligned with our pipeline
    - Model persistence (save/load)
    
    Interface Design:
    - fit(train_loader) - train on normal samples
    - predict(images, return_heatmaps) - batch prediction (same as PatchCore)
    - save/load - model persistence
    
    Attributes:
        model_torch: PadimModel (anomalib's underlying PyTorch model)
        device: torch.device for computation
        backbone: Feature extractor backbone name
        layers: List of layers for multi-scale feature extraction
        n_features: Reduced feature dimension
        fitted: Whether model has been trained
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        layers: List[str] = ['layer1', 'layer2', 'layer3'],
        n_features: Optional[int] = 100,
        image_size: int = 224,
        device: str = 'cuda'
    ):
        """
        Initialize PaDiM wrapper.
        
        Args:
            backbone: CNN backbone ('resnet18', 'resnet50', 'wide_resnet50_2')
            layers: List of layers to extract features from (multi-scale)
            n_features: Reduced feature dimension (None = no reduction, uses all features)
            image_size: Input image size (square)
            device: Computation device ('cuda' or 'cpu')
            
        Example:
            >>> model = PadimWrapper(
            ...     backbone='resnet50',
            ...     layers=['layer1', 'layer2', 'layer3'],
            ...     n_features=100,
            ...     device='cuda'
            ... )
        """
        super().__init__()
        
        if not ANOMALIB_AVAILABLE:
            raise ImportError(
                "anomalib is required for PaDiM. Install with: pip install anomalib"
            )
        
        self.backbone = backbone
        self.layers = layers
        self.n_features = n_features
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize anomalib PaDiM's underlying torch model
        # Note: We use PadimModel directly for more control
        self.model_torch = PadimModel(
            layers=layers,
            backbone=backbone,
            pre_trained=True,
            n_features=n_features
        ).to(self.device)
        
        self.fitted = False
        
        # Training statistics (aligned with PatchCore)
        self.training_stats = {
            'num_samples': 0,
            'training_time_seconds': 0.0,
            'memory_bank_size_mb': 0.0
        }
        
        # Spatial dimensions for heatmap reconstruction
        self.spatial_dims: Optional[Tuple[int, int]] = None
    
    def fit(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        verbose: bool = True
    ) -> None:
        """
        Fit PaDiM on normal training data (ALIGNED WITH PATCHCORE INTERFACE).
        
        This method:
        1. Extracts multi-scale features from all training images
        2. Computes Gaussian parameters (mean, covariance) per spatial location
        3. Stores statistics for anomaly detection
        
        Args:
            train_loader: DataLoader with normal training samples (only label=0)
                         Expected batch format: (images, masks, labels, paths)
            verbose: Whether to show progress bar
            
        Note:
            PaDiM does NOT require gradient-based optimization.
            This is a statistical approach (like PatchCore's memory bank).
        
        Example:
            >>> from torch.utils.data import DataLoader
            >>> train_loader = DataLoader(train_dataset, batch_size=32)
            >>> model.fit(train_loader, verbose=True)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training PaDiM on {len(train_loader.dataset)} normal samples")
            print(f"Backbone: {self.backbone} | Layers: {self.layers}")
            print(f"N features: {self.n_features}")
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")
        
        self.model_torch.train()
        start_time = time.time()
        
        # Collect all embeddings from training images
        embeddings_list = []
        
        with torch.no_grad():
            for images, _, labels, _ in tqdm(
                train_loader, 
                desc="Extracting features", 
                disable=not verbose
            ):
                # Verify only normal images
                assert torch.all(labels == 0), "Training must contain only normal images (label=0)"
                
                images = images.to(self.device)
                
                # Extract patch embeddings (multi-scale features)
                features = self.model_torch.feature_extractor(images)
                embeddings = self.model_torch.generate_embedding(features)
                embeddings_list.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)
        B, C, H, W = all_embeddings.shape
        
        self.training_stats['num_samples'] = B
        self.spatial_dims = (H, W)
        
        if verbose:
            print(f"\nComputing Gaussian distributions...")
            print(f"  Embedding shape: {all_embeddings.shape}")
            print(f"  Spatial resolution: {H}x{W}")
        
        # Compute mean and covariance per spatial location
        # Reshape: (B, C, H, W) -> (H*W, B, C)
        embeddings_reshaped = all_embeddings.permute(2, 3, 0, 1).reshape(H * W, B, C)
        
        # Compute statistics
        mean = torch.mean(embeddings_reshaped, dim=1)  # (H*W, C)
        
        # Covariance with regularization
        embeddings_centered = embeddings_reshaped - mean.unsqueeze(1)
        cov = torch.einsum('lbc,lbd->lcd', embeddings_centered, embeddings_centered) / (B - 1)
        
        # Add regularization (identity matrix * epsilon)
        identity = torch.eye(C, device=self.device).unsqueeze(0).repeat(H * W, 1, 1)
        cov = cov + 0.01 * identity
        
        # Store in model
        self.model_torch.mean = mean.reshape(H, W, C)  # (H, W, C)
        self.model_torch.cov = cov.reshape(H, W, C, C)  # (H, W, C, C)
        self.model_torch.cov_inv = torch.linalg.inv(cov).reshape(H, W, C, C)
        
        self.fitted = True
        self.training_stats['training_time_seconds'] = time.time() - start_time
        
        # Estimate memory usage
        mean_size_mb = self.model_torch.mean.element_size() * self.model_torch.mean.nelement() / (1024 ** 2)
        cov_size_mb = self.model_torch.cov.element_size() * self.model_torch.cov.nelement() / (1024 ** 2)
        self.training_stats['memory_bank_size_mb'] = mean_size_mb + cov_size_mb
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] Training completed in {self.training_stats['training_time_seconds']:.2f}s")
            print(f"  Memory bank size: {self.training_stats['memory_bank_size_mb']:.2f} MB")
            print(f"  Gaussian distributions: {H}x{W} spatial locations")
            print(f"{'='*60}\n")
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        return_heatmaps: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict anomaly scores for batch of images (ALIGNED WITH PATCHCORE).
        
        Args:
            images: Input images (B, 3, H, W) or (3, H, W) for single image
            return_heatmaps: Whether to return spatial heatmaps
            
        Returns:
            Tuple of (image_scores, heatmaps):
            - image_scores: Per-image scores (B,) - higher = more anomalous
            - heatmaps: Spatial anomaly maps (B, H_out, W_out) or None
            
        Example:
            >>> images = torch.randn(4, 3, 224, 224)
            >>> scores, heatmaps = model.predict(images, return_heatmaps=True)
            >>> print(scores.shape)  # (4,)
            >>> print(heatmaps.shape)  # (4, 224, 224)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction. Call .fit() first.")
        
        self.model_torch.eval()
        
        # Handle single image input
        if images.ndim == 3:
            images = images.unsqueeze(0)
            single_image = True
        else:
            single_image = False
        
        images = images.to(self.device)
        B, _, H_img, W_img = images.shape
        
        # Extract embeddings (multi-scale features concatenated)
        features = self.model_torch.feature_extractor(images)
        embeddings = self.model_torch.generate_embedding(features)  # (B, C, H, W)
        _, C, H, W = embeddings.shape
        
        assert self.spatial_dims == (H, W), f"Spatial dims mismatch: {self.spatial_dims} vs {(H, W)}"
        
        # Reshape embeddings: (B, C, H, W) -> (B, H*W, C)
        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Compute Mahalanobis distance for each patch
        # Distance = (x - mu)^T * Sigma^{-1} * (x - mu)
        mean = self.model_torch.mean.reshape(H * W, C)  # (H*W, C)
        cov_inv = self.model_torch.cov_inv.reshape(H * W, C, C)  # (H*W, C, C)
        
        # Vectorized computation
        diff = embeddings_flat - mean.unsqueeze(0)  # (B, H*W, C)
        
        # Mahalanobis distance: d^2 = diff^T * Sigma^{-1} * diff
        distances = torch.zeros(B, H * W, device=self.device)
        for i in range(H * W):
            temp = torch.matmul(diff[:, i, :].unsqueeze(1), cov_inv[i])  # (B, 1, C)
            distances[:, i] = torch.sum(temp * diff[:, i, :].unsqueeze(1), dim=2).squeeze()
        
        # Reshape to spatial map: (B, H*W) -> (B, H, W)
        distance_maps = distances.reshape(B, H, W)
        
        # Image-level score: maximum distance (Eq. 6 in paper)
        image_scores = torch.max(distance_maps.reshape(B, -1), dim=1)[0]
        image_scores = image_scores.cpu().numpy()
        
        # Heatmaps: upsample to input resolution
        heatmaps = None
        if return_heatmaps:
            # Upsample (B, H, W) -> (B, H_img, W_img)
            distance_maps_upsampled = F.interpolate(
                distance_maps.unsqueeze(1),  # Add channel dim: (B, 1, H, W)
                size=(H_img, W_img),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Remove channel: (B, H_img, W_img)
            
            heatmaps = distance_maps_upsampled.cpu().numpy()
        
        # Handle single image output
        if single_image:
            image_scores = image_scores[0]
            if heatmaps is not None:
                heatmaps = heatmaps[0]
        
        return image_scores, heatmaps
    
    def predict_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_heatmaps: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Predict anomaly scores for entire DataLoader (convenience method).
        
        Args:
            dataloader: DataLoader with test images
                       Expected batch format: (images, masks, labels, paths)
            return_heatmaps: Whether to return all pixel-level heatmaps
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with:
            - 'scores': np.ndarray of image-level anomaly scores
            - 'labels': np.ndarray of ground truth labels
            - 'heatmaps': List of heatmaps (if return_heatmaps=True)
            - 'image_paths': List of image paths
            
        Example:
            >>> results = model.predict_dataloader(test_loader, return_heatmaps=True)
            >>> print(results['scores'].shape)  # (N,)
            >>> print(len(results['heatmaps']))  # N
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        self.model_torch.eval()
        
        all_scores = []
        all_labels = []
        all_heatmaps = [] if return_heatmaps else None
        all_paths = []
        
        with torch.no_grad():
            for images, _, labels, paths in tqdm(
                dataloader, 
                desc="Predicting", 
                disable=not verbose
            ):
                # Batch prediction
                scores, heatmaps = self.predict(images, return_heatmaps=return_heatmaps)
                
                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy())
                all_paths.extend(paths)
                
                if return_heatmaps:
                    # Handle single vs batch
                    if heatmaps.ndim == 2:  # Single image
                        all_heatmaps.append(heatmaps)
                    else:  # Batch
                        all_heatmaps.extend(heatmaps)
        
        # Concatenate results
        all_scores = np.concatenate(all_scores) if len(all_scores) > 0 else np.array([])
        all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
        
        results = {
            'scores': all_scores,
            'labels': all_labels,
            'image_paths': all_paths
        }
        
        if return_heatmaps:
            results['heatmaps'] = all_heatmaps
        
        return results
    
    def save(self, save_path: Path, include_stats: bool = True) -> None:
        """
        Save trained PaDiM model (ALIGNED WITH PATCHCORE).
        
        Args:
            save_path: Path to save model (.pt or .pth)
            include_stats: Whether to save training statistics as separate JSON
            
        Example:
            >>> model.save(Path('outputs/models/padim_hazelnut_clean.pt'))
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted model. Call .fit() first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint (store Gaussian parameters + config)
        checkpoint = {
            # Model parameters
            'mean': self.model_torch.mean.cpu(),
            'cov': self.model_torch.cov.cpu(),
            'cov_inv': self.model_torch.cov_inv.cpu(),
            
            # Configuration
            'backbone': self.backbone,
            'layers': self.layers,
            'n_features': self.n_features,
            'image_size': self.image_size,
            'spatial_dims': self.spatial_dims,
            
            # Metadata
            'fitted': self.fitted,
            'training_stats': self.training_stats
        }
        
        torch.save(checkpoint, save_path)
        
        # Save stats as separate JSON for easy inspection (same as PatchCore)
        if include_stats:
            stats_path = save_path.with_suffix('.json')
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        
        if include_stats:
            print(f"[OK] Model saved: {save_path.name}")
            print(f"  Stats saved: {stats_path.name}")
        else:
            print(f"[OK] Model saved: {save_path.name}")
    
    def load(self, load_path: Path) -> None:
        """
        Load trained PaDiM model (ALIGNED WITH PATCHCORE).
        
        Args:
            load_path: Path to saved model (.pt or .pth)
            
        Example:
            >>> model = PadimWrapper(device='cuda')
            >>> model.load(Path('outputs/models/padim_hazelnut_clean.pt'))
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Restore configuration
        self.backbone = checkpoint['backbone']
        self.layers = checkpoint['layers']
        self.n_features = checkpoint.get('n_features', None)
        self.image_size = checkpoint['image_size']
        self.spatial_dims = checkpoint.get('spatial_dims', None)
        self.fitted = checkpoint['fitted']
        self.training_stats = checkpoint.get('training_stats', {})
        
        # Reinitialize model with saved config
        self.model_torch = PadimModel(
            layers=self.layers,
            backbone=self.backbone,
            pre_trained=True,
            n_features=self.n_features
        ).to(self.device)
        
        # Restore Gaussian parameters
        self.model_torch.mean = checkpoint['mean'].to(self.device)
        self.model_torch.cov = checkpoint['cov'].to(self.device)
        self.model_torch.cov_inv = checkpoint['cov_inv'].to(self.device)
        
        print(f"[OK] Model loaded: {load_path.name}")
    
    def get_info(self) -> Dict:
        """
        Get model information and statistics (ALIGNED WITH PATCHCORE).
        
        Returns:
            Dictionary with model configuration and training stats
            
        Example:
            >>> info = model.get_info()
            >>> print(info['training_stats'])
        """
        return {
            'model_type': 'PaDiM',
            'backbone': self.backbone,
            'layers': self.layers,
            'n_features': self.n_features,
            'image_size': self.image_size,
            'spatial_dims': self.spatial_dims,
            'device': str(self.device),
            'fitted': self.fitted,
            'training_stats': self.training_stats,
            'distance_metric': 'mahalanobis'
        }
