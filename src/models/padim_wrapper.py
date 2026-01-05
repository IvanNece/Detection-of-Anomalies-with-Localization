"""
PaDiM wrapper.

This module provides a clean interface for anomalib's PaDiM implementation,
maintaining consistency with our project structure (PatchCore interface) and 
enabling easy comparison between methods.

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
import numpy as np
from tqdm import tqdm

# anomalib imports
try:
    from anomalib.models.image.padim.torch_model import PadimModel
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    print("Warning: anomalib not available. Install with: pip install anomalib")


class PadimWrapper(nn.Module):
    """
    Wrapper for anomalib PaDiM with PatchCore-compatible interface.
        
    Attributes:
        model: PadimModel (anomalib's PyTorch model)
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
        
        # Initialize anomalib PadimModel - uses native implementation
        self.model = PadimModel(
            backbone=backbone,
            layers=layers,
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
    
    def fit(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        verbose: bool = True
    ) -> None:
        """
        Fit PaDiM on normal training data using anomalib's native implementation.
        
        This method:
        1. Sets model to training mode
        2. Runs forward pass on all training images (accumulates embeddings in memory bank)
        3. Calls model.fit() to compute Gaussian parameters (mean, covariance)
        
        Args:
            train_loader: DataLoader with normal training samples (only label=0)
                         Expected batch format: (images, masks, labels, paths)
            verbose: Whether to show progress bar
            
        Note:
            PaDiM does NOT require gradient-based optimization.
            This is a statistical approach (like PatchCore's memory bank).
        
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training PaDiM on {len(train_loader.dataset)} normal samples")
            print(f"Backbone: {self.backbone} | Layers: {self.layers}")
            print(f"N features: {self.n_features}")
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")
        
        # Set model to training mode - this makes forward() accumulate in memory_bank
        self.model.train()
        start_time = time.time()
        
        num_samples = 0
        
        with torch.no_grad():
            for images, _, labels, _ in tqdm(
                train_loader, 
                desc="Extracting features", 
                disable=not verbose
            ):
                # Verify only normal images
                assert torch.all(labels == 0), "Training must contain only normal images (label=0)"
                
                images = images.to(self.device)
                
                # Forward pass in training mode accumulates embeddings in memory_bank
                # This is anomalib's native behavior
                self.model(images)
                
                num_samples += images.shape[0]
        
        if verbose:
            print(f"\nFitting Gaussian distributions...")
            print(f"  Total samples: {num_samples}")
            print(f"  Memory bank size: {len(self.model.memory_bank)} batches")
        
        # Fit Gaussian to memory bank using anomalib's native method
        # This computes mean and inverse covariance for Mahalanobis distance
        self.model.fit()
        
        self.fitted = True
        self.training_stats['num_samples'] = num_samples
        self.training_stats['training_time_seconds'] = time.time() - start_time
        
        # Estimate memory usage from Gaussian parameters
        mean_size = self.model.gaussian.mean.element_size() * self.model.gaussian.mean.nelement()
        cov_size = self.model.gaussian.inv_covariance.element_size() * self.model.gaussian.inv_covariance.nelement()
        self.training_stats['memory_bank_size_mb'] = (mean_size + cov_size) / (1024 ** 2)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] Training completed in {self.training_stats['training_time_seconds']:.2f}s")
            print(f"  Gaussian mean shape: {self.model.gaussian.mean.shape}")
            print(f"  Inv covariance shape: {self.model.gaussian.inv_covariance.shape}")
            print(f"  Memory usage: {self.training_stats['memory_bank_size_mb']:.2f} MB")
            print(f"{'='*60}\n")
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        return_heatmaps: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict anomaly scores using anomalib's native forward pass.
        
        Args:
            images: Input images (B, 3, H, W) or (3, H, W) for single image
            return_heatmaps: Whether to return spatial heatmaps
            
        Returns:
            Tuple of (image_scores, heatmaps):
            - image_scores: Per-image scores (B,) - higher = more anomalous
            - heatmaps: Spatial anomaly maps (B, H_out, W_out) or None
            
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction. Call .fit() first.")
        
        # Set model to eval mode - this makes forward() compute anomaly maps
        self.model.eval()
        
        # Handle single image input
        if images.ndim == 3:
            images = images.unsqueeze(0)
            single_image = True
        else:
            single_image = False
        
        images = images.to(self.device)
        
        # Forward pass in eval mode returns InferenceBatch with pred_score and anomaly_map
        # This uses anomalib's native AnomalyMapGenerator with Mahalanobis distance
        output = self.model(images)
        
        # Extract scores and heatmaps from InferenceBatch
        image_scores = output.pred_score.cpu().numpy()
        
        heatmaps = None
        if return_heatmaps:
            heatmaps = output.anomaly_map.cpu().numpy()
            # Remove channel dimension if present (B, 1, H, W) -> (B, H, W)
            if heatmaps.ndim == 4 and heatmaps.shape[1] == 1:
                heatmaps = heatmaps.squeeze(1)
        
        # Handle single image output
        if single_image:
            image_scores = image_scores[0] if image_scores.ndim > 0 else float(image_scores)
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
        
        self.model.eval()
        
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
                
                # Handle both single and batch outputs
                if isinstance(scores, np.ndarray):
                    if scores.ndim == 0:
                        all_scores.append(float(scores))
                    else:
                        all_scores.extend(scores.flatten().tolist())
                else:
                    all_scores.append(float(scores))
                
                all_labels.extend(labels.cpu().numpy().tolist())
                all_paths.extend(paths)
                
                if return_heatmaps and heatmaps is not None:
                    if heatmaps.ndim == 2:  # Single image
                        all_heatmaps.append(heatmaps)
                    else:  # Batch
                        all_heatmaps.extend([h for h in heatmaps])
        
        results = {
            'scores': np.array(all_scores),
            'labels': np.array(all_labels),
            'image_paths': all_paths
        }
        
        if return_heatmaps:
            results['heatmaps'] = all_heatmaps
        
        return results
    
    def save(self, save_path: Path, include_stats: bool = True) -> None:
        """
        Save trained PaDiM model.
        
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
        
        # Prepare checkpoint with Gaussian parameters and configuration
        checkpoint = {
            # Gaussian parameters (from anomalib's MultiVariateGaussian)
            'gaussian_mean': self.model.gaussian.mean.cpu(),
            'gaussian_inv_covariance': self.model.gaussian.inv_covariance.cpu(),
            
            # Random feature indices (important for reproducibility)
            'idx': self.model.idx.cpu(),
            
            # Configuration
            'backbone': self.backbone,
            'layers': self.layers,
            'n_features': self.n_features,
            'image_size': self.image_size,
            
            # Metadata
            'fitted': self.fitted,
            'training_stats': self.training_stats
        }
        
        torch.save(checkpoint, save_path)
        
        # Save stats as separate JSON for easy inspection
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
        Load trained PaDiM model.
        
        Args:
            load_path: Path to saved model (.pt or .pth)
            
        Example:
            >>> model = PadimWrapper(device='cuda')
            >>> model.load(Path('outputs/models/padim_hazelnut_clean.pt'))
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        # Restore configuration
        self.backbone = checkpoint['backbone']
        self.layers = checkpoint['layers']
        self.n_features = checkpoint.get('n_features', None)
        self.image_size = checkpoint['image_size']
        self.fitted = checkpoint['fitted']
        self.training_stats = checkpoint.get('training_stats', {})
        
        # Reinitialize model with saved config
        self.model = PadimModel(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=True,
            n_features=self.n_features
        ).to(self.device)
        
        # Restore random feature indices (critical for reproducibility)
        self.model.idx = checkpoint['idx'].to(self.device)
        
        # Restore Gaussian parameters
        self.model.gaussian.mean = checkpoint['gaussian_mean'].to(self.device)
        self.model.gaussian.inv_covariance = checkpoint['gaussian_inv_covariance'].to(self.device)
        
        print(f"[OK] Model loaded: {load_path.name}")
    
    def get_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model configuration and training stats
            
        Example:
            >>> info = model.get_info()
            >>> print(info['training_stats'])
        """
        info = {
            'model_type': 'PaDiM',
            'backbone': self.backbone,
            'layers': self.layers,
            'n_features': self.n_features,
            'image_size': self.image_size,
            'device': str(self.device),
            'fitted': self.fitted,
            'training_stats': self.training_stats,
            'distance_metric': 'mahalanobis',
            'implementation': 'anomalib_native'
        }
        
        if self.fitted:
            info['gaussian_mean_shape'] = list(self.model.gaussian.mean.shape)
            info['gaussian_inv_cov_shape'] = list(self.model.gaussian.inv_covariance.shape)
        
        return info
