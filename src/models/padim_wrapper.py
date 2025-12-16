"""
PaDiM (Probabilistic Anomaly Detection with Multi-scale features) wrapper.

This module provides a clean interface for anomalib's PaDiM implementation,
maintaining consistency with our project structure and enabling easy comparison
with PatchCore.

PaDiM Overview:
- Extracts multi-scale features from pre-trained CNN (ResNet)
- Models normal appearance using Gaussian distributions per spatial location
- Computes Mahalanobis distance for anomaly scoring
- No gradient-based training required (memory-based approach)

Reference:
    Defard et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection
    and Localization" (ICPR 2021)
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# anomalib imports
from anomalib.models import Padim
from anomalib.data import PredictDataset


class PadimWrapper:
    """
    Wrapper for anomalib PaDiM to maintain interface consistency.
    
    This wrapper provides:
    - Consistent interface with other models (e.g., PatchCore)
    - Easy configuration from experiment_config.yaml
    - Training and inference methods aligned with our pipeline
    - Model persistence (save/load)
    
    Attributes:
        model: anomalib Padim instance
        device: torch.device for computation
        backbone: Feature extractor backbone name
        layers: List of layers for multi-scale feature extraction
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
            n_features: Reduced feature dimension (None = no reduction)
            image_size: Input image size (square)
            device: Computation device ('cuda' or 'cpu')
            
        Example:
            >>> model = PadimWrapper(
            ...     backbone='resnet50',
            ...     layers=['layer2', 'layer3'],
            ...     n_features=100,
            ...     device='cuda'
            ... )
        """
        self.backbone = backbone
        self.layers = layers
        self.n_features = n_features
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize anomalib PaDiM model
        self.model = Padim(
            backbone=backbone,
            layers=layers,
            input_size=(image_size, image_size),
            n_features=n_features
        )
        
        self.model.to(self.device)
        self.fitted = False
        
        # Training statistics
        self.training_stats = {
            'num_samples': 0,
            'training_time_seconds': 0.0,
            'memory_bank_size_mb': 0.0
        }
    
    def fit(self, train_loader: torch.utils.data.DataLoader, verbose: bool = True):
        """
        Fit PaDiM on normal training data.
        
        This method:
        1. Extracts features from all training images
        2. Computes Gaussian parameters (mean, covariance) per spatial location
        3. Stores statistics for anomaly detection
        
        Args:
            train_loader: DataLoader with normal training samples
            verbose: Whether to show progress bar
            
        Note:
            PaDiM does NOT require gradient-based optimization.
            This is a single forward pass to compute statistics.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training PaDiM on {len(train_loader.dataset)} normal samples")
            print(f"Backbone: {self.backbone} | Layers: {self.layers}")
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")
        
        self.model.train()
        start_time = time.time()
        
        # Collect all training images
        all_images = []
        for batch in tqdm(train_loader, desc="Loading training data", disable=not verbose):
            images = batch['image'].to(self.device)
            all_images.append(images)
        
        all_images = torch.cat(all_images, dim=0)
        self.training_stats['num_samples'] = len(all_images)
        
        if verbose:
            print(f"\nExtracting features from {len(all_images)} images...")
        
        # Extract features and compute Gaussian parameters
        with torch.no_grad():
            # anomalib PaDiM handles feature extraction and statistical modeling internally
            # We pass the data through the model's feature extraction
            outputs = self.model(all_images)
            
            # PaDiM computes mean and covariance for each spatial location
            # This is handled internally by anomalib
        
        self.fitted = True
        self.training_stats['training_time_seconds'] = time.time() - start_time
        
        # Estimate memory usage
        if hasattr(self.model, 'mean') and self.model.mean is not None:
            # Calculate size of stored statistics
            mean_size = self.model.mean.element_size() * self.model.mean.nelement()
            cov_size = self.model.cov.element_size() * self.model.cov.nelement() if hasattr(self.model, 'cov') else 0
            total_size_mb = (mean_size + cov_size) / (1024 ** 2)
            self.training_stats['memory_bank_size_mb'] = total_size_mb
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ Training completed in {self.training_stats['training_time_seconds']:.2f}s")
            print(f"  Memory bank size: {self.training_stats['memory_bank_size_mb']:.2f} MB")
            print(f"{'='*60}\n")
    
    def predict(
        self,
        image: torch.Tensor,
        return_heatmap: bool = True
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Predict anomaly score and heatmap for a single image.
        
        Args:
            image: Input tensor of shape (3, H, W) or (1, 3, H, W)
            return_heatmap: Whether to return pixel-level anomaly map
        
        Returns:
            Tuple of (image_score, heatmap):
            - image_score: float, image-level anomaly score (higher = more anomalous)
            - heatmap: np.ndarray (H, W), pixel-level anomaly map (or None)
            
        Example:
            >>> image = torch.randn(1, 3, 224, 224)
            >>> score, heatmap = model.predict(image)
            >>> print(f"Anomaly score: {score:.4f}")
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction. Call .fit() first.")
        
        self.model.eval()
        
        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # Get predictions from anomalib PaDiM
            output = self.model(image)
            
            # Extract anomaly score (image-level)
            if isinstance(output, dict):
                image_score = output.get('pred_score', output.get('anomaly_score', 0.0))
                if isinstance(image_score, torch.Tensor):
                    image_score = image_score.item()
                
                # Extract anomaly map (pixel-level)
                if return_heatmap:
                    anomaly_map = output.get('anomaly_map', None)
                    if anomaly_map is not None:
                        if isinstance(anomaly_map, torch.Tensor):
                            heatmap = anomaly_map.squeeze().cpu().numpy()
                        else:
                            heatmap = anomaly_map
                    else:
                        heatmap = None
                else:
                    heatmap = None
            else:
                # Fallback if output is not dict
                image_score = float(output) if isinstance(output, (int, float)) else 0.0
                heatmap = None
        
        return image_score, heatmap
    
    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_heatmaps: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Predict anomaly scores for a batch of images.
        
        Args:
            dataloader: DataLoader with test images
            return_heatmaps: Whether to return all pixel-level heatmaps
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with:
            - 'scores': List of image-level anomaly scores
            - 'labels': List of ground truth labels
            - 'heatmaps': List of heatmaps (if return_heatmaps=True)
            - 'image_paths': List of image paths
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        self.model.eval()
        
        all_scores = []
        all_labels = []
        all_heatmaps = [] if return_heatmaps else None
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", disable=not verbose):
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                paths = batch['image_path']
                
                # Get predictions
                batch_size = images.shape[0]
                for i in range(batch_size):
                    score, heatmap = self.predict(images[i], return_heatmap=return_heatmaps)
                    all_scores.append(score)
                    all_labels.append(labels[i])
                    all_paths.append(paths[i])
                    
                    if return_heatmaps:
                        all_heatmaps.append(heatmap)
        
        results = {
            'scores': np.array(all_scores),
            'labels': np.array(all_labels),
            'image_paths': all_paths
        }
        
        if return_heatmaps:
            results['heatmaps'] = all_heatmaps
        
        return results
    
    def save(self, save_path: Path, include_stats: bool = True):
        """
        Save trained PaDiM model.
        
        Args:
            save_path: Path to save model (.pt or .pth)
            include_stats: Whether to save training statistics
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted model.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'backbone': self.backbone,
            'layers': self.layers,
            'n_features': self.n_features,
            'image_size': self.image_size,
            'fitted': self.fitted,
            'training_stats': self.training_stats
        }
        
        torch.save(checkpoint, save_path)
        
        # Save stats as separate JSON for easy inspection
        if include_stats:
            stats_path = save_path.with_suffix('.json')
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        
        print(f"✓ Model saved to {save_path}")
        if include_stats:
            print(f"  Stats saved to {stats_path}")
    
    def load(self, load_path: Path):
        """
        Load trained PaDiM model.
        
        Args:
            load_path: Path to saved model (.pt or .pth)
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
        self.fitted = checkpoint['fitted']
        self.training_stats = checkpoint.get('training_stats', {})
        
        # Reinitialize model with saved config
        self.model = Padim(
            backbone=self.backbone,
            layers=self.layers,
            input_size=(self.image_size, self.image_size),
            n_features=self.n_features
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        print(f"✓ Model loaded from {load_path}")
    
    def get_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model configuration and training stats
        """
        return {
            'backbone': self.backbone,
            'layers': self.layers,
            'n_features': self.n_features,
            'image_size': self.image_size,
            'device': str(self.device),
            'fitted': self.fitted,
            'training_stats': self.training_stats
        }
