"""
Complete evaluator for anomaly detection methods.

This module provides an Evaluator class that orchestrates the complete
evaluation pipeline:
1. Threshold calibration on validation set
2. Image-level evaluation on test set
3. Pixel-level evaluation on test set
4. Results aggregation and saving

The evaluator supports both PatchCore and PaDiM methods with a unified interface.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import metrics modules
from ..metrics.threshold_selection import calibrate_threshold, ThresholdCalibrator
from ..metrics.image_metrics import (
    compute_image_metrics,
    compute_roc_curve,
    compute_pr_curve,
    aggregate_metrics
)
from ..metrics.pixel_metrics import (
    compute_pixel_metrics,
    aggregate_pixel_metrics
)


class Evaluator:
    """
    Complete evaluator for anomaly detection methods.
    
    This class provides a unified interface for evaluating both PatchCore
    and PaDiM methods, handling threshold calibration, metric computation,
    and result aggregation.
    
    Workflow:
        1. calibrate_threshold() - Find F1-optimal threshold on validation set
        2. evaluate_image_level() - Compute image-level metrics on test set
        3. evaluate_pixel_level() - Compute pixel-level metrics on test set
        4. save_results() - Save all results to JSON
    
    Attributes:
        method_name: Name of the method ('patchcore' or 'padim')
        class_name: Name of the class ('hazelnut', 'carpet', 'zipper')
        domain: Domain type ('clean' or 'shift')
        threshold: Calibrated threshold value
        image_metrics: Dictionary of image-level metrics
        pixel_metrics: Dictionary of pixel-level metrics
        
    Example:
        >>> evaluator = Evaluator('patchcore', 'hazelnut', 'clean')
        >>> evaluator.calibrate_threshold(val_scores, val_labels)
        >>> evaluator.evaluate_image_level(test_scores, test_labels)
        >>> evaluator.evaluate_pixel_level(test_masks, test_heatmaps)
        >>> evaluator.save_results(Path('outputs/results/'))
    """
    
    def __init__(
        self,
        method_name: str,
        class_name: str,
        domain: str = 'clean'
    ):
        """
        Initialize evaluator.
        
        Args:
            method_name: Name of the method ('patchcore' or 'padim')
            class_name: Name of the class ('hazelnut', 'carpet', 'zipper')
            domain: Domain type ('clean' or 'shift')
        """
        self.method_name = method_name.lower()
        self.class_name = class_name.lower()
        self.domain = domain.lower()
        
        # Results storage
        self.threshold: Optional[float] = None
        self.calibration_metrics: Dict[str, float] = {}
        self.image_metrics: Dict[str, float] = {}
        self.pixel_metrics: Dict[str, float] = {}
        
        # Curve data for visualization
        self.roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.pr_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        
        # Metadata
        self.timestamp = datetime.now().isoformat()
    
    def calibrate_threshold(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        n_thresholds: int = 1000
    ) -> float:
        """
        Calibrate threshold on validation set using F1-optimal approach.
        
        This implements the threshold selection protocol from the Project Proposal:
        "Explore a range of thresholds and select the one that maximizes F1 on Val-clean."
        
        Args:
            val_scores: Validation set anomaly scores
            val_labels: Validation set labels (0=normal, 1=anomalous)
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            Calibrated threshold value
            
        Example:
            >>> threshold = evaluator.calibrate_threshold(val_scores, val_labels)
            >>> print(f"Optimal threshold: {threshold:.4f}")
        """
        val_scores = np.asarray(val_scores)
        val_labels = np.asarray(val_labels)
        
        # Calibrate threshold
        self.threshold, self.calibration_metrics = calibrate_threshold(
            val_scores, val_labels, n_thresholds, return_all_metrics=True
        )
        
        print(f"[{self.class_name}] Calibrated threshold: {self.threshold:.4f}")
        print(f"  Validation F1: {self.calibration_metrics['f1']:.4f}")
        print(f"  Samples: {len(val_labels)} ({sum(val_labels==0)} normal, {sum(val_labels==1)} anomalous)")
        
        return self.threshold
    
    def evaluate_image_level(
        self,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
        threshold: Optional[float] = None,
        compute_curves: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate image-level detection on test set.
        
        Computes all image-level metrics defined in the Project Proposal:
        - AUROC, AUPRC (threshold-independent)
        - F1, Accuracy, Precision, Recall (at threshold)
        
        Args:
            test_scores: Test set anomaly scores
            test_labels: Test set labels (0=normal, 1=anomalous)
            threshold: Decision threshold. If None, uses calibrated threshold.
            compute_curves: If True, also compute ROC and PR curves for visualization.
            
        Returns:
            Dictionary of image-level metrics
            
        Raises:
            ValueError: If no threshold is available (not calibrated)
        """
        test_scores = np.asarray(test_scores)
        test_labels = np.asarray(test_labels)
        
        # Use provided threshold or calibrated threshold
        if threshold is not None:
            eval_threshold = threshold
        elif self.threshold is not None:
            eval_threshold = self.threshold
        else:
            raise ValueError("No threshold available. Call calibrate_threshold() first or provide threshold.")
        
        # Compute all image-level metrics
        self.image_metrics = compute_image_metrics(
            test_labels, test_scores, threshold=eval_threshold
        )
        
        # Compute curves for visualization
        if compute_curves:
            self.roc_curve = compute_roc_curve(test_labels, test_scores)
            self.pr_curve = compute_pr_curve(test_labels, test_scores)
        
        print(f"[{self.class_name}] Image-level evaluation:")
        print(f"  AUROC: {self.image_metrics['auroc']:.4f}")
        print(f"  AUPRC: {self.image_metrics['auprc']:.4f}")
        print(f"  F1: {self.image_metrics['f1']:.4f}")
        print(f"  Accuracy: {self.image_metrics['accuracy']:.4f}")
        
        return self.image_metrics
    
    def evaluate_pixel_level(
        self,
        test_masks: List[np.ndarray],
        test_heatmaps: List[np.ndarray],
        compute_pro: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate pixel-level localization on test set.
        
        Computes pixel-level metrics defined in the Project Proposal:
        - Pixel AUROC
        - PRO (Per-Region Overlap)
        
        Args:
            test_masks: List of ground truth masks, each (H, W) binary.
                       Can include None for normal images.
            test_heatmaps: List of predicted anomaly maps, each (H, W) float.
            compute_pro: If True, also compute PRO metric (slower).
            
        Returns:
            Dictionary of pixel-level metrics
        """
        self.pixel_metrics = compute_pixel_metrics(
            test_masks, test_heatmaps, compute_pro_metric=compute_pro
        )
        
        print(f"[{self.class_name}] Pixel-level evaluation:")
        if self.pixel_metrics.get('pixel_auroc') is not None:
            print(f"  Pixel AUROC: {self.pixel_metrics['pixel_auroc']:.4f}")
        if self.pixel_metrics.get('pro') is not None:
            print(f"  PRO: {self.pixel_metrics['pro']:.4f}")
        
        return self.pixel_metrics
    
    def get_results(self) -> Dict:
        """
        Get all evaluation results as dictionary.
        
        Returns:
            Dictionary with all metrics and metadata
        """
        return {
            'method': self.method_name,
            'class': self.class_name,
            'domain': self.domain,
            'timestamp': self.timestamp,
            'threshold': self.threshold,
            'calibration_metrics': self.calibration_metrics,
            'image_level': self.image_metrics,
            'pixel_level': self.pixel_metrics
        }
    
    def save_results(
        self,
        output_dir: Path,
        prefix: str = ''
    ) -> Path:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_dir: Output directory
            prefix: Optional prefix for filename
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        if prefix:
            filename = f"{prefix}_{self.method_name}_{self.class_name}_{self.domain}_results.json"
        else:
            filename = f"{self.method_name}_{self.class_name}_{self.domain}_results.json"
        
        output_path = output_dir / filename
        
        # Save results
        results = self.get_results()
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[OK] Results saved: {output_path.name}")
        return output_path
    
    def __repr__(self) -> str:
        threshold_str = f"{self.threshold:.4f}" if self.threshold else "None"
        return f"Evaluator({self.method_name}, {self.class_name}, {self.domain}, threshold={threshold_str})"


class MultiClassEvaluator:
    """
    Evaluator for multiple classes with result aggregation.
    
    This class manages evaluation across all classes (hazelnut, carpet, zipper)
    and computes macro-averaged metrics.
    
    Attributes:
        method_name: Name of the method ('patchcore' or 'padim')
        domain: Domain type ('clean' or 'shift')
        evaluators: Dictionary mapping class names to Evaluator instances
    """
    
    def __init__(
        self,
        method_name: str,
        domain: str = 'clean',
        classes: List[str] = ['hazelnut', 'carpet', 'zipper']
    ):
        """
        Initialize multi-class evaluator.
        
        Args:
            method_name: Name of the method ('patchcore' or 'padim')
            domain: Domain type ('clean' or 'shift')
            classes: List of class names to evaluate
        """
        self.method_name = method_name.lower()
        self.domain = domain.lower()
        self.classes = classes
        
        # Create evaluator for each class
        self.evaluators: Dict[str, Evaluator] = {
            class_name: Evaluator(method_name, class_name, domain)
            for class_name in classes
        }
        
        # Aggregated results
        self.macro_image_metrics: Dict[str, float] = {}
        self.macro_pixel_metrics: Dict[str, float] = {}
        
        # Metadata
        self.timestamp = datetime.now().isoformat()
    
    def get_evaluator(self, class_name: str) -> Evaluator:
        """Get evaluator for a specific class."""
        if class_name not in self.evaluators:
            raise ValueError(f"Unknown class: {class_name}")
        return self.evaluators[class_name]
    
    def aggregate_results(self) -> Dict[str, float]:
        """
        Compute macro-averaged metrics across all classes.
        
        Returns:
            Dictionary with aggregated metrics
        """
        # Collect per-class image metrics
        per_class_image = {
            class_name: evaluator.image_metrics
            for class_name, evaluator in self.evaluators.items()
            if evaluator.image_metrics
        }
        
        if per_class_image:
            self.macro_image_metrics = aggregate_metrics(per_class_image)
        
        # Collect per-class pixel metrics
        per_class_pixel = {
            class_name: evaluator.pixel_metrics
            for class_name, evaluator in self.evaluators.items()
            if evaluator.pixel_metrics
        }
        
        if per_class_pixel:
            self.macro_pixel_metrics = aggregate_pixel_metrics(per_class_pixel)
        
        print(f"\n{'='*60}")
        print(f"MACRO-AVERAGE RESULTS ({self.method_name.upper()} - {self.domain.upper()})")
        print(f"{'='*60}")
        
        if self.macro_image_metrics:
            print(f"Image-level:")
            for key, value in self.macro_image_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        if self.macro_pixel_metrics:
            print(f"Pixel-level:")
            for key, value in self.macro_pixel_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        return {
            'image_level': self.macro_image_metrics,
            'pixel_level': self.macro_pixel_metrics
        }
    
    def get_all_results(self) -> Dict:
        """
        Get all results (per-class and macro-averaged).
        
        Returns:
            Dictionary with complete results structure
        """
        results = {
            'metadata': {
                'method': self.method_name,
                'domain': self.domain,
                'timestamp': self.timestamp,
                'classes': self.classes
            },
            'per_class': {
                class_name: evaluator.get_results()
                for class_name, evaluator in self.evaluators.items()
            },
            'macro_average': {
                'image_level': self.macro_image_metrics,
                'pixel_level': self.macro_pixel_metrics
            }
        }
        
        return results
    
    def save_results(
        self,
        output_path: Path
    ) -> Path:
        """
        Save all results to a single JSON file.
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = self.get_all_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n[OK] All results saved: {output_path.name}")
        return output_path
    
    def __repr__(self) -> str:
        return f"MultiClassEvaluator({self.method_name}, {self.domain}, classes={self.classes})"


def evaluate_model_on_dataloader(
    model,
    dataloader: DataLoader,
    device: str = 'cuda',
    return_heatmaps: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a model (PatchCore or PaDiM) on a DataLoader.
    
    This is a convenience function that runs model.predict() on all images
    in the dataloader and collects the results.
    
    Args:
        model: Model with predict() method (PatchCore or PadimWrapper)
        dataloader: DataLoader with test images
        device: Device for computation
        return_heatmaps: If True, also collect pixel-level heatmaps
        verbose: If True, show progress bar
        
    Returns:
        Dictionary with:
        - 'scores': np.ndarray of image-level scores
        - 'labels': np.ndarray of ground truth labels
        - 'heatmaps': List of heatmaps (if return_heatmaps=True)
        - 'masks': List of ground truth masks
        - 'paths': List of image paths
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    all_heatmaps = []
    all_masks = []
    all_paths = []
    
    with torch.no_grad():
        for images, masks, labels, paths in tqdm(
            dataloader, 
            desc="Evaluating", 
            disable=not verbose
        ):
            # Move to device
            images = images.to(device)
            
            # Get predictions
            scores, heatmaps = model.predict(images, return_heatmaps=return_heatmaps)
            
            # Collect results
            all_scores.extend(scores.flatten().tolist() if isinstance(scores, np.ndarray) and scores.ndim > 0 else [scores])
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
            
            # Collect masks (convert to numpy)
            for mask in masks:
                if mask is not None:
                    all_masks.append(mask.numpy().squeeze())
                else:
                    all_masks.append(None)
            
            # Collect heatmaps
            if return_heatmaps and heatmaps is not None:
                if heatmaps.ndim == 2:  # Single image
                    all_heatmaps.append(heatmaps)
                else:  # Batch
                    all_heatmaps.extend([h for h in heatmaps])
    
    results = {
        'scores': np.array(all_scores),
        'labels': np.array(all_labels),
        'masks': all_masks,
        'paths': all_paths
    }
    
    if return_heatmaps:
        results['heatmaps'] = all_heatmaps
    
    return results
