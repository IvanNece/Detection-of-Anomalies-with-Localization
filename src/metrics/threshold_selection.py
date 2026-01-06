"""
Threshold selection and calibration for anomaly detection.

This module provides functions to find the optimal decision threshold
that maximizes F1 score on the validation set. 
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 1000,
    return_all_metrics: bool = False
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Find optimal threshold that maximizes F1 score.
    
    Args:
        scores: Anomaly scores (higher = more anomalous). Shape: (N,)
        labels: Ground truth labels (0=normal, 1=anomalous). Shape: (N,)
        n_thresholds: Number of thresholds to evaluate (default: 1000)
        return_all_metrics: If True, also return metrics at optimal threshold
        
    Returns:
        If return_all_metrics=False: optimal_threshold (float)
        If return_all_metrics=True: Tuple of (optimal_threshold, metrics_dict)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    # Validate inputs
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have same length. Got {len(scores)} and {len(labels)}")
    
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain both normal (0) and anomalous (1) samples")
    
    # Generate threshold candidates
    # Use score range with small margin to include edge cases
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    margin = (score_max - score_min) * 0.01
    
    thresholds = np.linspace(score_min - margin, score_max + margin, n_thresholds)
    
    # Evaluate F1 for each threshold
    best_threshold = thresholds[0]
    best_f1 = 0.0
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        
        # Compute F1 score (handle edge cases)
        if np.sum(predictions) == 0 or np.sum(predictions) == len(predictions):
            f1 = 0.0
        else:
            f1 = f1_score(labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    if return_all_metrics:
        # Compute all metrics at optimal threshold
        predictions = (scores >= best_threshold).astype(int)
        metrics = {
            'threshold': float(best_threshold),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)),
            'accuracy': float(accuracy_score(labels, predictions)),
            'n_samples': len(labels),
            'n_normal': int(np.sum(labels == 0)),
            'n_anomalous': int(np.sum(labels == 1))
        }
        return best_threshold, metrics
    
    return best_threshold


def calibrate_threshold_with_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 1000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Find optimal threshold and return the full F1-threshold curve.
    
    Useful for visualization and analysis of threshold sensitivity.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0=normal, 1=anomalous)
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        Tuple of (optimal_threshold, thresholds_array, f1_scores_array)
        
    Example:
        >>> threshold, thresholds, f1_scores = calibrate_threshold_with_curve(scores, labels)
        >>> plt.plot(thresholds, f1_scores)
        >>> plt.axvline(threshold, color='r', label=f'Optimal: {threshold:.2f}')
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    # Generate threshold candidates
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    margin = (score_max - score_min) * 0.01
    
    thresholds = np.linspace(score_min - margin, score_max + margin, n_thresholds)
    f1_scores = np.zeros(n_thresholds)
    
    for i, threshold in enumerate(thresholds):
        predictions = (scores >= threshold).astype(int)
        f1_scores[i] = f1_score(labels, predictions, zero_division=0)
    
    # Find optimal
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    return best_threshold, thresholds, f1_scores


class ThresholdCalibrator:
    """
    Class for threshold calibration with save/load capabilities.
    
    Stores calibrated thresholds per class and method for easy retrieval
    during evaluation on test sets.
    
    Attributes:
        method_name: Name of the method ('patchcore' or 'padim')
        thresholds: Dictionary mapping class_name -> threshold
        calibration_metrics: Dictionary with calibration statistics
    """
    
    def __init__(self, method_name: str):
        """
        Initialize threshold calibrator.
        
        Args:
            method_name: Name of the method ('patchcore' or 'padim')
        """
        self.method_name = method_name
        self.thresholds: Dict[str, float] = {}
        self.calibration_metrics: Dict[str, Dict] = {}
    
    def calibrate(
        self,
        class_name: str,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        n_thresholds: int = 1000
    ) -> float:
        """
        Calibrate threshold for a specific class.
        
        Args:
            class_name: Name of the class ('hazelnut', 'carpet', 'zipper')
            val_scores: Validation set anomaly scores
            val_labels: Validation set labels
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            Calibrated threshold value
        """
        threshold, metrics = calibrate_threshold(
            val_scores, val_labels, n_thresholds, return_all_metrics=True
        )
        
        self.thresholds[class_name] = threshold
        self.calibration_metrics[class_name] = metrics
        
        return threshold
    
    def get_threshold(self, class_name: str) -> float:
        """Get calibrated threshold for a class."""
        if class_name not in self.thresholds:
            raise ValueError(f"No threshold calibrated for class '{class_name}'")
        return self.thresholds[class_name]
    
    def save(self, output_path: Path) -> None:
        """
        Save calibrated thresholds to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'method': self.method_name,
            'thresholds': self.thresholds,
            'calibration_metrics': self.calibration_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Thresholds saved: {output_path.name}")
    
    @classmethod
    def load(cls, input_path: Path) -> 'ThresholdCalibrator':
        """
        Load calibrated thresholds from JSON file.
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            ThresholdCalibrator instance
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        calibrator = cls(data['method'])
        calibrator.thresholds = data['thresholds']
        calibrator.calibration_metrics = data.get('calibration_metrics', {})
        
        print(f"[OK] Thresholds loaded: {input_path.name}")
        return calibrator
    
    def get_summary(self) -> Dict:
        """Get summary of all calibrated thresholds."""
        return {
            'method': self.method_name,
            'classes': list(self.thresholds.keys()),
            'thresholds': self.thresholds.copy(),
            'calibration_f1': {
                k: v.get('f1', None) 
                for k, v in self.calibration_metrics.items()
            }
        }
    
    def __repr__(self) -> str:
        classes = list(self.thresholds.keys())
        return f"ThresholdCalibrator(method='{self.method_name}', classes={classes})"
