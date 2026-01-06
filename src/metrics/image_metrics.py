"""
Image-level metrics for anomaly detection evaluation.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUROC).
    
    AUROC measures the model's ability to rank anomalous images higher
    than normal images, regardless of threshold choice.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous). Shape: (N,)
        scores: Anomaly scores (higher = more anomalous). Shape: (N,)
        
    Returns:
        AUROC value in [0, 1]. Higher is better. 0.5 = random guess.
        
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain both normal (0) and anomalous (1) samples")
    
    return float(roc_auc_score(labels, scores))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve (AUPRC).
    
    AUPRC is especially important for imbalanced datasets where
    anomalies are rare. It focuses on the performance for the
    positive (anomalous) class.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous). Shape: (N,)
        scores: Anomaly scores (higher = more anomalous). Shape: (N,)
        
    Returns:
        AUPRC value in [0, 1]. Higher is better.
        Baseline for imbalanced data = positive_rate.
        
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain both normal (0) and anomalous (1) samples")
    
    return float(average_precision_score(labels, scores))


def compute_f1_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> float:
    """
    Compute F1 score at a specific threshold.
    
    F1 is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous). Shape: (N,)
        scores: Anomaly scores (higher = more anomalous). Shape: (N,)
        threshold: Decision threshold (score >= threshold -> anomalous)
        
    Returns:
        F1 score in [0, 1]. Higher is better.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    predictions = (scores >= threshold).astype(int)
    return float(f1_score(labels, predictions, zero_division=0))


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray
) -> Dict[str, float]:
    """
    Compute all classification metrics for binary predictions.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous). Shape: (N,)
        predictions: Binary predictions (0=normal, 1=anomalous). Shape: (N,)
        
    Returns:
        Dictionary with:
        - accuracy: (TP + TN) / (TP + TN + FP + FN)
        - precision: TP / (TP + FP)
        - recall (sensitivity): TP / (TP + FN)
        - specificity: TN / (TN + FP)
        - f1: 2 * precision * recall / (precision + recall)
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    # Compute metrics
    accuracy = float(accuracy_score(labels, predictions))
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    f1 = float(f1_score(labels, predictions, zero_division=0))
    
    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def compute_image_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all image-level metrics for anomaly detection.
    
    This is the main function for image-level evaluation, computing
    both threshold-independent (AUROC, AUPRC) and threshold-dependent
    (F1, Accuracy, Precision, Recall) metrics.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous). Shape: (N,)
        scores: Anomaly scores (higher = more anomalous). Shape: (N,)
        threshold: Decision threshold. If None, only threshold-independent
                  metrics are computed.
        
    Returns:
        Dictionary with all metrics:
        - auroc: Area Under ROC Curve
        - auprc: Area Under Precision-Recall Curve
        - threshold: Decision threshold (if provided)
        - f1: F1 score at threshold (if threshold provided)
        - accuracy: Accuracy at threshold (if threshold provided)
        - precision: Precision at threshold (if threshold provided)
        - recall: Recall at threshold (if threshold provided)
        - specificity: Specificity at threshold (if threshold provided)
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    # Threshold-independent metrics
    metrics = {
        'auroc': compute_auroc(labels, scores),
        'auprc': compute_auprc(labels, scores),
        'n_samples': len(labels),
        'n_normal': int(np.sum(labels == 0)),
        'n_anomalous': int(np.sum(labels == 1))
    }
    
    # Threshold-dependent metrics (if threshold provided)
    if threshold is not None:
        predictions = (scores >= threshold).astype(int)
        classification_metrics = compute_classification_metrics(labels, predictions)
        
        metrics['threshold'] = float(threshold)
        metrics.update(classification_metrics)
    
    return metrics


def compute_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve for visualization.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous)
        scores: Anomaly scores (higher = more anomalous)
        
    Returns:
        Tuple of (fpr, tpr, thresholds):
        - fpr: False Positive Rates
        - tpr: True Positive Rates (Recall)
        - thresholds: Threshold values
        
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def compute_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve for visualization.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous)
        scores: Anomaly scores (higher = more anomalous)
        
    Returns:
        Tuple of (precision, recall, thresholds):
        - precision: Precision values
        - recall: Recall values
        - thresholds: Threshold values
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return precision, recall, thresholds


def compute_confusion_matrix(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Compute confusion matrix at a specific threshold.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomalous)
        scores: Anomaly scores (higher = more anomalous)
        threshold: Decision threshold
        
    Returns:
        Confusion matrix as 2x2 numpy array:
        [[TN, FP],
         [FN, TP]]
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    predictions = (scores >= threshold).astype(int)
    return confusion_matrix(labels, predictions, labels=[0, 1])


def aggregate_metrics(
    per_class_metrics: Dict[str, Dict[str, float]],
    aggregation: str = 'macro'
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple classes.
    
    Args:
        per_class_metrics: Dictionary mapping class names to metric dictionaries
        aggregation: Aggregation method ('macro' = simple average)
        
    Returns:
        Aggregated metrics dictionary
    """
    if aggregation != 'macro':
        raise ValueError(f"Only 'macro' aggregation is supported. Got '{aggregation}'")
    
    if not per_class_metrics:
        return {}
    
    # Get all metric keys from first class
    metric_keys = list(next(iter(per_class_metrics.values())).keys())
    
    # Filter to only numeric metrics
    numeric_keys = []
    for key in metric_keys:
        values = [per_class_metrics[c].get(key) for c in per_class_metrics]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            numeric_keys.append(key)
    
    # Compute macro average
    aggregated = {}
    for key in numeric_keys:
        values = [per_class_metrics[c][key] for c in per_class_metrics if key in per_class_metrics[c]]
        if values:
            aggregated[key] = float(np.mean(values))
    
    return aggregated
