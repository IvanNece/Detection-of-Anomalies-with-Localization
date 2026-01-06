"""
Metrics package for anomaly detection evaluation.

"""

from .threshold_selection import (
    calibrate_threshold,
    calibrate_threshold_with_curve,
    ThresholdCalibrator
)

from .image_metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1_at_threshold,
    compute_classification_metrics,
    compute_image_metrics,
    compute_roc_curve,
    compute_pr_curve,
    compute_confusion_matrix,
    aggregate_metrics
)

from .pixel_metrics import (
    compute_pixel_auroc,
    compute_pro,
    compute_pixel_metrics,
    compute_pixel_roc_curve,
    aggregate_pixel_metrics
)

__all__ = [
    # Threshold selection
    'calibrate_threshold',
    'calibrate_threshold_with_curve',
    'ThresholdCalibrator',
    
    # Image-level metrics
    'compute_auroc',
    'compute_auprc',
    'compute_f1_at_threshold',
    'compute_classification_metrics',
    'compute_image_metrics',
    'compute_roc_curve',
    'compute_pr_curve',
    'compute_confusion_matrix',
    'aggregate_metrics',
    
    # Pixel-level metrics
    'compute_pixel_auroc',
    'compute_pro',
    'compute_pixel_metrics',
    'compute_pixel_roc_curve',
    'aggregate_pixel_metrics'
]
