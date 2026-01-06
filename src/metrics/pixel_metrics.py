"""
Pixel-level metrics for anomaly localization evaluation.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score, roc_curve


def compute_pixel_auroc(
    masks_true: List[np.ndarray],
    heatmaps: List[np.ndarray]
) -> float:
    """
    Compute pixel-level AUROC for anomaly localization.
    
    This metric evaluates how well the model discriminates between
    defective pixels and non-defective pixels across all images.
    
    Algorithm:
        1. Flatten all ground truth masks and heatmaps
        2. Compute ROC AUC on the combined pixel arrays
    
    Args:
        masks_true: List of ground truth masks, each (H, W) binary [0, 1].
                   0 = normal pixel, 1 = anomalous pixel.
                   Can include None for normal images (will be treated as all zeros).
        heatmaps: List of predicted anomaly maps, each (H, W) float.
                 Higher values = more anomalous.
        
    Returns:
        Pixel-level AUROC in [0, 1]. Higher is better.
    """
    if len(masks_true) != len(heatmaps):
        raise ValueError(
            f"Number of masks ({len(masks_true)}) must equal number of heatmaps ({len(heatmaps)})"
        )
    
    if len(masks_true) == 0:
        raise ValueError("Empty masks_true and heatmaps lists")
    
    # Collect all pixels
    all_labels = []
    all_scores = []
    
    for mask, heatmap in zip(masks_true, heatmaps):
        # Handle None masks (normal images with no defects)
        if mask is None:
            mask = np.zeros_like(heatmap)
        
        # Validate shapes match
        if mask.shape != heatmap.shape:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match heatmap shape {heatmap.shape}"
            )
        
        # Flatten and collect
        all_labels.append(mask.flatten())
        all_scores.append(heatmap.flatten())
    
    # Concatenate all pixels
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    
    # Binarize labels (threshold at 0.5)
    all_labels = (all_labels > 0.5).astype(int)
    
    # Check if we have both classes
    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        # Only one class present
        if unique_labels[0] == 0:
            # All normal pixels - no anomalies to evaluate
            return 1.0  # Perfect (no false positives possible)
        else:
            # All anomalous pixels - shouldn't happen in practice
            return 1.0
    
    return float(roc_auc_score(all_labels, all_scores))


def compute_pro(
    masks_true: List[np.ndarray],
    heatmaps: List[np.ndarray],
    n_thresholds: int = 200,
    fpr_integration_limit: float = 0.3
) -> float:
    """
    Compute Per-Region Overlap (PRO) metric.
    
    PRO is defined in the MVTec AD paper as the area under the PRO curve,
    integrated up to a false positive rate limit (default 0.3).
    
    For each connected component in the ground truth mask, PRO computes
    the overlap with the predicted anomaly region at various thresholds,
    then averages across all components.
    
    Algorithm:
        For each threshold t:
            1. Binarize heatmaps at threshold t
            2. For each connected component in ground truth:
               - Compute IoU (overlap) with predicted anomaly region
            3. Average IoU across all components -> PRO at t
        4. Compute FPR at each threshold
        5. Integrate PRO curve up to FPR = 0.3
        6. Normalize by integration limit
    
    Args:
        masks_true: List of ground truth masks, each (H, W) binary [0, 1]
        heatmaps: List of predicted anomaly maps, each (H, W) float
        n_thresholds: Number of thresholds to evaluate
        fpr_integration_limit: Upper FPR limit for integration (default: 0.3)
        
    Returns:
        PRO score in [0, 1]. Higher is better.
        
    Reference:
        Bergmann et al., "The MVTec AD Dataset", CVPR 2019
    """
    if len(masks_true) != len(heatmaps):
        raise ValueError(
            f"Number of masks ({len(masks_true)}) must equal number of heatmaps ({len(heatmaps)})"
        )
    
    if len(masks_true) == 0:
        raise ValueError("Empty masks_true and heatmaps lists")
    
    # Collect all valid masks and heatmaps (skip None masks)
    valid_masks = []
    valid_heatmaps = []
    
    for mask, heatmap in zip(masks_true, heatmaps):
        if mask is not None and np.any(mask > 0.5):
            valid_masks.append((mask > 0.5).astype(np.float32))
            valid_heatmaps.append(heatmap)
    
    if len(valid_masks) == 0:
        # No anomalous masks to evaluate
        return 1.0  # Perfect score (no false negatives possible)
    
    # Collect all anomaly scores to determine threshold range
    all_scores = np.concatenate([h.flatten() for h in valid_heatmaps])
    all_gt = np.concatenate([m.flatten() for m in valid_masks])
    
    # All normal pixels for FPR computation
    all_normal_pixels = np.concatenate([
        h.flatten()[m.flatten() < 0.5]
        for m, h in zip(valid_masks, valid_heatmaps)
    ])
    
    # Generate thresholds
    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    thresholds = np.linspace(score_min, score_max, n_thresholds)
    
    # Precompute connected components for each mask
    component_info = []
    for mask in valid_masks:
        labeled_mask, num_components = connected_components(mask > 0.5)
        component_info.append((labeled_mask, num_components))
    
    # Compute PRO at each threshold
    pro_values = []
    fpr_values = []
    
    for threshold in thresholds:
        # Compute Per-Region Overlap for this threshold
        region_overlaps = []
        
        for idx, (mask, heatmap) in enumerate(zip(valid_masks, valid_heatmaps)):
            labeled_mask, num_components = component_info[idx]
            
            # Binarize prediction
            pred_mask = (heatmap >= threshold).astype(float)
            
            # Compute overlap for each connected component
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_mask == component_id).astype(float)
                
                # Intersection: pixels that are in both component and prediction
                intersection = np.sum(component_mask * pred_mask)
                
                # Component area
                component_area = np.sum(component_mask)
                
                if component_area > 0:
                    overlap = intersection / component_area
                    region_overlaps.append(overlap)
        
        # Average overlap across all regions
        if region_overlaps:
            pro = np.mean(region_overlaps)
        else:
            pro = 0.0
        
        pro_values.append(pro)
        
        # Compute FPR (false positive rate on normal pixels)
        if len(all_normal_pixels) > 0:
            fpr = np.mean(all_normal_pixels >= threshold)
        else:
            fpr = 0.0
        
        fpr_values.append(fpr)
    
    pro_values = np.array(pro_values)
    fpr_values = np.array(fpr_values)
    
    # Sort by FPR for integration
    sorted_indices = np.argsort(fpr_values)
    fpr_sorted = fpr_values[sorted_indices]
    pro_sorted = pro_values[sorted_indices]
    
    # Remove duplicates for proper integration
    # Keep only points where FPR changes
    unique_indices = [0]
    for i in range(1, len(fpr_sorted)):
        if fpr_sorted[i] != fpr_sorted[unique_indices[-1]]:
            unique_indices.append(i)
    
    fpr_unique = fpr_sorted[unique_indices]
    pro_unique = pro_sorted[unique_indices]
    
    # Filter to FPR <= integration limit
    valid_mask = fpr_unique <= fpr_integration_limit
    
    if np.sum(valid_mask) < 2:
        # Not enough points for integration
        return float(pro_unique[valid_mask].mean() if np.any(valid_mask) else 0.0)
    
    fpr_filtered = fpr_unique[valid_mask]
    pro_filtered = pro_unique[valid_mask]
    
    # Integrate using trapezoidal rule
    auc = float(np.trapz(pro_filtered, fpr_filtered))
    
    # Normalize by integration limit
    normalized_pro = auc / fpr_integration_limit
    
    return normalized_pro


def compute_pixel_metrics(
    masks_true: List[np.ndarray],
    heatmaps: List[np.ndarray],
    compute_pro_metric: bool = True
) -> Dict[str, float]:
    """
    Compute all pixel-level metrics.
    
    This is the main function for pixel-level evaluation, computing
    both Pixel AUROC and PRO metrics.
    
    Args:
        masks_true: List of ground truth masks, each (H, W) binary [0, 1].
                   Can include None for normal images.
        heatmaps: List of predicted anomaly maps, each (H, W) float.
        compute_pro_metric: If True, also compute PRO (slower).
        
    Returns:
        Dictionary with:
        - pixel_auroc: Pixel-level AUROC
        - pro: Per-Region Overlap (if compute_pro_metric=True)
        - n_images: Number of images evaluated
        - n_anomalous_images: Number of images with masks
        
    """
    # Count anomalous images
    n_anomalous = sum(1 for m in masks_true if m is not None and np.any(m > 0.5))
    
    metrics = {
        'n_images': len(masks_true),
        'n_anomalous_images': n_anomalous
    }
    
    # Compute Pixel AUROC
    try:
        metrics['pixel_auroc'] = compute_pixel_auroc(masks_true, heatmaps)
    except Exception as e:
        print(f"Warning: Could not compute pixel AUROC: {e}")
        metrics['pixel_auroc'] = None
    
    # Compute PRO
    if compute_pro_metric:
        try:
            metrics['pro'] = compute_pro(masks_true, heatmaps)
        except Exception as e:
            print(f"Warning: Could not compute PRO: {e}")
            metrics['pro'] = None
    
    return metrics


def compute_pixel_roc_curve(
    masks_true: List[np.ndarray],
    heatmaps: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pixel-level ROC curve for visualization.
    
    Args:
        masks_true: List of ground truth masks, each (H, W) binary
        heatmaps: List of predicted anomaly maps, each (H, W) float
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    # Collect all pixels
    all_labels = []
    all_scores = []
    
    for mask, heatmap in zip(masks_true, heatmaps):
        if mask is None:
            mask = np.zeros_like(heatmap)
        
        all_labels.append((mask > 0.5).astype(int).flatten())
        all_scores.append(heatmap.flatten())
    
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    return fpr, tpr, thresholds


def aggregate_pixel_metrics(
    per_class_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate pixel-level metrics across classes (macro-average).
    
    Args:
        per_class_metrics: Dictionary mapping class names to metric dictionaries
        
    Returns:
        Aggregated metrics dictionary
    """
    if not per_class_metrics:
        return {}
    
    aggregated = {}
    
    for key in ['pixel_auroc', 'pro']:
        values = [
            per_class_metrics[c][key] 
            for c in per_class_metrics 
            if key in per_class_metrics[c] and per_class_metrics[c][key] is not None
        ]
        if values:
            aggregated[key] = float(np.mean(values))
    
    return aggregated
