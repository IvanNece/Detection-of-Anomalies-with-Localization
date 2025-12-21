"""
Data splitting utilities for MVTec AD dataset.

This module provides functions to create reproducible train/val/test splits
for the clean domain, following the project's split strategy.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def create_clean_split(
    class_name: str,
    dataset_path: Path,
    train_ratio: float = 0.8,
    val_anomaly_ratio: float = 0.3,
    seed: int = 42
) -> Dict[str, Dict[str, List]]:
    """
    Create clean domain split for a single class.
    
    Split strategy:
    - Train-clean: 80% of train/good (only normal images)
    - Val-clean: 20% of train/good + 30% of test anomalies
    - Test-clean: all test/good + 70% of test anomalies
    
    Args:
        class_name: Class name ('hazelnut', 'carpet', or 'zipper')
        dataset_path: Path to MVTec AD root directory
        train_ratio: Fraction of train/good for training (default: 0.8)
        val_anomaly_ratio: Fraction of anomalies for validation (default: 0.3)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits, each containing:
        - 'images': list of image paths (as strings)
        - 'masks': list of mask paths (None for normal images)
        - 'labels': list of labels (0=normal, 1=anomalous)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    class_path = dataset_path / class_name
    
    if not class_path.exists():
        raise FileNotFoundError(f"Class directory not found: {class_path}")
    
    # ========================================================================
    # 1. Collect all normal images from train/good
    # ========================================================================
    train_good_path = class_path / 'train' / 'good'
    if not train_good_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_good_path}")
    
    train_good_images = sorted(train_good_path.glob('*.png'))
    train_good_images = [str(p) for p in train_good_images]
    
    # Shuffle and split train/good into train and val
    random.shuffle(train_good_images)
    split_idx = int(len(train_good_images) * train_ratio)
    
    train_normal_images = train_good_images[:split_idx]
    val_normal_images = train_good_images[split_idx:]
    
    # ========================================================================
    # 2. Collect all test normal images (test/good)
    # ========================================================================
    test_good_path = class_path / 'test' / 'good'
    test_normal_images = []
    if test_good_path.exists():
        test_normal_images = sorted(test_good_path.glob('*.png'))
        test_normal_images = [str(p) for p in test_normal_images]
    
    # ========================================================================
    # 3. Collect all anomalous images from test/<defect_type>/
    # ========================================================================
    test_path = class_path / 'test'
    ground_truth_path = class_path / 'ground_truth'
    
    anomalous_data = []  # List of (image_path, mask_path)
    
    for defect_dir in sorted(test_path.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != 'good':
            defect_type = defect_dir.name
            
            # Get anomalous images
            defect_images = sorted(defect_dir.glob('*.png'))
            
            # Get corresponding masks
            mask_dir = ground_truth_path / defect_type
            
            for img_path in defect_images:
                img_name = img_path.name
                
                # Find corresponding mask (MVTec AD masks have '_mask' suffix)
                # e.g., image: 007.png -> mask: 007_mask.png
                mask_name = img_name.replace('.png', '_mask.png')
                
                if not mask_dir.exists():
                    # Mask directory doesn't exist for this defect type
                    anomalous_data.append((str(img_path), None))
                    continue
                
                mask_path = mask_dir / mask_name
                
                if mask_path.exists():
                    anomalous_data.append((str(img_path), str(mask_path)))
                else:
                    # Mask file not found
                    anomalous_data.append((str(img_path), None))
    
    # Debug: Count masks found
    masks_found = sum(1 for _, mask in anomalous_data if mask is not None)
    print(f"  {class_name}: Found {len(anomalous_data)} anomalies, {masks_found} with masks")
    
    # Shuffle anomalous data
    random.shuffle(anomalous_data)
    
    # Split anomalous data: 30% for val, 70% for test
    val_anomaly_count = int(len(anomalous_data) * val_anomaly_ratio)
    
    val_anomalous_data = anomalous_data[:val_anomaly_count]
    test_anomalous_data = anomalous_data[val_anomaly_count:]
    
    # ========================================================================
    # 4. Build split dictionaries
    # ========================================================================
    
    # TRAIN split (only normal images, no masks)
    train_split = {
        'images': train_normal_images,
        'masks': [None] * len(train_normal_images),
        'labels': [0] * len(train_normal_images)  # 0 = normal
    }
    
    # VAL split (normal + anomalous)
    val_images = val_normal_images + [item[0] for item in val_anomalous_data]
    val_masks = [None] * len(val_normal_images) + [item[1] for item in val_anomalous_data]
    val_labels = [0] * len(val_normal_images) + [1] * len(val_anomalous_data)
    
    val_split = {
        'images': val_images,
        'masks': val_masks,
        'labels': val_labels
    }
    
    # TEST split (normal + anomalous)
    test_images = test_normal_images + [item[0] for item in test_anomalous_data]
    test_masks = [None] * len(test_normal_images) + [item[1] for item in test_anomalous_data]
    test_labels = [0] * len(test_normal_images) + [1] * len(test_anomalous_data)
    
    test_split = {
        'images': test_images,
        'masks': test_masks,
        'labels': test_labels
    }
    
    return {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }


def create_all_clean_splits(
    dataset_path: Path,
    classes: List[str],
    train_ratio: float = 0.8,
    val_anomaly_ratio: float = 0.3,
    seed: int = 42
) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    Create clean domain splits for all classes.
    
    Args:
        dataset_path: Path to MVTec AD root directory
        classes: List of class names to process
        train_ratio: Fraction of train/good for training
        val_anomaly_ratio: Fraction of anomalies for validation
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping class names to their splits
    """
    all_splits = {}
    
    for class_name in classes:
        print(f"Creating split for {class_name}...")
        split = create_clean_split(
            class_name=class_name,
            dataset_path=dataset_path,
            train_ratio=train_ratio,
            val_anomaly_ratio=val_anomaly_ratio,
            seed=seed
        )
        all_splits[class_name] = split
        
        # Print summary
        val_normal = len([l for l in split['val']['labels'] if l == 0])
        val_anomalous = len([l for l in split['val']['labels'] if l == 1])
        test_normal = len([l for l in split['test']['labels'] if l == 0])
        test_anomalous = len([l for l in split['test']['labels'] if l == 1])
        
        print(f"  Train: {len(split['train']['images'])} images (all normal)")
        print(f"  Val:   {len(split['val']['images'])} images ({val_normal} normal, {val_anomalous} anomalous)")
        print(f"  Test:  {len(split['test']['images'])} images ({test_normal} normal, {test_anomalous} anomalous)")
        print()
    
    return all_splits


def save_splits(
    splits: Dict[str, Dict[str, Dict[str, List]]],
    output_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save splits to JSON file.
    
    Args:
        splits: Dictionary of splits (from create_all_clean_splits)
        output_path: Path to output JSON file
        metadata: Optional metadata to include (seed, date, config, etc.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data with metadata
    data = {
        'splits': splits,
        'metadata': metadata or {}
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Splits saved to {output_path}")


def load_splits(splits_path: Path) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    Load splits from JSON file.
    
    Args:
        splits_path: Path to splits JSON file
        
    Returns:
        Dictionary of splits
        
    Example:
        >>> splits = load_splits(Path('data/processed/clean_splits.json'))
        >>> train_images = splits['hazelnut']['train']['images']
    """
    with open(splits_path, 'r') as f:
        data = json.load(f)
    
    # Return only splits (ignore metadata)
    return data.get('splits', data)


def verify_split(split: Dict[str, Dict[str, List]]) -> Dict[str, any]:
    """
    Verify split integrity and return statistics.
    
    Checks:
    - No overlap between train/val/test
    - All files exist
    - Label consistency (mask exists iff label=1)
    
    Args:
        split: Split dictionary for a single class
        
    Returns:
        Dictionary with verification results and statistics
    """
    stats = {
        'train_count': len(split['train']['images']),
        'val_count': len(split['val']['images']),
        'test_count': len(split['test']['images']),
        'total_count': 0,
        'no_overlap': False,
        'all_files_exist': True,
        'label_consistency': True
    }
    
    # Check overlap
    train_set = set(split['train']['images'])
    val_set = set(split['val']['images'])
    test_set = set(split['test']['images'])
    
    stats['no_overlap'] = (
        len(train_set & val_set) == 0 and
        len(train_set & test_set) == 0 and
        len(val_set & test_set) == 0
    )
    
    stats['total_count'] = len(train_set | val_set | test_set)
    
    # Check file existence and label consistency
    for split_name in ['train', 'val', 'test']:
        for img_path, mask_path, label in zip(
            split[split_name]['images'],
            split[split_name]['masks'],
            split[split_name]['labels']
        ):
            # Check image exists
            if not Path(img_path).exists():
                stats['all_files_exist'] = False
            
            # Check label consistency
            if label == 0 and mask_path is not None:
                stats['label_consistency'] = False
            if label == 1 and mask_path is not None and not Path(mask_path).exists():
                stats['all_files_exist'] = False
    
    return stats
