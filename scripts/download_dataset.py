"""
Download MVTec AD dataset.

This script downloads the MVTec Anomaly Detection dataset from Kaggle
or the official MVTec website and extracts it to the data/raw directory.

Usage:
    python scripts/download_dataset.py [--source kaggle|official]
"""

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Literal

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.paths import paths


def download_from_kaggle() -> bool:
    """
    Download MVTec AD from Kaggle using kaggle API.
    
    Requires:
        - kaggle API installed (pip install kaggle)
        - Kaggle credentials configured (~/.kaggle/kaggle.json)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import kaggle
        
        print("Downloading MVTec AD from Kaggle...")
        print("   Dataset: ipythonx/mvtec-ad")
        
        # Create temp directory
        temp_dir = paths.DATA / 'temp_download'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Download
        kaggle.api.dataset_download_files(
            'ipythonx/mvtec-ad',
            path=str(temp_dir),
            unzip=True
        )
        
        # Move to correct location
        # The Kaggle dataset might have different structure
        downloaded_path = temp_dir / 'mvtec_anomaly_detection'
        if not downloaded_path.exists():
            # Try alternative structure
            downloaded_path = temp_dir
        
        target_path = paths.DATA_RAW / 'mvtec_ad'
        
        if downloaded_path.exists():
            print(f"Extracting to {target_path}...")
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.move(str(downloaded_path), str(target_path))
            
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            print("Download complete!")
            return True
        else:
            print("ERROR: Downloaded files not found in expected location")
            return False
            
    except ImportError:
        print("ERROR: Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"ERROR: Error downloading from Kaggle: {e}")
        print("\nTroubleshooting:")
        print("1. Install kaggle API: pip install kaggle")
        print("2. Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("3. Place kaggle.json in ~/.kaggle/")
        return False


def verify_dataset() -> bool:
    """
    Verify that the dataset was downloaded correctly.
    
    Checks for:
        - Expected directory structure
        - Presence of selected classes (hazelnut, carpet, zipper)
        - Presence of train/test splits
    
    Returns:
        True if verification passed, False otherwise
    """
    print("\nVerifying dataset structure...")
    
    mvtec_path = paths.DATA_RAW / 'mvtec_ad'
    
    if not mvtec_path.exists():
        print(f"ERROR: Dataset directory not found: {mvtec_path}")
        return False
    
    # Check for required classes
    required_classes = ['hazelnut', 'carpet', 'zipper']
    
    for class_name in required_classes:
        class_path = mvtec_path / class_name
        
        if not class_path.exists():
            print(f"ERROR: Class directory not found: {class_name}")
            return False
        
        # Check for train/test structure
        train_good = class_path / 'train' / 'good'
        test_good = class_path / 'test' / 'good'
        ground_truth = class_path / 'ground_truth'
        
        if not train_good.exists():
            print(f"ERROR: Train directory not found for {class_name}")
            return False
        
        if not test_good.exists():
            print(f"ERROR: Test directory not found for {class_name}")
            return False
        
        # Count images
        train_count = len(list(train_good.glob('*.png')))
        test_count = len(list(test_good.glob('*.png')))
        
        print(f"OK: {class_name}: {train_count} train, {test_count} test (good)")
    
    print("\nDataset verification passed!")
    print(f"Dataset location: {mvtec_path}")
    return True


def main():
    """Main function to download MVTec AD dataset."""
    parser = argparse.ArgumentParser(
        description='Download MVTec Anomaly Detection dataset'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['kaggle'],
        default='kaggle',
        help='Download source (default: kaggle)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset without downloading'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MVTec Anomaly Detection Dataset - Downloader")
    print("=" * 70)
    
    # Ensure data directory exists
    paths.DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    if args.verify_only:
        verify_dataset()
        return
    
    # Check if already downloaded
    if (paths.DATA_RAW / 'mvtec_ad').exists():
        print("WARNING: Dataset already exists at data/raw/mvtec_ad")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download. Verifying existing dataset...")
            verify_dataset()
            return
    
    # Download based on source
    success = False
    
    if args.source == 'kaggle':
        success = download_from_kaggle()
    
    if success:
        # Verify
        verify_dataset()
        print("\nSetup complete! You can now proceed with the notebooks.")
    else:
        print("\nERROR: Download failed. Please try:")
        print("   1. Different source: --source official")
        print("   2. Manual download from: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print("   3. Extract to: data/raw/mvtec_ad/")
        sys.exit(1)


if __name__ == '__main__':
    main()
