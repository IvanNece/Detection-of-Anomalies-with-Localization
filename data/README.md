# Data Directory Structure

This directory contains the dataset files and split definitions for the anomaly detection project.

---

## 📂 Directory Overview

```
data/
├── raw/                    # Original MVTec AD dataset
├── processed/              # Split definitions (JSON files)
└── shifted/                # Generated shifted dataset
```

---

## 📋 Detailed Structure

### 1. `raw/` - Original MVTec AD Dataset

**Structure:** Official MVTec AD format (NOT pre-split)

```
raw/mvtec_ad/
├── hazelnut/
│   ├── train/good/                    # All normal training images (391 images)
│   ├── test/good/                     # Normal test images (40 images)
│   ├── test/crack/                    # Anomalous: crack defect (18 images)
│   ├── test/cut/                      # Anomalous: cut defect (17 images)
│   ├── test/hole/                     # Anomalous: hole defect (18 images)
│   ├── test/print/                    # Anomalous: print defect (17 images)
│   ├── ground_truth/crack/            # Pixel-wise masks for crack
│   ├── ground_truth/cut/              # Pixel-wise masks for cut
│   ├── ground_truth/hole/             # Pixel-wise masks for hole
│   └── ground_truth/print/            # Pixel-wise masks for print
├── carpet/
└── zipper/
```

**Key Points:**
- ❌ **NOT split** into train/val/test folders
- ✅ Split information is defined in `processed/clean_splits.json`
- ✅ Contains all original images without modifications
- ✅ Ground truth masks are separate from images

**Location:**
- Local: `data/raw/mvtec_ad/`
- Google Drive: `/content/drive/MyDrive/mvtec_ad/`

---

### 2. `processed/` - Split Definitions

**Structure:** JSON files mapping images to train/val/test splits

```
processed/
├── clean_splits.json          # Splits for original (clean) dataset
└── shifted_splits.json        # Splits for shifted dataset
```

**`clean_splits.json` example:**
```json
{
  "splits": {
    "hazelnut": {
      "train": {
        "images": ["/path/to/train/good/001.png", ...],
        "masks": [null, ...],
        "labels": [0, 0, ...]
      },
      "val": { ... },
      "test": { ... }
    }
  }
}
```

**Key Points:**
- ✅ Defines which RAW images belong to train/val/test
- ✅ Created with seed 42 for reproducibility
- ✅ Train: 80% of train/good, Val: 20% train/good + 30% anomalies, Test: rest
- ✅ Paths point to RAW dataset locations

---

### 3. `shifted/` - Generated Shifted Dataset

**Structure:** Physically pre-split into train/val/test folders

```
shifted/
├── hazelnut/
│   ├── train/
│   │   ├── images/                   # 312 shifted normal images
│   │   └── masks/                    # Empty (train = only normal)
│   ├── val/
│   │   ├── images/                   # 100 shifted images (79 normal + 21 anomalous)
│   │   └── masks/                    # 21 masks for anomalous samples
│   └── test/
│       ├── images/                   # 89 shifted images (40 normal + 49 anomalous)
│       └── masks/                    # 49 masks for anomalous samples
├── carpet/
└── zipper/
```

**Key Points:**
- ✅ **Already split** into train/val/test folders
- ✅ Images have `_shifted.png` suffix
- ✅ All transforms applied: geometric + photometric + illumination
- ✅ Masks are synchronized with geometric transforms
- ✅ Generated with incremental seeds for variety (42, 43, 44, ...)

**Location:**
- Local: `data/shifted/`
- Google Drive: `/content/drive/MyDrive/mvtec_shifted/`

---

## 📊 Dataset Statistics

| Class | Train | Val (Normal) | Val (Anomalous) | Test (Normal) | Test (Anomalous) | Total |
|-------|-------|--------------|-----------------|---------------|------------------|-------|
| Hazelnut | 312 | 79 | 21 | 40 | 49 | 501 |
| Carpet | 224 | 56 | 26 | 28 | 63 | 397 |
| Zipper | 192 | 48 | 35 | 32 | 84 | 391 |
| **TOTAL** | **728** | **183** | **82** | **100** | **196** | **1289** |

---

## 🔄 Workflow Summary

1. **Download RAW dataset** → `data/raw/mvtec_ad/`
2. **Create splits** (Notebook 02) → `data/processed/clean_splits.json`
3. **Generate shifted dataset** (Notebook 03) → `data/shifted/` + `shifted_splits.json`
4. **Training/Evaluation** → Use JSON files to load correct splits

---

## 📁 Important Files

| File | Purpose | Size |
|------|---------|------|
| `clean_splits.json` | Maps RAW images to train/val/test | ~200 KB |
| `shifted_splits.json` | Maps SHIFTED images to train/val/test | ~200 KB |
| `raw/mvtec_ad/` | Original dataset | ~4.5 GB |
| `shifted/` | Shifted dataset | ~82 MB |

---

## ⚠️ Key Differences: RAW vs SHIFTED

| Aspect | RAW Dataset | SHIFTED Dataset |
|--------|-------------|-----------------|
| **Physical Split** | ❌ No (uses JSON) | ✅ Yes (train/val/test folders) |
| **Structure** | train/good/, test/<defect>/, ground_truth/ | train/images/, val/images/, test/images/ |
| **Modifications** | ❌ Original images | ✅ Transformed (rotation, color, blur, illumination) |
| **File Naming** | Original names (e.g., 001.png) | With suffix (e.g., 001_shifted.png) |
| **Split Info** | In `clean_splits.json` | In `shifted_splits.json` + folder structure |
| **Use Case** | Clean domain training/evaluation | Domain shift evaluation + adaptation |

---

## 🚀 Usage Examples

### Load Clean Dataset
```python
from src.data.splitter import load_splits

splits = load_splits('data/processed/clean_splits.json')
train_images = splits['hazelnut']['train']['images']  # Paths to RAW images
```

### Load Shifted Dataset
```python
from src.data.splitter import load_splits

splits = load_splits('data/processed/shifted_splits.json')
train_images = splits['hazelnut']['train']['images']  # Paths to SHIFTED images
```

