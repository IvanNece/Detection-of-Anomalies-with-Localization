# Detection of Anomalies with Localization

**Computer Vision Project - Machine Learning for Vision and Multimedia**  
**Politecnico di Torino**

Authors: Ivan Necerini (s345147), Jacopo Rialti (s346357), Fabio Veroli (s336301)

---

## 📋 Project Overview

Anomaly detection system for industrial quality control using MVTec AD dataset. Comparison of two state-of-the-art methods:

- **PatchCore** (main) - Memory bank with Greedy Coreset Subsampling
- **PaDiM** (baseline) - Gaussian modeling with Mahalanobis distance

**Features**: Image-level detection, pixel-level localization, domain shift robustness evaluation.

**Dataset**: MVTec AD - 3 classes (Hazelnut, Carpet, Zipper)

---

## 🚀 Quick Setup

### Prerequisites
- Python 3.10+
- ~10GB disk space
- CUDA GPU (recommended for training)

### Installation

```powershell
# 1. Clone repository
git clone https://github.com/IvanNece/Detection-of-Anomalies-with-Localization.git
cd Detection-of-Anomalies-with-Localization

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Setup Kaggle credentials
# Download kaggle.json from https://www.kaggle.com/settings
Copy-Item "path\to\kaggle.json" ".kaggle\kaggle.json"
$env:KAGGLE_CONFIG_DIR = "$PWD\.kaggle"

# 5. Download dataset
python scripts/download_dataset.py --source kaggle

# 6. Verify setup
python scripts/download_dataset.py --verify-only
```

---

## 📝 Documentation

- **Pipeline**: `PIPELINE.md`
- **Proposal**: `ProjectProposal.md`
- **Instructions**: `Instruction.md`

---


