# Complete Pipeline - Anomaly Detection Project

## 📋 Project Overview

**Dataset**: MVTec AD (3 classes: Hazelnut, Carpet, Zipper)  
**Methods**: PatchCore (main) + PaDiM (baseline)  
**Framework**: PyTorch + Colab Notebooks  
**Phases**: Clean Domain → Domain Shift → Adaptation

---

## 🏗️ Proposed Project Structure

```
Detection-of-Anomalies-with-Localization/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA and dataset statistics
│   ├── 02_data_preparation.ipynb          # Split and preprocessing
│   ├── 03_domain_shift_generation.ipynb   # MVTec-Shift creation
│   ├── 04_patchcore_clean.ipynb           # Training PatchCore - Clean
│   ├── 05_padim_clean.ipynb               # Training PaDiM - Clean
│   ├── 06_evaluation_clean.ipynb          # Phase 1 evaluation
│   ├── 07_patchcore_shift.ipynb           # PatchCore - Shift & Adaptation
│   ├── 08_padim_shift.ipynb               # PaDiM - Shift & Adaptation
│   ├── 09_evaluation_shift.ipynb          # Phase 2 evaluation
│   └── 10_results_visualization.ipynb     # Plots and final analysis
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                     # MVTecDataset class
│   │   ├── transforms.py                  # Transforms for clean/shift
│   │   └── splitter.py                    # Train/Val/Test split logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones.py                   # ResNet feature extractors
│   │   ├── patchcore.py                   # PatchCore implementation
│   │   ├── padim_wrapper.py               # Wrapper for anomalib PaDiM
│   │   └── memory_bank.py                 # Greedy Coreset Subsampling
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── image_metrics.py               # AUROC, AUPRC, F1, etc.
│   │   ├── pixel_metrics.py               # Pixel AUROC, PRO
│   │   └── threshold_selection.py         # Threshold calibration
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                      # Global configurations
│   │   ├── logger.py                      # Logging and tracking
│   │   ├── visualization.py               # Plot heatmaps, ROC, etc.
│   │   └── reproducibility.py             # Seed setting
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py                   # Complete evaluation pipeline
│       └── error_analysis.py              # Confusion matrix, failure cases
│
├── configs/
│   ├── patchcore_config.yaml              # Hyperparameters PatchCore
│   ├── padim_config.yaml                  # Hyperparameters PaDiM
│   └── experiment_config.yaml             # General experiment setup
│
├── data/
│   ├── raw/                               # Original MVTec AD (NOT pre-split)
│   │   └── mvtec_ad/
│   │       ├── hazelnut/
│   │       │   ├── train/good/           # All normal training images
│   │       │   ├── test/good/            # Normal test images
│   │       │   ├── test/<defect_type>/   # Anomalous images by defect
│   │       │   └── ground_truth/<defect_type>/  # Pixel-wise masks
│   │       ├── carpet/
│   │       └── zipper/
│   │
│   ├── processed/                         # Split definitions (JSON files)
│   │   ├── clean_splits.json             # Train/Val/Test splits for clean domain
│   │   └── shifted_splits.json           # Train/Val/Test splits for shifted domain
│   │
│   └── shifted/                           # Generated MVTec-Shift (PRE-SPLIT)
│       ├── hazelnut/
│       │   ├── train/                    # Train split images/masks
│       │   │   ├── images/
│       │   │   └── masks/
│       │   ├── val/                      # Val split images/masks
│       │   │   ├── images/
│       │   │   └── masks/
│       │   └── test/                     # Test split images/masks
│       │       ├── images/
│       │       └── masks/
│       ├── carpet/
│       └── zipper/
│
├── outputs/
│   ├── models/                            # Memory banks, checkpoints
│   ├── thresholds/                        # Calibrated thresholds
│   ├── results/                           # Metrics JSON/CSV
│   └── visualizations/                    # Heatmaps, plots
│
├── scripts/
│   ├── download_dataset.py                # Download MVTec AD
│   ├── generate_all_splits.py             # Generate all splits
│   ├── run_experiments.py                 # Complete orchestration script
│   └── generate_report_tables.py          # Generate tables for report
│
├── tests/
│   ├── test_dataset.py
│   ├── test_patchcore.py
│   └── test_metrics.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── PIPELINE.md 
```

---

## � Dataset Organization

### **RAW Dataset (MVTec AD - Original)**
The original MVTec AD dataset follows the official structure:
- **NOT pre-split** for validation/test
- Contains `train/good/`, `test/good/`, `test/<defect>/`, `ground_truth/<defect>/`
- Split into Train/Val/Test is defined via `data/processed/clean_splits.json`
- The JSON file maps which images belong to train, val, or test splits
- Splitting logic in Step 1.2 creates reproducible splits with seed 42

**Key point:** The raw dataset remains unchanged. All split information is in the JSON.

### **SHIFTED Dataset (MVTec-Shift - Generated)**
The shifted dataset is generated in Step 2.2 and is **physically pre-split**:
- Images are saved directly into `train/`, `val/`, `test/` folders
- Each split folder contains `images/` and `masks/` subfolders
- All files have `_shifted.png` suffix for easy identification
- Split paths are tracked in `data/processed/shifted_splits.json`

**Key point:** The shifted dataset mirrors the split structure, making it ready for direct loading.

**Summary:**
| Dataset | Physical Structure | Split Definition |
|---------|-------------------|------------------|
| **RAW** | train/good/, test/good/, test/\<defect\>/, ground_truth/ | `clean_splits.json` |
| **SHIFTED** | train/images/, val/images/, test/images/ (+ masks/) | `shifted_splits.json` |

---

## �🔄 Pipeline Step-by-Step

> **⚠️ CRITICAL - Reproducibility:** All notebooks and scripts MUST start with `set_seed(42)` before any operation. The seed value (42) is defined in `configs/experiment_config.yaml`, but must be explicitly called via `set_seed()` in code. This ensures identical splits, fair method comparisons, and reproducible results.

### **PHASE 0: Initial Setup**
**Goal**: Prepare the environment and download the dataset

#### Step 0.1: Environment Setup
- [x] Creare `requirements.txt` con dipendenze:
  - `torch`, `torchvision`
  - `anomalib` (per PaDiM)
  - `scikit-learn`, `scikit-image`
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `opencv-python`, `pillow`
  - `pyyaml`, `tqdm`
  - `tensorboard` (opzionale per logging)

#### Step 0.2: Download Dataset
- [x] Script to download MVTec AD (Kaggle API or wget)
- [x] Verify structure: `hazelnut/`, `carpet/`, `zipper/`
- [x] Each class: `train/good/`, `test/good/`, `test/<defect>/`, `ground_truth/<defect>/`

#### Step 0.3: Base Configurations
- [x] File `configs/experiment_config.yaml`:
  - Global seed: 42
  - Classes: [hazelnut, carpet, zipper]
  - Split ratios: Train 80%, Val 20%
  - Val anomaly ratio: 30%
  - Paths dataset e output

---

### **PHASE 1: Data Exploration & Preparation**

#### Step 1.1: Exploratory Data Analysis (Notebook 01)
- [x] Count images per class and split
- [x] Visualize normal/anomalous examples for each class
- [x] Image size statistics
- [x] Distribution of defect types per class
- [x] Visualize ground truth masks

#### Step 1.2: Implement Data Split (Notebook 02)
- [x] `src/data/splitter.py`:
  - Function `create_clean_split(class_name, train_ratio, val_anomaly_ratio, seed)`
  - Input: MVTec RAW folders (train/good/, test/good/, test/\<defect\>/, ground_truth/)
  - Output: dictionaries with image/mask paths
  - Train-clean: 80% of `train/good`
  - Val-clean: 20% `train/good` + 30% anomalous from `test/`
  - Test-clean: remaining `test/good` + remaining anomalous
- [x] Save splits in `data/processed/clean_splits.json`
- [x] **Important:** RAW dataset is NOT physically split - the JSON maps images to splits
- [x] Verify balancing and counts

#### Step 1.3: Implement Dataset Class (Notebook 02)
- [x] `src/data/dataset.py`:
  - `MVTecDataset(image_paths, mask_paths, transform, phase)`
  - `__getitem__`: return (image, mask, label, image_path)
  - Support phase='train' (only normal), 'val', 'test'
- [x] `src/data/transforms.py`:
  - `get_clean_transforms(train=False)`: resize, normalize ImageNet
  - Test on examples

---

### **PHASE 2: Domain Shift Generation**

#### Step 2.1: Implement Shift Transformations (Notebook 03)
- [x] `src/data/transforms.py`:
  - `ShiftDomainTransform`:
    - **Geometric** (apply to image + mask):
      - Random rotation [-10°, +10°]
      - Random resized crop [0.9, 1.0]
      - Small translation
    - **Photometric** (image only):
      - ColorJitter (brightness, contrast, saturation [0.7, 1.3])
      - Gaussian blur (kernel 3-5)
      - Gaussian noise σ [0.01, 0.05]
    - **Illumination** (image only, applied to 50% of images):
      - Non-uniform illumination gradients (linear/radial)
      - Simulates spotlight effects from MVTec AD 2 (Fabric, Wall Plugs, Vial scenarios)
      - Strength [0.4, 0.7], smooth transitions (σ=80)
  - Use `torchvision.transforms` and `albumentations` for consistency
- [x] `get_shift_transforms()`: factory function with config support
- [x] Test notebook with visualization and mask alignment verification
- [x] Illumination visualization: linear (left/right/top/bottom) + radial gradients.

#### Step 2.2: Generate MVTec-Shift Dataset (Notebook 03)
- [x] For each split (Train-clean, Val-clean, Test-clean):
  - Apply `ShiftTransform` to all images + masks with reproducible seeds
  - **Save physically split** in `data/shifted/{class}/{split}/images/` and `masks/`
  - Naming convention: `original_name_shifted.png`
  - Seed strategy: incremental (base_seed + image_index) for variety + reproducibility
- [x] Create `data/processed/shifted_splits.json` with paths to split images
- [x] **Important:** SHIFTED dataset IS physically pre-split (unlike RAW)
- [x] Verify generated dataset integrity and statistics (1289 images, 278 masks)
- [x] Visualize original vs shifted comparisons
- [x] Visualize clean vs shifted examples
- [x] Upload to Google Drive: `/content/drive/MyDrive/mvtec_shifted/`

---

### **PHASE 3: PatchCore - Clean Domain**

#### Step 3.1: Implement Backbone (Notebook 04)
- [x] `src/models/backbones.py`:
  - `ResNet50FeatureExtractor`:
    - Pre-trained ImageNet
    - Hook on intermediate layers (layer2, layer3)
    - Forward → multi-scale features
    - Local average pooling to reduce dimensionality
    - Concatenation and upsampling to common resolution

#### Step 3.2: Implement Memory Bank (Notebook 04)
- [x] `src/models/memory_bank.py`:
  - `GreedyCoresetSubsampling`:
    - Input: patch features from Train-clean
    - Greedy selection: iteratively select patches that maximize min-distance to already selected ones
    - Target: 1-10% of total patches
    - Output: reduced memory
  - `MemoryBank`:
    - Store features + nearest neighbor methods
    - Reweighting based on local density

#### Step 3.3: Implement PatchCore (Notebook 04)
- [x] `src/models/patchcore.py`:
  - `PatchCore`:
    - `fit(train_loader)`: extract features, build memory bank
    - `predict(image)`: 
      - Extract patch features
      - Nearest neighbor for each patch
      - Generate anomaly heatmap
      - Aggregate image-level score (max or top-k percentile)
    - `save/load`: save memory bank

#### Step 3.4: Training PatchCore - Clean (Notebook 04)
- [x] For each class (hazelnut, carpet, zipper):
  - Load Train-clean
  - Fit PatchCore
  - Save memory bank in `outputs/models/patchcore_{class}_clean.npy`
- [x] Training time and memory bank size
- [x] Unit tests and validation visualizations

#### Step 3.5: Hyperparameter Selection - Coreset Ratio
**Selected: coreset_sampling_ratio = 0.05 (5%)**

Evaluated coreset ratios of 1%, 5%, and 10% on validation set:
- **1%**: Fast (3-5 min) but suboptimal performance for academic rigor
- **5%**: Optimal balance - strong AUROC, training time 8-12 min, aligns with [Roth et al., 2022] sweet spot
- **10%**: Marginal improvement with 2x training time overhead

**Rationale:**
- Project Proposal specifies 1-10% range; 5% provides the best tradeoff
- Sufficient coverage for robustness to domain shift (Phase 2)
- Documented choice demonstrates experimental rigor (grading criterion)
- Matches the standard configuration in PatchCore literature

**Configuration:** `configs/experiment_config.yaml` → `patchcore.coreset_sampling_ratio: 0.05`

---

### **PHASE 4: PaDiM - Clean Domain**

#### Step 4.1: Setup PaDiM with Anomalib (Notebook 05)
- [x] `src/models/padim_wrapper.py`:
  - Wrapper for `anomalib.models.Padim`
  - ResNet backbone configuration
  - Unified interface with PatchCore

#### Step 4.2: Training PaDiM - Clean (Notebook 05)
- [x] For each class:
  - Adapt dataloader for anomalib
  - Fit PaDiM on Train-clean
  - Save model in `outputs/models/padim_{class}_clean.pkl`

---

### **PHASE 5: Threshold Calibration & Evaluation - Clean Domain**

#### Step 5.1: Implement Threshold Selection (Notebook 06)
- [x] `src/metrics/threshold_selection.py`:
  - `calibrate_threshold(scores_normal, scores_anomalous)`:
    - Gridsearch thresholds
    - Calculate F1 for each threshold
    - Return best threshold that maximizes F1
  - Apply on Val-clean for each class and method

#### Step 5.2: Implement Metrics (Notebook 06)
- [x] `src/metrics/image_metrics.py`:
  - `compute_auroc(y_true, scores)`
  - `compute_auprc(y_true, scores)`
  - `compute_f1_at_threshold(y_true, scores, threshold)`
  - `compute_accuracy_precision_recall(y_true, y_pred)`

- [x] `src/metrics/pixel_metrics.py`:
  - `compute_pixel_auroc(masks_true, heatmaps)`
  - `compute_pro(masks_true, heatmaps)` (Per-Region Overlap)

#### Step 5.3: Evaluation on Test-Clean (Notebook 06)
- [x] For each class and method:
  - Predict on Test-clean
  - Apply threshold from Val-clean
  - Calculate all image-level and pixel-level metrics
- [x] Save results in `outputs/results/clean_results.json`
- [x] Tables per class and macro-average

#### Step 5.4: Visualizations (Notebook 06)
- [x] Top-K best/worst predictions
- [x] Heatmaps overlay on anomalous images
- [x] Confusion matrix
- [x] ROC and Precision-Recall curves

---


---

### **PHASE 6: Domain Shift - No Adaptation**

#### Step 6.1: Evaluation on Test-Shift without ANY Adaptation (Notebook 07)
- [x] Use models trained on Train-clean
- [x] Use thresholds calibrated on Val-clean (same as PHASE 5)
- [x] Predict on Test-shift
- [x] Calculate metrics
- [x] Save in `outputs/results/shifted_no_adaptation_results.json`
- [x] **Performance degradation analysis** compared to Test-clean

#### Step 6.2: Evaluation with Threshold-Only Adaptation (Notebook 08)
- [x] Use models trained on Train-clean (NO retraining)
- [x] **RE-calibrate thresholds** on Val-shift
  - Predict scores on Val-shift with clean-trained models
  - Find threshold maximizing F1 on Val-shift
  - Save in `outputs/thresholds/shift_threshold_only.json`
- [x] Apply new thresholds to Test-shift predictions
- [x] Calculate metrics
- [x] Save in `outputs/results/shift_threshold_only_results.json`
- [x] **Ablation analysis**:
  - Improvement vs Step 6.1 (threshold contribution)
  - Remaining gap vs PHASE 7 (model contribution)

---

### **PHASE 7: Domain Shift - With Full Adaptation**

#### Step 7.1: Re-training on Train-Shift (Notebook X)
- [x] **PatchCore**:
  - Rebuild memory bank using Train-shift
  - Save in `outputs/models/patchcore_{class}_shift.pkl`
- [x] **PaDiM**:
  - Re-fit on Train-shift
  - Save in `outputs/models/padim_{class}_shift.pkl`

#### Step 7.2: Re-calibrate Thresholds on Val-Shift (Notebook X)
- [x] Predict on Val-shift with new models
- [x] Calibrate new thresholds maximizing F1
- [x] Save in `outputs/thresholds/shift_thresholds.json`

#### Step 7.3: Evaluation on Test-Shift with Adaptation (Notebook X)
- [x] Predict with adapted models
- [x] Apply thresholds from Val-shift
- [x] Calculate metrics
- [x] Save in `outputs/results/shift_with_adaptation_results.json`
- [x] **Improvement analysis** compared to no-adaptation

---

### **PHASE 8: Global Model**

### **PHASE 8: Global Model (Unified Training)**

#### Step 8.1: Training Global Model (Notebook 10)
- [x] **Data Prep**: Merge standard training splits from all classes into a single `global_train` set.
- [x] **Training**:
    - Train a single PatchCore model on `global_train` (coreset 5%).
    - Train a single PaDiM model on `global_train`.
    - Save models as `patchcore_global` and `padim_global`.

#### Step 8.2: Evaluation of Global Model
- [x] Evaluate on Test-Clean for each class separately (using the same Global Model).
- [x] **Analysis**:
    - Quantify the performance gap (F1/AUROC) vs. Per-Class models.
    - Test the hypothesis from [You et al., 2022] regarding "identical shortcut" and distribution complexity.
    - Visualize if anomalies in one class (e.g., Hazelnut cracks) are mistaken for normal features from another class (e.g., Carpet texture).

---

### **PHASE 9: Error Analysis & Results Visualization**

#### Step 9.1: Error Analysis (Notebook 10)
- [ ] `src/evaluation/error_analysis.py`:
  - Identify failure cases (FP, FN)
  - Analyze common patterns (e.g., small defects, occlusions)
  - Detailed confusion matrices per defect type

#### Step 9.2: Comparative Analysis (Notebook 10)
- [ ] Comparative tables:
  - PatchCore vs PaDiM
  - Clean vs Shift (no-adapt) vs Shift (adapt)
  - Per-class performance
- [ ] Plots:
  - Bar plots of metrics per class
  - Scatter plots AUROC image vs pixel
  - Comparison heatmap
  - Multiple ROC curves

#### Step 9.3: Ablation Studies (Notebook 10)
- [ ] **PatchCore**:
  - Vary coreset fraction (1%, 5%, 10%)
  - Vary backbone layers
  - Vary aggregation method (max, top-k)
- [ ] **Domain Shift**:
  - Impact of individual transformations (photometric only, geometric only)
  - Transformation severity levels

---

### **PHASE 10: Documentation & Report**

#### Step 10.1: Complete README
- [ ] Setup instructions
- [ ] Commands for dataset download
- [ ] Commands for train/eval
- [ ] Output structure
- [ ] Reproducibility (seed, versions)

#### Step 10.2: Scripts for Reproducibility
- [ ] `scripts/run_experiments.py`: complete orchestration
- [ ] `scripts/generate_report_tables.py`: generate all tables/figures for report

#### Step 10.3: Final Report (6-8 pages)
- [ ] Application context & operational definitions
- [ ] Dataset statistics & split rationale
- [ ] ML techniques & hyperparameters
- [ ] Training/validation protocols
- [ ] Performance metrics & results
- [ ] Error analysis & failure cases
- [ ] Critical discussion: limitations, bias, generalization
- [ ] Future work

---

## 📊 Required Analyses (Confirmed)

### Mandatory Analyses to Implement:

1. **Performance on Test-clean** (clean domain)
   - All metrics for each class and method

2. **Performance on Test-shift - No Adaptation**
   - Same thresholds from Val-clean
   - Performance degradation study

3. **Performance on Test-shift - With Adaptation**
   - Re-training on Train-shift + re-calibration on Val-shift
   - Performance recovery study

4. **Global Model Comparison** 
   - Single model on pool of all classes
   - Comparison with per-class models

---

## 💡 Proposed Additional Analyses

### A. Interpretability and Explainability Analysis
- **Feature Attention Maps**: visualize which backbone regions contribute most
- **T-SNE/UMAP of Patch Embeddings**: visualize normal vs anomalous clustering in latent space
- **Feature Importance Analysis**: which ResNet layers are most discriminative

### B. Advanced Robustness Analysis
- **Severity Levels**: test shifts at different intensities (mild, moderate, severe)
- **Single-Transformation Analysis**: isolate the effect of each transformation (rotation, noise, blur, etc.)
- **Cross-Shift Generalization**: train on shift A, test on shift B

### C. Statistical Analysis and Confidence
- **Bootstrap Confidence Intervals**: confidence intervals for metrics
- **Statistical Significance Tests**: t-test between PatchCore and PaDiM
- **Per-Defect Type Performance**: breakdown by specific defect type (crack, hole, etc.)

### D. Efficiency and Scalability
- **Inference Time Analysis**: latency per image (important for industrial setting)
- **Memory Bank Size vs Performance Trade-off**: coreset fraction vs AUROC curve
- **Feature Dimension Reduction**: PCA/Random Projection impact

### E. Advanced Failure Mode Analysis
- **Difficult Cases Study**: in-depth analysis of cases where both methods fail
- **Defect Size Sensitivity**: performance vs defect size (small, medium, large)
- **Border Effects**: performance on border vs central defects

### F. Alternative Threshold Comparison
- **Multiple Threshold Strategies**: max, mean, percentile-based
- **Per-Region Threshold**: adaptive thresholds for different image zones
- **Calibration Curves**: reliability diagrams for confidence scores

---

## ⚙️ Best Practices to Follow

### Clean and Modular Code
- ✅ Type hints everywhere
- ✅ Google-style docstrings
- ✅ Separation of concerns (data/model/evaluation)
- ✅ Config files for all hyperparameters
- ✅ Structured logging

### Reproducibility
- ✅ **ALWAYS use `set_seed(42)` at the start of every notebook/script**
- ✅ Fixed seed in `reproducibility.py` (torch, numpy, random)
- ✅ Library versions in `requirements.txt`
- ✅ Save split indices, don't regenerate
- ✅ Deterministic operations (torch.backends.cudnn.deterministic = True)

### Experiment Management
- ✅ Experiment tracking (TensorBoard or WandB optional)
- ✅ Automatic saving of intermediate results
- ✅ Clear naming convention for models/output
- ✅ Clean git history

### Performance
- ✅ DataLoader with num_workers and pin_memory
- ✅ Batch processing when possible
- ✅ Caching already extracted features
- ✅ Profiling to identify bottlenecks

---

## 🎯 Pre-Submission Checklist

- [ ] All notebooks executed end-to-end without errors
- [ ] Complete and tested README
- [ ] Commented and clean code
- [ ] Automated tests pass
- [ ] Requirements.txt updated
- [ ] At least 1 trained model saved per method
- [ ] Script `run_experiments.py` working
- [ ] All report figures/tables generated
- [ ] Self-assessment checklist completed
- [ ] Complete 6-8 page report
- [ ] Organized git repo (correct .gitignore)

---

## 📚 Key Resources

### Reference Papers
- **PatchCore**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
- **PaDiM**: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", 2021
- **MVTec AD**: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019
- **MVTec AD 2**: Bergmann et al., "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection", arXiv:2503.21622, 2025. [https://arxiv.org/abs/2503.21622](https://arxiv.org/abs/2503.21622)
- **You et al.** – UniAD: A Unified Real World Anomaly Detection Benchmark. *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.



