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
│   ├── raw/                               # Original MVTec AD
│   ├── processed/                         # Saved splits (pickle/json)
│   └── shifted/                           # Generated MVTec-Shift
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

## 🔄 Pipeline Step-by-Step

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
  - Input: MVTec folders
  - Output: dictionaries with image/mask paths
  - Train-clean: 80% of `train/good`
  - Val-clean: 20% `train/good` + 30% anomalous from `test/`
  - Test-clean: remaining `test/good` + remaining anomalous
- [x] Save splits in `data/processed/clean_splits.json`
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
- [ ] `src/data/transforms.py`:
  - `ShiftTransform`:
    - **Geometric** (apply to image + mask):
      - Random rotation [-10°, +10°]
      - Random resized crop [0.9, 1.0]
      - Small translation
    - **Photometric** (image only):
      - ColorJitter (brightness, contrast, saturation [0.7, 1.3])
      - Gaussian blur (kernel 3-5)
      - Gaussian noise σ [0.01, 0.05]
    - Optional: simulated vignetting
  - Use `torchvision.transforms` and `albumentations` for consistency

#### Step 2.2: Generate MVTec-Shift Dataset (Notebook 03)
- [ ] For each split (Train-clean, Val-clean, Test-clean):
  - Apply `ShiftTransform` to all images + masks
  - Save in `data/shifted/` maintaining structure
- [ ] Create `data/processed/shifted_splits.json`
- [ ] Visualize clean vs shifted examples

---

### **PHASE 3: PatchCore - Clean Domain**

#### Step 3.1: Implement Backbone (Notebook 04)
- [ ] `src/models/backbones.py`:
  - `ResNet50FeatureExtractor`:
    - Pre-trained ImageNet
    - Hook on intermediate layers (e.g., layer2, layer3)
    - Forward → multi-scale features
    - Local average pooling to reduce dimensionality
    - Concatenation and upsampling to common resolution

#### Step 3.2: Implement Memory Bank (Notebook 04)
- [ ] `src/models/memory_bank.py`:
  - `GreedyCoresetSubsampling`:
    - Input: patch features from Train-clean
    - Greedy selection: iteratively select patches that maximize min-distance to already selected ones
    - Target: 1-10% of total patches
    - Output: reduced memory
  - `MemoryBank`:
    - Store features + nearest neighbor methods
    - Reweighting based on local density

#### Step 3.3: Implement PatchCore (Notebook 04)
- [ ] `src/models/patchcore.py`:
  - `PatchCore`:
    - `fit(train_loader)`: extract features, build memory bank
    - `predict(image)`: 
      - Extract patch features
      - Nearest neighbor for each patch
      - Generate anomaly heatmap
      - Aggregate image-level score (max or top-k percentile)
    - `save/load`: save memory bank

#### Step 3.4: Training PatchCore - Clean (Notebook 04)
- [ ] For each class (hazelnut, carpet, zipper):
  - Load Train-clean
  - Fit PatchCore
  - Save memory bank in `outputs/models/patchcore_{class}_clean.pkl`
- [ ] Training time and memory bank size

---

### **PHASE 4: PaDiM - Clean Domain**

#### Step 4.1: Setup PaDiM with Anomalib (Notebook 05)
- [ ] `src/models/padim_wrapper.py`:
  - Wrapper for `anomalib.models.Padim`
  - ResNet backbone configuration
  - Unified interface with PatchCore

#### Step 4.2: Training PaDiM - Clean (Notebook 05)
- [ ] For each class:
  - Adapt dataloader for anomalib
  - Fit PaDiM on Train-clean
  - Save model in `outputs/models/padim_{class}_clean.pkl`

---

### **PHASE 5: Threshold Calibration & Evaluation - Clean Domain**

#### Step 5.1: Implement Threshold Selection (Notebook 06)
- [ ] `src/metrics/threshold_selection.py`:
  - `calibrate_threshold(scores_normal, scores_anomalous)`:
    - Gridsearch thresholds
    - Calculate F1 for each threshold
    - Return best threshold that maximizes F1
  - Apply on Val-clean for each class and method

#### Step 5.2: Implement Metrics (Notebook 06)
- [ ] `src/metrics/image_metrics.py`:
  - `compute_auroc(y_true, scores)`
  - `compute_auprc(y_true, scores)`
  - `compute_f1_at_threshold(y_true, scores, threshold)`
  - `compute_accuracy_precision_recall(y_true, y_pred)`

- [ ] `src/metrics/pixel_metrics.py`:
  - `compute_pixel_auroc(masks_true, heatmaps)`
  - `compute_pro(masks_true, heatmaps)` (Per-Region Overlap)

#### Step 5.3: Evaluation on Test-Clean (Notebook 06)
- [ ] For each class and method:
  - Predict on Test-clean
  - Apply threshold from Val-clean
  - Calculate all image-level and pixel-level metrics
- [ ] Save results in `outputs/results/clean_results.json`
- [ ] Tables per class and macro-average

#### Step 5.4: Visualizations (Notebook 06)
- [ ] Top-K best/worst predictions
- [ ] Heatmaps overlay on anomalous images
- [ ] Confusion matrix
- [ ] ROC and Precision-Recall curves

---

### **PHASE 6: Domain Shift - No Adaptation**

#### Step 6.1: Evaluation on Test-Shift without Adaptation (Notebook 09)
- [ ] Use models trained on Train-clean
- [ ] Use thresholds calibrated on Val-clean (same as PHASE 5)
- [ ] Predict on Test-shift
- [ ] Calculate metrics
- [ ] Save in `outputs/results/shift_no_adaptation_results.json`
- [ ] **Performance degradation analysis** compared to Test-clean

---

### **PHASE 7: Domain Shift - With Adaptation**

#### Step 7.1: Re-training on Train-Shift (Notebook 07, 08)
- [ ] **PatchCore**:
  - Rebuild memory bank using Train-shift
  - Save in `outputs/models/patchcore_{class}_shift.pkl`
- [ ] **PaDiM**:
  - Re-fit on Train-shift
  - Save in `outputs/models/padim_{class}_shift.pkl`

#### Step 7.2: Re-calibrate Thresholds on Val-Shift (Notebook 07, 08)
- [ ] Predict on Val-shift with new models
- [ ] Calibrate new thresholds maximizing F1
- [ ] Save in `outputs/thresholds/shift_thresholds.json`

#### Step 7.3: Evaluation on Test-Shift with Adaptation (Notebook 09)
- [ ] Predict with adapted models
- [ ] Apply thresholds from Val-shift
- [ ] Calculate metrics
- [ ] Save in `outputs/results/shift_with_adaptation_results.json`
- [ ] **Improvement analysis** compared to no-adaptation

---

### **PHASE 8: Global Model**

#### Step 8.1: Training Global Model
- [ ] Pool all normal images from the 3 classes
- [ ] Train PatchCore/PaDiM on the pool
- [ ] Evaluate on each class separately
- [ ] **Limitations analysis** of heterogeneous normality

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

