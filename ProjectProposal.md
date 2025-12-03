Necerini Ivan s345147

Rialti Jacopo s346357

Veroli Fabio s336301

---

## Dataset: MVTec AD

**Source**

- Official: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad?utm_source=chatgpt.com)
- Kaggle mirror: [https://www.kaggle.com/datasets/ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad?utm_source=chatgpt.com)

**Selected classes**

- Objects: **Hazelnut**, **Pill**
- Textures: **Carpet**, **Zipper**

**Normal vs anomaly**

- **Normal**: images in `train/good` and `test/good` for the selected classes.
- **Anomalous**: images in `test/<defect_type>/` with pixel-wise masks in `ground_truth/<defect_type>/`.

### Split – Phase 1 (clean domain)

We train **one single global one-class model** on all four classes together:

- all **normal images** from hazelnut, pill, carpet and zipper are **pooled** to define the “normal” distribution;
- all **defective images** from these classes are treated as “anomalous”.

The model does **not** predict the object category; it learns a **joint notion of normality** over this four-class domain and performs **binary anomaly detection (normal vs anomalous)** within it.

For each class we define:

- **Train-clean**
    - Content: majority (e.g. ~80%) of `train/good` images.
    - Only normal images, used to learn nominal appearance and to build the memory / statistical models.
    - No anomalous images are used here.
- **Val-clean**
    - Content:
        - remaining normal images from `train/good` (e.g. ~20%),
        - a small subset of anomalous images sampled from `test/<defect_type>/`.
    - Purpose:
        - calibrate the anomaly-score threshold (normal vs anomalous),
        - tune hyperparameters (patch size, coreset fraction, feature layers, etc.).
    - **Val-clean is disjoint from Test-clean.**
- **Test-clean**
    - Content: all remaining normal images from `test/good` and all remaining anomalous images from `test/<defect_type>/`, with their ground-truth masks.
    - Used **only** for final evaluation (no calibration, no tuning).

Splits are built per class and then merged across hazelnut, pill, carpet and zipper.

### Split – Phase 2 (shifted domain)

To study robustness to **domain shift**, we create a shifted version (**MVTec-Shift**) by applying realistic transformations to Train-clean, Val-clean and Test-clean:

- lighting changes (brightness, contrast, saturation),
- viewpoint changes (rotations, translations, slight perspective),
- blur and sensor noise,
- moderate background changes.

From these transformed images we define:

- **Train-shift**: transformed normal images from Train-clean (used to rebuild / adapt the models).
- **Val-shift**: transformed normal + anomalous images from Val-clean (for threshold calibration under shift).
- **Test-shift**: transformed normal + anomalous images from Test-clean (with transformed masks), used only for evaluation.

We will compare:

- performance on **Test-clean**,
- performance on **Test-shift** with clean-domain models (no adaptation),
- performance on **Test-shift** after adaptation using Train-shift + Val-shift.

## Architectures

We compare two one-class anomaly detection methods, both trained **only on normal images**.

### 1. Main method – PatchCore-style (ResNet-50)

- **Backbone**: ResNet-50 pre-trained on ImageNet, used as a frozen feature extractor.
- **Feature extraction**: intermediate feature maps (e.g. from multiple layers) are upsampled and concatenated to obtain multi-scale feature maps.
- **Patch embeddings**: features are extracted as patch-level vectors over the spatial grid.
- **Memory bank**:
    - built from patch embeddings of normal images in **Train-clean**, pooled across all four classes;
    - optionally reduced via **coreset sampling** (e.g. 10–20% of patches) to keep the memory compact.
- **Anomaly scoring (nearest neighbour)**:
    - for each test patch, compute the distance to its **nearest neighbour (k-NN, k = 1)** in the memory bank;
    - aggregate patch scores into an **image-level anomaly score** (e.g. max or top-k percentile);
    - upsample patch scores to obtain an **anomaly heatmap** for localization.
- **Decision rule**:
    - final label (normal / anomalous) obtained by **thresholding** the image-level anomaly score;
    - thresholds are calibrated on Val-clean (and Val-shift) only.
    - no additional supervised classifier.

### 2. Baseline – PaDiM (anomalib)

As a baseline we use **PaDiM** in the official **anomalib** implementation:

- **Backbone**: ResNet-based feature extractor, configured to match PatchCore as closely as possible (image size, feature layers).
- **Modeling**: for each spatial position, PaDiM fits a **multivariate Gaussian** over patch features from normal training data.
- **Anomaly score**:
    - pixel-wise scores from the **Mahalanobis distance** to the normal distribution;
    - image-level score by aggregating pixel scores.
- **Decision rule**: same threshold-calibration protocol as PatchCore on Val-clean / Val-shift.

Both methods use the **same splits** (clean and shifted) for a fair comparison.

## Training Setup

### PatchCore-style

- **Backbone**: ResNet-50, frozen.
- **Training data**: normal images from Train-clean (and Train-shift for adaptation).
- **Procedure**:
    - run a forward pass on training images to extract patch features.
    - build the memory bank.
- **Main hyperparameters**:
    - input image size,
    - feature layers used,
    - patch size and stride,
    - coreset fraction,
    - distance metric for nearest neighbour.
- **Threshold calibration**:
    - compute anomaly scores on Val-clean / Val-shift (normal + anomalous);
    - choose the operating threshold by maximising a chosen criterion on validation (e.g. AUROC, F1 or a desired precision–recall trade-off).

### PaDiM baseline

- **Training data**: normal images from Train-clean (and Train-shift for adaptation).
- **Procedure**:
    - extract patch features and estimate mean and covariance per spatial location;
    - no gradient-based optimisation.
- **Hyperparameters**:
    - feature layers and dimensionality reduction,
    - Mahalanobis distance configuration.
- **Threshold calibration**:
    - same protocol as PatchCore on Val-clean / Val-shift.

## Evaluation Metrics

We will evaluate both **detection** (image-level) and **localization** (pixel-level), in the **clean** and **shifted** domains, for both PatchCore-style and PaDiM.

**Image-level metrics**

- **AUROC**: ability to separate normal vs anomalous images across thresholds.
- **AUPRC**: precision–recall behaviour for the anomalous class in an imbalanced setting.
- **Thresholded performance**: accuracy / precision / recall at the calibrated threshold from validation.

**Pixel-level metrics**

- **Pixel-wise AUROC**: discrimination between defective and non-defective pixels using ground-truth masks.
- **PRO (Per-Region Overlap)**: overlap between predicted anomalous regions and ground-truth defect regions, to assess localization quality.

These metrics will be reported for:

- PatchCore-style vs PaDiM,
- Test-clean vs Test-shift (with and without adaptation),

to analyse robustness, failure cases and the impact of domain shift.