# Project proposal: Detection of Anomalies with Localization
Necerini Ivan s345147

Rialti Jacopo s346357

Veroli Fabio s336301

---

## Dataset: MVTec AD

**Source**

- Official: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad?utm_source=chatgpt.com)
- Kaggle mirror: [https://www.kaggle.com/datasets/ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad?utm_source=chatgpt.com)

**Selected classes**

- Objects: **Hazelnut**
- Textures: **Carpet**, **Zipper**

**Normal vs anomaly**

- **Normal**: images in `train/good` and `test/good` for the selected classes.
- **Anomalous**: images in `test/<defect_type>/`, with pixel-wise masks in `ground_truth/<defect_type>/`.

### Split – Phase 1 (clean domain)

The **main setup** trains **one separate one-class model per category** (hazelnut, carpet, zipper).

For each class, data are split as follows:

- **Train-clean**
    - ~80% of images in `train/good`, selected with a fixed random seed (e.g., 42).
    - Only normal images, used to learn the nominal appearance and to build the memory / statistical model.
- **Val-clean**
    - Remaining ~20% of `train/good` (normal images), plus
    - a small subset (e.g., 30%) of anomalous images sampled from `test/<defect_type>/`.
    - Used to calibrate the anomaly-score threshold (normal vs anomalous) and tune hyperparameters (patch size, coreset fraction, feature layers, etc.).
- **Test-clean**
    - All remaining normal images from `test/good` and all remaining anomalous images from `test/<defect_type>/`, with ground-truth masks.
    - Used **only** for final evaluation (no calibration, no tuning).

Exact image counts per split will be reported in the final report; splits are generated reproducibly using a fixed random seed and the same protocol for all 3 classes.

**Global model.**

In addition to the per-class setup, we may train a **single global model** pooling all normal images from the 3 classes.
This setup serves as a baseline to demonstrate the "identical shortcut" problem and the difficulty of modeling heterogeneous distributions without class-conditional information, as discussed in the context of unified anomaly detection benchmarks [You et al., 2022].

### Split – Phase 2 (shifted domain)

To study robustness to **domain shift**, we build a synthetic shifted version (**MVTec-Shift**) by applying realistic transformations to Train-clean, Val-clean and Test-clean.

Transformation types and ranges are inspired by the illumination and acquisition variations proposed in MVTec AD 2; in addition, we introduce mild geometric and sensor-level perturbations (small rotations, blur, noise) to explore more general robustness.

For each image–mask pair:

- **Geometric transforms** (applied to image and mask with the same parameters):
    - Random rotation in [−10°, +10°],
    - Random resized crop with scale [0.9, 1.0] and aspect ratio [0.9, 1.1],
    - Small translations.
- **Photometric transforms** (applied only to the image):
    - Color jitter: brightness, contrast, saturation factors sampled in [0.7, 1.3],
    - Gaussian blur (kernel size 3–5),
    - Additive Gaussian noise with σ in [0.01, 0.05] (images normalized to [0, 1]).
- **Optional illumination variations** (applied to 50% of shifted images):
    - **Non-uniform illumination gradients** to simulate industrial spotlight conditions:
      - **Linear gradients**: spotlight from side (left/right/top/bottom random) - simulates Fabric, Wall Plugs, Walnuts scenarios from MVTec AD 2
      - **Radial gradients**: center/edge lighting variations - simulates Vial scenario from MVTec AD 2
      - Darkening factor: [0.4, 0.7] (40-70% brightness reduction on gradient edge)
      - Smooth Gaussian transition (σ=80) for natural appearance

From the transformed data we define:

- **Train-shift**: transformed normal images from Train-clean (used to rebuild / adapt the models).
- **Val-shift**: transformed normal + anomalous images from Val-clean (for threshold calibration under shift).
- **Test-shift**: transformed normal + anomalous images from Test-clean (with transformed masks), used only for evaluation.

We will compare, per class:

- performance on **Test-clean** (clean domain),
- performance on **Test-shift** with thresholds calibrated on Val-clean (no adaptation),
- performance on **Test-shift** after adaptation using Train-shift + Val-shift.

---

## Architectures

We compare two one-class anomaly detection methods, both trained **only on normal images** and implemented in **PyTorch**. PatchCore is implemented from scratch; PaDiM uses the official *anomalib* implementation as baseline.

### 1. Main method – PatchCore-style (ResNet-50)

For each category:

- **Backbone**: ResNet-50 pre-trained on ImageNet, used as a frozen feature extractor.
- **Feature extraction**: intermediate feature maps from multiple layers are locally aggregated (via average pooling), upsampled, and concatenated to obtain locally aware multi-scale feature maps.
- **Patch embeddings**: patch-level feature vectors are extracted on a regular spatial grid.
- **Memory bank**:
    - built from patch embeddings of normal images in **Train-clean** for that class;
    - reduced via **Greedy Coreset Subsampling** to approx. 1% - 10% of the total patches. This algorithm iteratively selects points that maximize the coverage of the feature space, ensuring high performance with low memory usage [Roth et al.].
- **Anomaly scoring (nearest neighbour)**:
    - for each test patch, compute the distance to its nearest neighbour in the memory bank;
    - compute the distance to the **nearest neighbour (k=1)**. Crucially, this score is **reweighted** based on the local density of the retrieved neighbours in the memory bank to increase robustness against outliers.
    - aggregate patch scores into an **image-level anomaly score** (e.g., max or top-k percentile of the heatmap);
    - upsample patch scores to obtain an **anomaly heatmap** for localization.
- **Decision rule**:
    - final image label (normal / anomalous) is obtained by **thresholding** the image-level anomaly score (see Threshold selection below);
    - no additional supervised classifier is used.

### 2. Baseline – PaDiM (anomalib)

For each category, we use **PaDiM** via the official *anomalib* implementation:

- **Backbone**: ResNet-based feature extractor, configured to match PatchCore as closely as possible (image size, feature layers).
- **Modeling**: for each spatial position, estimate a **multivariate Gaussian** over patch features from normal training data.
- **Anomaly score**:
    - pixel-wise scores given by the **Mahalanobis distance** to the normal distribution,
    - image-level anomaly score obtained by aggregating pixel scores (e.g., maximum value).
- **Decision rule**:
    - same threshold-calibration protocol as PatchCore on Val-clean / Val-shift (per class).

Both methods share the same splits (clean and shifted) for a fair comparison.

---

## Training Setup

Both PatchCore and PaDiM are **non-parametric** with respect to anomaly scoring: there is no gradient-based optimization of the backbone.

### PatchCore-style

- **Backbone**: ResNet-50, frozen.
- **Training data**: normal images from Train-clean (and Train-shift for the adaptation experiments).
- **Procedure**:
    - forward pass to extract patch features for all training images;
    - build the memory bank (with coreset sampling).
- **Main hyperparameters**:
    - input image size,
    - backbone layers used for features,
    - patch size and stride,
    - coreset fraction,
    - distance metric (e.g., Euclidean) for nearest neighbour.

### PaDiM baseline

- **Training data**: normal images from Train-clean (and Train-shift for adaptation).
- **Procedure**:
    - extract patch features and estimate mean and covariance per spatial location;
    - no gradient-based optimization.
- **Hyperparameters**:
    - backbone layers used,
    - dimensionality reduction of features,
    - Mahalanobis distance configuration.

---

## Threshold Selection and Evaluation Metrics

### Threshold selection (image-level)

For each **class** and each **method**:

1. Compute the image-level anomaly score on **Val-clean** (normal + anomalous).
2. Explore a range of thresholds and select the one that **maximizes F1** on Val-clean.
3. Use this threshold to obtain binary labels on **Test-clean** and on **Test-shift (no adaptation)**.
4. In the adaptation experiments, repeat the same procedure on **Val-shift** and evaluate on **Test-shift (with adaptation)**.

This procedure yields one **per-class threshold** per method and per domain setting.

### Metrics

We evaluate both **detection** (image-level) and **localization** (pixel-level) for PatchCore and PaDiM, in both clean and shifted domains.

**Image-level metrics**

- **AUROC**: ability to separate normal vs anomalous images across thresholds.
- **AUPRC**: precision–recall behaviour for the anomalous class in the imbalanced setting.
- **Thresholded performance** at the selected threshold:
    - F1 score,
    - accuracy, precision, recall.

**Pixel-level metrics**

- **Pixel-wise AUROC**: discrimination between defective and non-defective pixels using ground-truth masks.
- **PRO (Per-Region Overlap)**: overlap between predicted anomalous regions and ground-truth defect regions, computed as the area under the PRO–threshold curve.

All metrics are reported **per class** (hazelnut, carpet, zipper) and as **macro-averages** across the 3 categories, to avoid hiding per-class performance differences while still providing a global summary.

### Schema del progetto

![Schema del progetto](Schema.png)

## References

### Reference Papers
- **PatchCore**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022 [https://arxiv.org/abs/2106.08265](https://arxiv.org/abs/2106.08265)

- **PaDiM**: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", 2021 [https://arxiv.org/abs/2011.08785](https://arxiv.org/abs/2011.08785)

- **MVTec AD**: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019 [https://ieeexplore.ieee.org/document/8954181](https://ieeexplore.ieee.org/document/8954181)

- **MVTec AD 2**: Bergmann et al., "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection", arXiv:2503.21622, 2025. [https://arxiv.org/abs/2503.21622](https://arxiv.org/abs/2503.21622)

- **Zhang et al.** – Prototypical Residual Networks for Anomaly Detection and Localization. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. [https://arxiv.org/abs/2212.02031] (https://arxiv.org/abs/2212.02031)

### Unified / Multi-class Anomaly Detection (Related Works)
- **CADA**: Guo J., Han H., Lu S., Zhang W., Li H. – *Absolute-Unified Multi-Class Anomaly Detection via Class-Agnostic Distribution Alignment*. arXiv:2404.00724, 2024. [https://arxiv.org/abs/2404.00724](https://arxiv.org/abs/2404.00724)

- **HierCore**: Heo J., Kang P. – *Multi-class Image Anomaly Detection for Practical Applications: Requirements and Robust Solutions*. arXiv:2508.02477, 2025. [https://arxiv.org/abs/2508.02477](https://arxiv.org/abs/2508.02477)

- **UniAD**: You Z., Cui L., Shen Y., Yang K., Lu X., Zheng Y., Le X. – *A Unified Model for Multi-class Anomaly Detection*. Advances in Neural Information Processing Systems (NeurIPS), 2022. [https://arxiv.org/abs/2206.03687](https://arxiv.org/abs/2206.03687)

- **OmniAL**: Zhao Y. – *OmniAL: A Unified CNN Framework for Unsupervised Anomaly Localization*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf)

