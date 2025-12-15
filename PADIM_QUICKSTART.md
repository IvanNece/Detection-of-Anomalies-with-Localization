# PaDiM Baseline - Phase 4

**Status:** ✅ Complete and Ready

---

## 🎯 Overview

**PaDiM** (Probabilistic Anomaly Detection with Multi-scale features) implementation as baseline for comparison with PatchCore.

**Method:** Learns normal appearance using Gaussian statistics → Computes Mahalanobis distance for anomaly detection

---

## 📦 Implementation

```
src/models/padim_wrapper.py              # Model wrapper
src/data/anomalib_adapter.py             # Data adapter (SHARED)
notebooks/05_padim_baseline_clean.ipynb  # Training notebook (Colab-ready)
scripts/{test_padim_setup,padim_example}.py
```

**Features:**
- Config-driven from `experiment_config.yaml`
- Consistent interface with PatchCore
- Built-in tests and validation
- Auto-download results

---

## 🚀 Usage

### 1. Open in Colab
Click badge in `notebooks/05_padim_baseline_clean.ipynb`

### 2. Run All Cells
- Trains for 3 classes (hazelnut, carpet, zipper)
- Saves to `outputs/models/`
- Tests predictions
- Creates download ZIP

**Time:** ~10 min (GPU) / ~30 min (CPU)

---

## 📊 Configuration

`configs/experiment_config.yaml`:
```yaml
padim:
  backbone: 'resnet50'
  layers: ['layer2', 'layer3']  # Same as PatchCore
  n_features: 100
  distance_metric: 'mahalanobis'
```

---

## 🧠 Method

```
Training:  Normal Images → ResNet → Gaussian(μ,Σ) per location
Inference: Test Image → ResNet → Mahalanobis Distance → Score
```

**vs PatchCore:** PaDiM uses Gaussian stats + Mahalanobis | PatchCore uses feature vectors + k-NN

---

## 💻 Examples

**Load model:**
```python
from src.models import PadimWrapper
model = PadimWrapper(device='cuda')
model.load('outputs/models/padim_hazelnut_clean.pt')
score, heatmap = model.predict(image)
```

**Standalone:**
```bash
python scripts/padim_example.py
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: anomalib` | `pip install anomalib>=1.0.0` |
| `File not found: clean_splits.json` | Run notebook 02 first |
| `CUDA OOM` | Reduce batch_size |

---

## 🤝 Integration

**Shared components** (coordinate with PatchCore team):
- `src/data/transforms.py`
- `src/data/anomalib_adapter.py`
- `data/processed/clean_splits.json`
- `configs/experiment_config.yaml`

**When merging:** Add PatchCoreWrapper to `__init__.py`, verify same preprocessing

---

## ✅ Validation

Notebook tests:
1. Model loading
2. Prediction (anomalous > normal scores)
3. Visualization

---

## 📈 Next Steps

1. ✅ Train PaDiM
2. ⏳ PatchCore completion
3. ⏳ Threshold calibration
4. ⏳ Evaluation
5. ⏳ Domain shift

---

**Docs:** `src/models/README.md`, `notebooks/README.md`  
**Paper:** Defard et al., ICPR 2021
