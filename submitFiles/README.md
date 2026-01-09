# Industrial Anomaly Detection with Localization

## A Comparative Study of PatchCore and PaDiM under Clean and Shifted Domains

---

## 👥 Authors

- Ivan Necerini (s345147)
- Jacopo Rialti (s346357)
- Fabio Veroli (s336301)

## 📁 Dataset Links

The datasets are available at the following links (personal Google Drive):

| Dataset                   | Link                 | Description                                                         |
| ------------------------- | -------------------- | ------------------------------------------------------------------- |
| **MVTec AD (Clean)**      | `[INSERT LINK HERE]` | Original MVTec AD dataset (Hazelnut, Carpet, Zipper)                |
| **MVTec-Shift (Shifted)** | `[INSERT LINK HERE]` | Synthetically shifted version with geometric/photometric transforms |

---

##  Notebooks Execution Order

Before running the notebooks, make sure to have the datasets downloaded and available in your Google Drive. 

To reproduce the results, execute notebooks **in sequential order** for full reproducibility, all the common source code used in the notebooks is available in the folder `src/`. 
Load the notebooks in Colab and run them in order. The notebooks will create the necessary folders and files and results will be saved on your Google Drive at the end of the experiment. 

| Step | Notebook                                     | Purpose                                | Output                                 |
| ---- | -------------------------------------------- | -------------------------------------- | -------------------------------------- |
| 1    | `01_data_exploration.ipynb`                  | Dataset analysis and visualization     | Exploratory plots                      |
| 2    | `02_data_preparation.ipynb`                  | Train/Val/Test split (80/20, seed=42)  | `data/processed/clean_splits.json`     |
| 3    | `03_domain_shift_generation.ipynb`           | Generate MVTec-Shift dataset           | `data/shifted/`, `shifted_splits.json` |
| 4    | `04_patchcore_clean.ipynb`                   | Train PatchCore on clean domain        | `outputs/models/patchcore_*_clean.npy` |
| 5    | `05_padim_clean.ipynb`                       | Train PaDiM on clean domain            | `outputs/models/padim_*_clean.pt`      |
| 6    | `06_evaluation_clean.ipynb`                  | Evaluate both on Test-clean            | `outputs/results/clean_results.json`   |
| 7    | `07_evaluation_shifted_no_adaptation.ipynb`  | Evaluate on Test-shift (no adaptation) | `shifted_no_adaptation_results.json`   |
| 8    | `08_evaluation_shifted_threshold_only.ipynb` | Threshold-only adaptation              | `shifted_threshold_only_results.json`  |
| 9    | `09_full_shift_adaptation.ipynb`             | Full model retraining on shifted       | Models + results for full adaptation   |
| 10   | `10_global_model_clean.ipynb`                | Model-Unified setting                  | `patchcore_global_clean.npy`           |
| 11   | `11_coreset_ratio_analysis_on_shift.ipynb`   | Coreset ablation (1%, 5%, 10%)         | Coreset comparison results             |

---

## 🗂️ Project Structure

```
AD/
├── configs/
│   └── experiment_config.yaml     # Parameters and Hyperparameters (seed=42, coreset=0.05, etc.)
├── data/
│   ├── processed/
│      ├── clean_splits.json      # Train/Val/Test indices for clean domain
│      └── shifted_splits.json    # Splits for shifted domain
├── notebooks/                     # Jupyter notebooks (01-11)
├── src/
│   ├── data/                      # Dataset, transforms, splitter
│   ├── models/                    # PatchCore wrapper, PaDiM wrapper
│   ├── metrics/                   # Evaluator, PRO metric
│   └── utils/                     # Config loader, visualization
├── outputs/
│   ├── models/                    # Trained models (.npy, .pt)
│   ├── results/                   # Evaluation metrics (.json, .csv)
│   ├── thresholds/                # Calibrated thresholds per experiment
│   └── visualizations/            # ROC curves, confusion matrices, heatmaps
├── report/                        # Paper PDF file
└── doc/                   # Submission materials
```

---

## 📊 Supplementary Materials

### Models

| Path                                   | Content                           |
| -------------------------------------- | --------------------------------- |
| `outputs/models/patchcore_*_clean.npy` | PatchCore memory banks (clean)    |
| `outputs/models/padim_*_clean.pt`      | PaDiM Gaussian parameters (clean) |
| `outputs/models/*_shift.npy/.pt`       | Fully adapted models              |
| `outputs/models/*_coreset_01/10.*`     | Coreset ablation models           |

### Results

| Path                                               | Content                  |
| -------------------------------------------------- | ------------------------ |
| `outputs/results/clean_results.json`               | Clean domain metrics     |
| `outputs/results/shifted_*_results.json`           | All shift scenarios      |
| `outputs/results/global_results.json`              | Unified model metrics    |
| `outputs/results/patchcore_coreset_comparison.csv` | Coreset ablation summary |

### Thresholds

| Path                                                 | Content                 |
| ---------------------------------------------------- | ----------------------- |
| `outputs/thresholds/patchcore_clean_thresholds.json` | Clean thresholds        |
| `outputs/thresholds/shift_threshold_only_*.json`     | Recalibrated thresholds |

### Visualizations

| Path                                              | Content                                |
| ------------------------------------------------- | -------------------------------------- |
| `outputs/visualizations/06_evaluation_clean/`     | ROC curves, confusion matrices (clean) |
| `outputs/visualizations/shifted_no_adaptation/`   | No-adaptation results                  |
| `outputs/visualizations/shifted_full_adaptation/` | Full adaptation results                |
| `outputs/visualizations/global/`                  | Unified model + shortcut analysis      |
| `outputs/visualizations/coreset/`                 | Coreset trade-off plots                |

---

## ⚙️ Key Hyperparameters

| Parameter        | Value                        | Location                         |
| ---------------- | ---------------------------- | -------------------------------- |
| Random seed      | 42                           | `configs/experiment_config.yaml` |
| Input resolution | 224×224                      | All notebooks                    |
| Backbone         | ResNet-50 (ImageNet, frozen) | PatchCore/PaDiM                  |
| Coreset ratio    | 0.05 (default)               | PatchCore                        |
| PaDiM n_features | 100                          | PaDiM                            |
| Threshold search | 1000 steps, F1-optimal       | Evaluation notebooks             |

---




