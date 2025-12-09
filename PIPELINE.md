# Pipeline Completa - Anomaly Detection Project

## 📋 Overview del Progetto

**Dataset**: MVTec AD (3 classi: Hazelnut, Carpet, Zipper)  
**Metodi**: PatchCore (main) + PaDiM (baseline)  
**Framework**: PyTorch + Colab Notebooks  
**Fasi**: Clean Domain → Domain Shift → Adaptation

---

## 🏗️ Struttura Proposta del Progetto

```
Detection-of-Anomalies-with-Localization/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA e statistiche dataset
│   ├── 02_data_preparation.ipynb          # Split e preprocessing
│   ├── 03_domain_shift_generation.ipynb   # Creazione MVTec-Shift
│   ├── 04_patchcore_clean.ipynb           # Training PatchCore - Clean
│   ├── 05_padim_clean.ipynb               # Training PaDiM - Clean
│   ├── 06_evaluation_clean.ipynb          # Valutazione Phase 1
│   ├── 07_patchcore_shift.ipynb           # PatchCore - Shift & Adaptation
│   ├── 08_padim_shift.ipynb               # PaDiM - Shift & Adaptation
│   ├── 09_evaluation_shift.ipynb          # Valutazione Phase 2
│   └── 10_results_visualization.ipynb     # Grafici e analisi finale
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                     # MVTecDataset class
│   │   ├── transforms.py                  # Transform per clean/shift
│   │   └── splitter.py                    # Logica split Train/Val/Test
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones.py                   # ResNet feature extractors
│   │   ├── patchcore.py                   # PatchCore implementation
│   │   ├── padim_wrapper.py               # Wrapper per anomalib PaDiM
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
│   │   ├── config.py                      # Configurazioni globali
│   │   ├── logger.py                      # Logging e tracking
│   │   ├── visualization.py               # Plot heatmaps, ROC, etc.
│   │   └── reproducibility.py             # Seed setting
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py                   # Pipeline valutazione completa
│       └── error_analysis.py              # Confusion matrix, failure cases
│
├── configs/
│   ├── patchcore_config.yaml              # Hyperparameters PatchCore
│   ├── padim_config.yaml                  # Hyperparameters PaDiM
│   └── experiment_config.yaml             # Setup generale esperimenti
│
├── data/
│   ├── raw/                               # MVTec AD originale
│   ├── processed/                         # Split salvati (pickle/json)
│   └── shifted/                           # MVTec-Shift generato
│
├── outputs/
│   ├── models/                            # Memory banks, checkpoints
│   ├── thresholds/                        # Thresholds calibrati
│   ├── results/                           # Metriche JSON/CSV
│   └── visualizations/                    # Heatmaps, grafici
│
├── scripts/
│   ├── download_dataset.py                # Download MVTec AD
│   ├── generate_all_splits.py             # Genera tutti gli split
│   ├── run_experiments.py                 # Script orchestrazione completa
│   └── generate_report_tables.py          # Genera tabelle per report
│
├── tests/
│   ├── test_dataset.py
│   ├── test_patchcore.py
│   └── test_metrics.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── PIPELINE.md (questo file)
```

---

## 🔄 Pipeline Step-by-Step

### **FASE 0: Setup Iniziale**
**Obiettivo**: Preparare l'ambiente e scaricare il dataset

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
- [x] Script per scaricare MVTec AD (Kaggle API o wget)
- [x] Verificare struttura: `hazelnut/`, `carpet/`, `zipper/`
- [x] Ogni classe: `train/good/`, `test/good/`, `test/<defect>/`, `ground_truth/<defect>/`

#### Step 0.3: Configurazioni Base
- [x] File `configs/experiment_config.yaml`:
  - Seed globale: 42
  - Classi: [hazelnut, carpet, zipper]
  - Split ratios: Train 80%, Val 20%
  - Val anomaly ratio: 30%
  - Paths dataset e output

---

### **FASE 1: Data Exploration & Preparation**

#### Step 1.1: Exploratory Data Analysis (Notebook 01)
- [ ] Contare immagini per classe e split
- [ ] Visualizzare esempi normal/anomalous per ogni classe
- [ ] Statistiche dimensioni immagini
- [ ] Distribuzione tipi di difetti per classe
- [ ] Visualizzare ground truth masks

#### Step 1.2: Implementare Data Split (Notebook 02)
- [ ] `src/data/splitter.py`:
  - Funzione `create_clean_split(class_name, train_ratio, val_anomaly_ratio, seed)`
  - Input: cartelle MVTec
  - Output: dizionari con paths immagini/masks
  - Train-clean: 80% di `train/good`
  - Val-clean: 20% `train/good` + 30% anomalous da `test/`
  - Test-clean: restante `test/good` + restante anomalous
- [ ] Salvare split in `data/processed/clean_splits.json`
- [ ] Verificare bilanciamento e conteggi

#### Step 1.3: Implementare Dataset Class (Notebook 02)
- [ ] `src/data/dataset.py`:
  - `MVTecDataset(image_paths, mask_paths, transform, phase)`
  - `__getitem__`: restituire (image, mask, label, image_path)
  - Supportare phase='train' (solo normal), 'val', 'test'
- [ ] `src/data/transforms.py`:
  - `get_clean_transforms(train=False)`: resize, normalize ImageNet
  - Testare su esempi

---

### **FASE 2: Domain Shift Generation**

#### Step 2.1: Implementare Trasformazioni Shift (Notebook 03)
- [ ] `src/data/transforms.py`:
  - `ShiftTransform`:
    - **Geometric** (apply to image + mask):
      - Random rotation [-10°, +10°]
      - Random resized crop [0.9, 1.0]
      - Small translation
    - **Photometric** (solo image):
      - ColorJitter (brightness, contrast, saturation [0.7, 1.3])
      - Gaussian blur (kernel 3-5)
      - Gaussian noise σ [0.01, 0.05]
    - Opzionale: vignetting simulato
  - Usare `torchvision.transforms` e `albumentations` per consistency

#### Step 2.2: Generare MVTec-Shift Dataset (Notebook 03)
- [ ] Per ogni split (Train-clean, Val-clean, Test-clean):
  - Applicare `ShiftTransform` a tutte le immagini + masks
  - Salvare in `data/shifted/` mantenendo struttura
- [ ] Creare `data/processed/shifted_splits.json`
- [ ] Visualizzare esempi clean vs shifted

---

### **FASE 3: PatchCore - Clean Domain**

#### Step 3.1: Implementare Backbone (Notebook 04)
- [ ] `src/models/backbones.py`:
  - `ResNet50FeatureExtractor`:
    - Pre-trained ImageNet
    - Hook su layer intermedi (es: layer2, layer3)
    - Forward → multi-scale features
    - Average pooling locale per ridurre dimensionalità
    - Concatenazione e upsampling a risoluzione comune

#### Step 3.2: Implementare Memory Bank (Notebook 04)
- [ ] `src/models/memory_bank.py`:
  - `GreedyCoresetSubsampling`:
    - Input: patch features da Train-clean
    - Greedy selection: iterativamente seleziona patch che massimizza min-distance a già selezionati
    - Target: 1-10% delle patches totali
    - Output: memoria ridotta
  - `MemoryBank`:
    - Store features + metodi nearest neighbour
    - Reweighting basato su densità locale

#### Step 3.3: Implementare PatchCore (Notebook 04)
- [ ] `src/models/patchcore.py`:
  - `PatchCore`:
    - `fit(train_loader)`: estrai features, build memory bank
    - `predict(image)`: 
      - Estrai patch features
      - Nearest neighbour per ogni patch
      - Genera anomaly heatmap
      - Aggregate image-level score (max o top-k percentile)
    - `save/load`: salvare memory bank

#### Step 3.4: Training PatchCore - Clean (Notebook 04)
- [ ] Per ogni classe (hazelnut, carpet, zipper):
  - Caricare Train-clean
  - Fit PatchCore
  - Salvare memory bank in `outputs/models/patchcore_{class}_clean.pkl`
- [ ] Tempo di training e dimensione memory bank

---

### **FASE 4: PaDiM - Clean Domain**

#### Step 4.1: Setup PaDiM con Anomalib (Notebook 05)
- [ ] `src/models/padim_wrapper.py`:
  - Wrapper per `anomalib.models.Padim`
  - Configurazione backbone ResNet
  - Interfaccia unificata con PatchCore

#### Step 4.2: Training PaDiM - Clean (Notebook 05)
- [ ] Per ogni classe:
  - Adattare dataloader per anomalib
  - Fit PaDiM su Train-clean
  - Salvare modello in `outputs/models/padim_{class}_clean.pkl`

---

### **FASE 5: Threshold Calibration & Evaluation - Clean Domain**

#### Step 5.1: Implementare Threshold Selection (Notebook 06)
- [ ] `src/metrics/threshold_selection.py`:
  - `calibrate_threshold(scores_normal, scores_anomalous)`:
    - Gridsearch thresholds
    - Calcola F1 per ogni threshold
    - Return best threshold che massimizza F1
  - Applicare su Val-clean per ogni classe e metodo

#### Step 5.2: Implementare Metriche (Notebook 06)
- [ ] `src/metrics/image_metrics.py`:
  - `compute_auroc(y_true, scores)`
  - `compute_auprc(y_true, scores)`
  - `compute_f1_at_threshold(y_true, scores, threshold)`
  - `compute_accuracy_precision_recall(y_true, y_pred)`

- [ ] `src/metrics/pixel_metrics.py`:
  - `compute_pixel_auroc(masks_true, heatmaps)`
  - `compute_pro(masks_true, heatmaps)` (Per-Region Overlap)

#### Step 5.3: Valutazione su Test-Clean (Notebook 06)
- [ ] Per ogni classe e metodo:
  - Predict su Test-clean
  - Applicare threshold da Val-clean
  - Calcolare tutte le metriche image-level e pixel-level
- [ ] Salvare risultati in `outputs/results/clean_results.json`
- [ ] Tabelle per classe e macro-average

#### Step 5.4: Visualizzazioni (Notebook 06)
- [ ] Top-K best/worst predictions
- [ ] Heatmaps overlay su immagini anomalous
- [ ] Confusion matrix
- [ ] ROC e Precision-Recall curves

---

### **FASE 6: Domain Shift - No Adaptation**

#### Step 6.1: Evaluation su Test-Shift senza Adaptation (Notebook 09)
- [ ] Usare modelli trained su Train-clean
- [ ] Usare thresholds calibrati su Val-clean (stessi di FASE 5)
- [ ] Predict su Test-shift
- [ ] Calcolare metriche
- [ ] Salvare in `outputs/results/shift_no_adaptation_results.json`
- [ ] **Analisi degradazione performance** rispetto a Test-clean

---

### **FASE 7: Domain Shift - With Adaptation**

#### Step 7.1: Re-training su Train-Shift (Notebook 07, 08)
- [ ] **PatchCore**:
  - Ricostruire memory bank usando Train-shift
  - Salvare in `outputs/models/patchcore_{class}_shift.pkl`
- [ ] **PaDiM**:
  - Re-fit su Train-shift
  - Salvare in `outputs/models/padim_{class}_shift.pkl`

#### Step 7.2: Re-calibrate Thresholds su Val-Shift (Notebook 07, 08)
- [ ] Predict su Val-shift con nuovi modelli
- [ ] Calibrare nuovi thresholds massimizzando F1
- [ ] Salvare in `outputs/thresholds/shift_thresholds.json`

#### Step 7.3: Evaluation su Test-Shift con Adaptation (Notebook 09)
- [ ] Predict con modelli adapted
- [ ] Applicare thresholds da Val-shift
- [ ] Calcolare metriche
- [ ] Salvare in `outputs/results/shift_with_adaptation_results.json`
- [ ] **Analisi miglioramento** rispetto a no-adaptation

---

### **FASE 8: Global Model (Opzionale)**

#### Step 8.1: Training Global Model (Extra)
- [ ] Pool tutti i normal images dalle 3 classi
- [ ] Train PatchCore/PaDiM sul pool
- [ ] Evaluate su ogni classe separatamente
- [ ] **Analisi limitazioni** di normality eterogenea

---

### **FASE 9: Error Analysis & Results Visualization**

#### Step 9.1: Error Analysis (Notebook 10)
- [ ] `src/evaluation/error_analysis.py`:
  - Identifica failure cases (FP, FN)
  - Analizza pattern comuni (es: difetti piccoli, occlusioni)
  - Confusion matrix dettagliate per tipo di difetto

#### Step 9.2: Comparative Analysis (Notebook 10)
- [ ] Tabelle comparative:
  - PatchCore vs PaDiM
  - Clean vs Shift (no-adapt) vs Shift (adapt)
  - Per-class performance
- [ ] Grafici:
  - Bar plots metriche per classe
  - Scatter plots AUROC image vs pixel
  - Heatmap confronto
  - ROC curves multiple

#### Step 9.3: Ablation Studies (Notebook 10)
- [ ] **PatchCore**:
  - Variare coreset fraction (1%, 5%, 10%)
  - Variare backbone layers
  - Variare aggregation method (max, top-k)
- [ ] **Domain Shift**:
  - Impatto singole trasformazioni (solo photometric, solo geometric)
  - Severity level delle trasformazioni

---

### **FASE 10: Documentation & Report**

#### Step 10.1: README Completo
- [ ] Setup instructions
- [ ] Comandi per download dataset
- [ ] Comandi per train/eval
- [ ] Struttura output
- [ ] Riproducibilità (seed, versions)

#### Step 10.2: Scripts per Reproducibility
- [ ] `scripts/run_experiments.py`: orchestrazione completa
- [ ] `scripts/generate_report_tables.py`: genera tutte le tabelle/figure per report

#### Step 10.3: Report Finale (6-8 pagine)
- [ ] Application context & operational definitions
- [ ] Dataset statistics & split rationale
- [ ] ML techniques & hyperparameters
- [ ] Training/validation protocols
- [ ] Performance metrics & results
- [ ] Error analysis & failure cases
- [ ] Critical discussion: limitations, bias, generalization
- [ ] Future work

---

## 📊 Analisi Richieste (Confermate)

### Analisi Obbligatorie da Implementare:

1. **Performance on Test-clean** (clean domain)
   - Tutte le metriche per ogni classe e metodo

2. **Performance on Test-shift - No Adaptation**
   - Stessi thresholds da Val-clean
   - Studio degradazione performance

3. **Performance on Test-shift - With Adaptation**
   - Re-training su Train-shift + re-calibration su Val-shift
   - Studio recovery performance

4. **Global Model Comparison** (extra)
   - Modello unico su pool di tutte le classi
   - Confronto con modelli per-class

---

## 💡 Proposte Analisi Aggiuntive per Eccellere (30/30)

### A. Analisi Interpretabilità e Spiegabilità
- **Attention Maps delle Features**: visualizzare quali regioni del backbone contribuiscono di più
- **T-SNE/UMAP delle Patch Embeddings**: visualizzare clustering normal vs anomalous nello spazio latente
- **Feature Importance Analysis**: quali layer ResNet sono più discriminativi

### B. Robustness Analysis Avanzata
- **Severity Levels**: testare shift a diverse intensità (mild, moderate, severe)
- **Single-Transformation Analysis**: isolare l'effetto di ogni trasformazione (rotation, noise, blur, etc.)
- **Cross-Shift Generalization**: train su shift A, test su shift B

### C. Analisi Statistica e Confidence
- **Bootstrap Confidence Intervals**: intervalli di confidenza per metriche
- **Statistical Significance Tests**: t-test tra PatchCore e PaDiM
- **Per-Defect Type Performance**: breakdown per tipo di difetto specifico (crack, hole, etc.)

### D. Efficienza e Scalabilità
- **Inference Time Analysis**: latenza per immagine (importante per industrial setting)
- **Memory Bank Size vs Performance Trade-off**: curva coreset fraction vs AUROC
- **Feature Dimension Reduction**: PCA/Random Projection impact

### E. Failure Mode Analysis Avanzata
- **Difficult Cases Study**: analisi approfondita dei casi dove entrambi i metodi falliscono
- **Defect Size Sensitivity**: performance vs dimensione del difetto (small, medium, large)
- **Border Effects**: performance su difetti ai bordi vs centrali

### F. Confronto con Soglie Alternative
- **Multiple Threshold Strategies**: max, mean, percentile-based
- **Per-Region Threshold**: thresholds adattivi per diverse zone dell'immagine
- **Calibration Curves**: reliability diagrams per confidence scores

---

## ⚙️ Best Practices da Seguire

### Codice Pulito e Modulare
- ✅ Type hints ovunque
- ✅ Docstrings Google-style
- ✅ Separazione concerns (data/model/evaluation)
- ✅ Config files per tutti i hyperparameters
- ✅ Logging strutturato

### Riproducibilità
- ✅ Seed fisso in `reproducibility.py` (torch, numpy, random)
- ✅ Versioni librerie in `requirements.txt`
- ✅ Save split indices, non rigenerare
- ✅ Deterministic operations (torch.backends.cudnn.deterministic = True)

### Gestione Esperimenti
- ✅ Experiment tracking (TensorBoard o WandB opzionale)
- ✅ Salvataggio automatico risultati intermedi
- ✅ Naming convention chiara per modelli/output
- ✅ Git history pulita

### Performance
- ✅ DataLoader con num_workers e pin_memory
- ✅ Batch processing quando possibile
- ✅ Caching features già estratte
- ✅ Profiling per identificare bottleneck

---

## 📅 Timeline Suggerita

| Settimana | Focus | Deliverables |
|-----------|-------|--------------|
| 1 | Setup + EDA + Data Prep | Notebooks 01-03, structure completa |
| 2 | PatchCore Clean | Notebook 04, implementation + training |
| 3 | PaDiM Clean + Eval Clean | Notebooks 05-06, primi risultati |
| 4 | Domain Shift | Notebooks 07-09, shift generation + eval |
| 5 | Analisi Avanzate + Ablation | Notebook 10, analisi extra |
| 6 | Report + Refinement | Documento finale, cleanup |

---

## 🎯 Checklist Pre-Consegna

- [ ] Tutti i notebook eseguiti end-to-end senza errori
- [ ] README completo e testato
- [ ] Codice commentato e pulito
- [ ] Test automatici passano
- [ ] Requirements.txt aggiornato
- [ ] Almeno 1 modello trained salvato per metodo
- [ ] Script `run_experiments.py` funzionante
- [ ] Tutte le figure/tabelle del report generate
- [ ] Self-assessment checklist compilato
- [ ] Report 6-8 pagine completo
- [ ] Git repo organizzato (.gitignore corretto)

---

## 📚 Risorse Chiave

### Papers di Riferimento
- **PatchCore**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
- **PaDiM**: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", 2021
- **MVTec AD**: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019

### Implementazioni di Riferimento
- Anomalib: https://github.com/openvinotoolkit/anomalib
- PatchCore original: https://github.com/amazon-science/patchcore-inspection

---

*Pipeline preparata per progetto magistrale - focus su rigor scientifico, reproducibility, e best practices industriali.*
