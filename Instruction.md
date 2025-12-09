# Instructions
## 1.2 Scope, Common Guidelines, and Deliverables (Applies to All Projects)

*** Documentation, Report, and Submission ***
Each group must maintain comprehensive documentation of design, training,
and evaluation, including preprocessing; the train/validation/test split with percentages and the underlying rationale; validation criteria and decision thresholds; loss functions and optimization strategies; quantitative and qualitative
results; and an error/confusion analysis with a discussion of failure cases. The
final report (6–8 pages) must present the application context and operational
definitions, the dataset with statistics, the adopted machine learning techniques
and hyper-parameters, the training/validation protocols, the performance metrics, the results, and a balanced discussion of limitations and possible future developments. The submission package must include the full project code and/or
notebooks, the weights of at least one trained model, scripts to regenerate all
tables and figures, and (when the dataset is not publicly available) a copy of
or a link to the dataset; it must also contain a README sufficient for full
reproducibility (covering setup, training, and evaluation) and the compiled selfassessment checklist. Optionally, groups may also provide an experiment log
(e.g., using the provided Sample ML tracker spreadsheet) and supplementary
material such as additional figures, videos, or other artifacts that support the
evaluation.

*** Assessment and Grading (Applies to All Projects) ***
Grading is based on the following pillars (building on the documentation and
results required above, without repeating them):
• Reproducibility & documentation: completeness and clarity of code,
README, seeds, and scripts.
• Experimental rigor: sound splits and protocols, well-motivated design
choices, and ablation studies where appropriate.
• Results & error analysis: quality of quantitative and qualitative evaluation, including confusion/error analysis and discussion of failure cases.
• Critical discussion: insight into limitations, bias, and generalization,
with well-argued future work.

---

# Assignment
## 2.2 Detection of Anomalies with Possible Localization

**Objective**
Design and implement a computer vision system capable of detecting anomalies
in a controlled visual domain. In this context, anomaly detection refers to the
automatic identification of images or regions that deviate from the normal appearance learned from reference data. The system must therefore learn what is
normal—based on images acquired under controlled conditions—and recognize
visual deviations that indicate potential defects, irregularities, or unexpected elements. Consider both image-level (normal/anomalous classification) and, when
possible, pixel-level (defect localization) analyses.

**Dataset**
Explicitly define what is “normal” and what constitutes an “anomaly.” Collect a
large number of normal images and a limited number of anomalous ones; provide
masks of defective regions for a subset of anomalous cases. May be collected
ad hoc or derived from a public source; must include, for each instance, the
image-level label, the data source, and—when available—region or pixel-level
annotations (masks).

**Experimental Plan**
• Phase 1: Train exclusively on normal data under controlled conditions
(e.g., homogeneous illumination, clean surfaces) to learn nominal appearance and calibrate decision thresholds.
• Phase 2: Evaluate robustness under domain shift (illumination, viewpoint,
camera, background). Train and test on new normal and anomalous images acquired under these conditions; report impact on detection and localization. The dataset may be synthetically altered to emulate the target domain (e.g., photometric changes, sensor noise, geometric perturbations,
background substitutions), but alterations must correspond to plausible
real-world cases and be documented. Propose and evaluate techniques to
mitigate observed degradation, with clear justification and quantitative
evidence.

