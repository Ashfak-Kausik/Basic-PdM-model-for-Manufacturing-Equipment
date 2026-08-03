# Basic PdM model for Manufacturing Equipment — reproduction, audit & clean baseline

This repo contains **two separate pipelines** built from `ML main.ipynb`, the
code underlying the paper *"Optimizing Predictive Maintenance of CNC Machines
Using Modified Ensemble Learning and Feature Engineering: A Data-Driven Case
Study"* (`documents/Optimizing_Predictive_Maintenance_of_CNC_Machines_...pdf`):

1. **`src/reproduce.py`** — a *faithful, bug-preserving* reproduction of the
   notebook (see "Faithful reproduction" below). This is the audit trail and
   is not modified by the clean pipeline.
2. **`src/config.py` / `src/data.py` / `src/features.py` / `src/models.py` /
   `src/evaluate.py` / `src/run_baseline.py`** — a *new, clean, canonical*
   pipeline (see "Clean canonical pipeline" below) that fixes the issues the
   audit found (leaky/unscaled/unstratified split, missing class weighting,
   the dead two-stage `ColumnFailure0` block) and reports its numbers
   honestly against both the paper and the faithful reproduction.

## Layout

```
requirements.txt                 Python dependencies (shared by both pipelines)
src/reproduce.py                  Faithful reproduction, extracted line-for-line from ML main.ipynb
src/config.py                     Canonical config for the clean pipeline (SEED, features, hyperparameters, paths)
src/data.py                       Data loading, feature-set construction, stratified split
src/features.py                   Engineered-feature formulas (RelationTemperature, Power, WearRPM, ToolWearTorque)
src/models.py                     Factory for the five scaled sklearn Pipelines
src/evaluate.py                   Metrics, ensemble derivation, figure generation
src/run_baseline.py               Entry point for the clean pipeline (single command, see below)
data/ai4i2020.csv                 AI4I 2020 Predictive Maintenance dataset (see Dataset section)
outputs/results_reproduced.csv    Faithful reproduction: full confusion matrix + metrics (untouched by the clean pipeline)
outputs/AUDIT.md                  Faithful reproduction vs. paper's Tables 4/5/8/9 + discrepancy list (untouched)
outputs/results_clean.csv         Clean pipeline: full confusion matrix + metrics, long format
outputs/tables/                   Paper-style tables (table4/5/6/8/10) from the clean pipeline
outputs/figures/                  ROC curve, confusion matrices, correlation heatmaps from the clean pipeline
outputs/split_class_balance.txt   Train/test row + failure counts under the clean pipeline's stratified split
outputs/DELTA_vs_paper.md         Clean numbers vs. paper vs. faithful reproduction, side by side
documents/                        Paper PDF, reviewer letter/review doc
ML main.ipynb                     Original notebook (source of truth for src/reproduce.py)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Dataset

`data/ai4i2020.csv` is the AI4I 2020 Predictive Maintenance Dataset
(10,000 rows, UCI ML Repository dataset #601 / Kaggle "AI4I 2020 Predictive
Maintenance Dataset"). It was downloaded directly from the UCI archive for
this reproduction:

```bash
curl -L -o ai4i.zip "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
unzip ai4i.zip -d /tmp/ai4i_extracted
cp /tmp/ai4i_extracted/ai4i2020.csv data/ai4i2020.csv
```

If you need to place the file manually (e.g. no network access), download
`ai4i2020.csv` from either:
- UCI: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Kaggle: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

and place it at `data/ai4i2020.csv` (10,000 rows, 14 columns, header starts
with `UDI,Product ID,Type,Air temperature [K],...`).

## Faithful reproduction (`src/reproduce.py`)

```bash
.venv/bin/python src/reproduce.py
```

This writes `outputs/results_reproduced.csv` (full confusion matrix +
accuracy/precision/recall/F1/specificity/false-negative-rate for the
**positive (failure) class**, for every base model on both the original and
feature-engineered data, plus the Traditional Voting and OR-Logic
ensembles) and prints a summary table to stdout.

Per the task that produced it, this script changes nothing about the
methodology — no feature scaling, no stratified split, no hyperparameter
correction, no fix for the bugs documented in `outputs/AUDIT.md` (in
particular the dead/stale "stage two" ensemble column and the full-dataset
min/max envelope leakage). Those are left exactly as found so the audit
reflects the code as it currently exists. **This script and its outputs
(`outputs/AUDIT.md`, `outputs/results_reproduced.csv`) are not touched by
the clean pipeline below** — they remain the permanent audit trail.

## Clean canonical pipeline (`src/run_baseline.py`)

A new, from-scratch pipeline that adopts one deliberate, documented config
(all of it in `src/config.py`, one-line editable) instead of the notebook's
inconsistent/undocumented one, and drops the two-stage `ColumnFailure0` /
risk-factor block entirely (the audit found it leaked test data into its
thresholds and was dead code anyway — see `outputs/AUDIT.md` section 4).

**Canonical config** (`src/config.py`):
- `SEED = 42`, `train_test_split(test_size=0.25, random_state=SEED, stratify=y)` — genuinely stratified, unlike the notebook.
- `ORIGINAL_FEATURES`: the five sensor columns only (`Type_encoded` deliberately excluded, unlike the notebook — see AUDIT.md 2c).
- `MODIFIED_FEATURES`: the five original + `RelationTemperature`, `Power (W)`, `WearRPM`, `ToolWearTorque` (nine total).
- Every model is `Pipeline([('scaler', StandardScaler()), ('clf', ...)])` — scaling fit on the training fold only, no leakage.
- `SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED)`
- `RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=SEED)` (unrestricted depth)
- `KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)`
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)`
- `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED)`
- Ensembles are derived from the five models' stacked test-set predictions: Traditional Voting = failure if `sum(preds) >= 3`; OR-Logic = failure if `sum(preds) >= 1`.

Note: `class_weight='balanced'` was tried on SVM/RF/LR and then **removed**.
With it, SVM and LR individually over-fired (SVM ~190 FP, LR ~350 FP on the
modified test set), and since OR-Logic unions all five models' positives, it
inherited the union of their false alarms — recall rose to ~95% but
precision collapsed to ~16% and specificity fell to ~83%, breaking the
paper's actual claim of high recall *while holding specificity near 99.5%*.
That balanced-weights run is preserved as a cost-sensitive baseline (for
reviewer comment R1.6) in `outputs/balanced_variant/` — see its `README.txt`.
The current `src/config.py` (no class weighting) is what `src/run_baseline.py`
runs by default.

Run it with:

```bash
.venv/bin/python src/run_baseline.py
```

This single command produces `outputs/results_clean.csv`, the paper-style
table CSVs in `outputs/tables/`, the figures in `outputs/figures/`,
`outputs/split_class_balance.txt`, and `outputs/DELTA_vs_paper.md` (a headline
summary, then one wide table — every model and ensemble, on the modified
feature set, with paper/faithful-reproduction/clean-run numbers side by
side — plus a narrative check of whether the paper's core OR-logic-vs-voting
contrast still holds). See that file for the full comparison and an honest
account of where and why the clean config's numbers diverge from the paper's
story.

Out of scope for this pipeline (left for later tasks): SMOTE, SHAP,
confidence intervals, latency timing, MLP, and any two-stage/`ColumnFailure0`/
risk-factor logic.
