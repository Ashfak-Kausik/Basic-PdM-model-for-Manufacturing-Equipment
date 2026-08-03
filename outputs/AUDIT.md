# Audit: `ML main.ipynb` code vs. the paper

**Scope:** this audits what the notebook's code *actually does*, reproduced
verbatim in `src/reproduce.py`, against the paper's Methods (§3.2, §4.2,
§4.4) and Results (Tables 4, 5, 8, 9) in
`documents/Optimizing_Predictive_Maintenance_of_CNC_Machines_...pdf`.
Nothing about the methodology was changed to produce these numbers — see
`src/reproduce.py` for the line-by-line port, with cell numbers cited in
comments.

Environment used: Python 3.10.12, pandas 2.3.3, scikit-learn 1.7.2 (paper
states "Scikit-learn v1.0+", "Pandas v1.3+" — this reproduction uses newer
minor versions than the paper's stated floor; no evidence this materially
affects results, see §6 below on the one exact match).

Dataset: `data/ai4i2020.csv`, downloaded from UCI ML Repository (dataset
#601), 10,000 rows — matches the paper's Table 3 (`Count = 10000` for all
fields) and the 339-failure / 3.39% imbalance described in §3.2 and §4.1.

---

## 1. Headline result

**The notebook's code does not reproduce the paper's numbers**, for every
model except Gradient Boosting on the feature-engineered data (which
matches exactly — see §6, this is important corroborating evidence, not
a coincidence). The root cause is that the paper's Methods sections
describe a different experimental configuration than what the code
implements — different hyperparameters for 4 of 5 base models, and a
different train/test split protocol than the one described in §3.2/§4.4.
Section 2-4 below itemize every discrepancy found.

---

## 2. (a) Hyperparameter discrepancies: code vs. paper §4.2

| Model | Paper §4.2 says | Code actually uses (`ML main.ipynb`, cells 17-21 & 35-51) | Match? |
|---|---|---|---|
| SVM | RBF kernel, **C=1.0**, γ = scale heuristic, **`class_weight='balanced'`** | `svm.SVC(C=10, degree=2)` → kernel defaults to `'rbf'`, so **`degree=2` is a no-op** (only affects `kernel='poly'`); `C=10` not 1.0; no class weighting | ❌ No |
| Random Forest | "unrestricted tree depth and minimal node-splitting constraints"; 100 trees, √n_features subsampling | `RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)` — depth **is** restricted (15), splitting constraints are **not** minimal | ❌ No |
| KNN | **k = 5**, Euclidean, uniform weights | `KNeighborsClassifier(n_neighbors=10)` | ❌ No (k=10, not 5) |
| Logistic Regression | L2, **C=1.0**, `solver='lbfgs'`, **`max_iter=1000`**, **`class_weight='balanced'`** | `LogisticRegression()` — all sklearn defaults: C=1.0 (matches by coincidence), solver='lbfgs' (matches), **`max_iter=100`** (not 1000 — the run in this reproduction hits `ConvergenceWarning: lbfgs failed to converge`), **no class weighting** | ❌ Partial (2/5 params match by default, 2 differ, 1 missing) |
| Gradient Boosting | 100 estimators, learning_rate=0.1, max_depth=3, log-loss | `GradientBoostingClassifier()` — these are sklearn's exact defaults | ✅ Yes (only because the paper's stated config happens to equal the library defaults the code uses) |

Also relevant: §4.4 states **"All experiments were executed with a fixed
random seed (`random_state=42`)."** The code's only `random_state=42` is
on the `RandomForestClassifier`; the `train_test_split` calls (cells 16 and
34) use **`random_state=25`**, not 42, and SVM/KNN/LogisticRegression/
GradientBoostingClassifier are instantiated with **no `random_state` at
all**. See §5 for direct evidence that `random_state=25` (not 42) is what
actually produced the paper's reported numbers.

## 2. (b) Split-protocol discrepancies: code vs. paper §3.2 / §4.4

| Claim | Paper text | Code (`ML main.ipynb`, cells 16 & 34) |
|---|---|---|
| Stratification | §3.2: *"partitioned into training (75%) and testing (25%) sets **using stratified sampling** to preserve failure class proportions"*; §4.4 repeats *"stratified train–test splitting (75%–25%) to preserve class distribution"* | `train_test_split(X, y, test_size=0.25, random_state=25)` — **no `stratify=` argument at all** |
| Feature scaling | Not explicitly claimed as absent or present, but §4.2's SVM description (`γ` via "scale heuristic" on presumably standardized features) and general ML practice implies scaled inputs | **No `StandardScaler`/scaling anywhere in the notebook.** SVM, KNN, and Logistic Regression are all fit on raw, unscaled features (`Air temperature [K]` ~294-305 vs `Power (W)` ~0-9000, `WearRPM` ~0-0.5, etc. — wildly different scales) |
| Random seed | §4.4: *"fixed random seed (`random_state=42`)"* for "all experiments" | `train_test_split` uses `random_state=25` |

## 2. (c) Feature-set discrepancies

The paper (§5.3.2, Table 6/7) repeatedly describes the modified pipeline as
using **"nine features: five original parameters plus four engineered
features."** The code's actual `X` (cell 32) is built as:

```python
X = f_data.drop(columns=['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
```

This drop list removes the raw `'Type'` string column but **not**
`'Type_encoded'` (added in cell 13). So `X` actually has **10 columns**,
not 9:

```
['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
 'Torque [Nm]', 'Tool wear [min]', 'Type_encoded',
 'RelationTemperature', 'Power (W)', 'WearRPM', 'ToolWearTorque']
```

(confirmed by `src/reproduce.py`'s printed column list). The same is true
of the "raw"/baseline pass (cell 14): it includes `Type_encoded` too, so
the paper's claim of "five key operational variables" for the baseline
models (§3.2) is also off by one feature (6, not 5) in the actual code.
This is a minor discrepancy in absolute terms but it means the paper's
methods description does not match what was (apparently) trained.

---

## 3. Reproduced positive-class metrics vs. the paper's tables

All values below are for the **positive/failure class specifically**
(precision/recall/F1 with `pos_label=1`; specificity = TN/(TN+FP);
false-negative rate = FN/(FN+TP)), matching how the paper itself reports
these numbers (e.g. "RFC ... true positive rate of 83.1%, successfully
detecting 69 failures while missing 14"). Full data in
`outputs/results_reproduced.csv`.

### 3a. vs. Table 4 — baseline models on original features

| Model | Metric | Paper Table 4 | Reproduced | Match? |
|---|---|---|---|---|
| SVM | Acc / Prec / Recall / F1 / Spec | 95.00 / 85 / 5.0 / 9.0 / 99.50 | 96.88 / 100.00 / 6.02 / 11.36 / 100.00 | ❌ |
| RFC | Acc / Prec / Recall / F1 / Spec | 97.00 / 92 / 75.0 / 83.0 / 98.75 | 98.60 / 91.38 / 63.86 / 75.18 / 99.79 | ❌ |
| KNN | Acc / Prec / Recall / F1 / Spec | 96.00 / 88 / 30.0 / 45.0 / 99.00 | 97.24 / 100.00 / 16.87 / 28.87 / 100.00 | ❌ |
| LR | Acc / Prec / Recall / F1 / Spec | 95.00 / 83 / 18.0 / 30.0 / 99.25 | 97.16 / 80.00 / 19.28 / 31.07 / 99.83 | ❌ |
| GBC | Acc / Prec / Recall / F1 / Spec | 97.00 / 91 / 72.0 / 80.0 / 98.80 | 98.84 / 95.00 / 68.67 / 79.72 / 99.88 | ❌ (close on recall, but no exact match — unlike the modified-data GBM, see §6) |

None of the baseline (raw-feature) numbers reproduce. Every reproduced
accuracy is *higher* than the paper's, and every reproduced recall is
*lower*, for all 5 models — a directionally consistent pattern, not noise.

### 3b. vs. Table 5 / Table 9 — models + ensembles on feature-engineered data

| Model | Metric | Paper (Table 5 / text) | Reproduced | Match? |
|---|---|---|---|---|
| SVM | Acc/Prec/Recall/F1 (TP/FN/TN/FP) | 97.0/100/9.6/17.5 (8/75/2417/0) | 96.76/100.00/2.41/4.71 (2/81/2417/0) | ❌ |
| RFC | Acc/Prec/Recall/F1 (TP/FN/TN/FP) | 98.9/94.5/83.1/88.5 (69/14/2413/4) | 99.20/95.65/79.52/86.84 (66/17/2414/3) | ❌ |
| KNN | Acc/Recall (TP/FN) — text only, not in Table 5 | 97.28/37.3 (31/52) | 97.68/31.33 (26/57) | ❌ |
| LR | Acc/Recall (TP/FN) — text only, not in Table 5 | 97.32/25.3 (21/62) | 97.20/22.89 (19/64) | ❌ |
| GBC | Acc/Prec/Recall/F1 (TP/FN/TN/FP) | 99.0/95.5/75.9/84.4 (63/20/2414/3) | **99.08/95.45/75.90/84.56 (63/20/2414/3)** | ✅ **exact match** |
| Traditional Voting | Acc/Prec/Recall/F1 (TP/FN/TN/FP) | 99.2/100/44.6/61.7 (37/46/2417/0) | 98.04/100.00/40.96/58.12 (34/49/2417/0) | ❌ |
| OR-Logic | Acc/Prec/Recall/F1 (TP/FN/TN/FP) | 99.2/86.9/88.0/87.4 (73/10/2406/11) | 98.96/87.01/80.72/83.75 (67/16/2407/10) | ❌ |

Table 9 (MCC, FN-rate) comparison:

| Method | Metric | Paper Table 9 | Reproduced | Match? |
|---|---|---|---|---|
| Traditional Voting | Acc/Prec/Recall/F1/Spec/MCC/FNRate | 99.16/94/44.6/60/100.0/0.63/55.4 | 98.04/100.0/40.96/58.12/100.0/0.634/59.04 | ❌ |
| Proposed OR-Logic | Acc/Prec/Recall/F1/Spec/MCC/FNRate | 99.16/88/88.0/88/99.50/0.88/12.0 | 98.96/87.01/80.72/83.75/99.59/0.833/19.28 | ❌ |
| Best Individual (RFC) | Acc/Prec/Recall/F1/Spec/MCC/FNRate | 98.92/95/83.1/89/99.83/0.87/16.9 | 99.20/95.65/79.52/86.84/99.88/0.868/20.48 | ❌ |

**Every table-5/9 row fails to reproduce except Gradient Boosting**, whose
hyperparameters are the only ones where paper and code genuinely agree
(§2a). This is strong, non-coincidental evidence that the *other* four
models' mismatched hyperparameters (§2a) are the direct cause of the other
rows' mismatches — see §6.

### 3c. Table 8 — ensemble performance on raw (non-feature-engineered) data

**Not reproducible at all: there is no code for this in the notebook.**
Cells 17-21 (the raw-data pass) only fit the 5 individual models and print
`accuracy_score` — no `results_df`, no `VotingClassifier`, and no OR-logic
sum is ever built from those predictions. The `VotingClassifier` (cell 55)
and the OR-logic sum (cell 58) are only ever constructed on the
feature-engineered data (`X_train`/`X_test` from cell 34). Table 8's two
rows ("Traditional Voting" / "Proposed OR-Logic" on raw data, 97.2%/96.5%
accuracy) have no corresponding computation anywhere in `ML main.ipynb`.

---

## 4. (c) Stage-two / `ColumnFailure0` envelope: full-dataset vs. train-only, and the leakage implication

**Confirmed directly from the code (cell 74, reproduced in
`src/reproduce.py` around the `failure_0_data = f_data[...]` block): the
min/max envelope is computed over the FULL dataset (train + test rows
together), not the training split alone.**

```python
failure_0_data = f_data[f_data['Machine failure'] == 0]   # f_data = ALL 10,000 rows
max_values_failure_0 = {c: failure_0_data[c].max() for c in [...]}
min_values_failure_0 = {c: failure_0_data[c].min() for c in [...]}
f_data['ColumnFailure0'] = 0
... f_data.loc[f_data[column] > max_value, 'ColumnFailure0'] = 1 ...
... f_data.loc[f_data[column] < min_value, 'ColumnFailure0'] = 1 ...
```

`f_data` at this point still contains every row from the original
10,000-row dataset — the earlier `train_test_split` calls (cells 16, 34)
never drop rows from `f_data`/`raw_data` itself, they only split copies of
`X`/`y`. So the "envelope" bounds used to derive `ColumnFailure0` for a
given test-set row are computed using the true `Machine failure` labels
and feature values of *other test-set rows* — this is direct **train/test
leakage**: the thresholds a held-out evaluation is supposed to be blind to
are fit using information from the held-out set itself.

**Leakage implication, plainly stated:** if this `ColumnFailure0` label
were used as the target of a trained model and evaluated for generalization,
any reported test-set accuracy for that model would be **optimistically
biased** — the envelope "knows" the true failure status of test rows before
the test evaluation happens, because it directly used those test rows'
`Machine failure` values to set its bounds. Any accuracy/recall figure
attributed to a "stage two" model built this way should not be trusted as
an estimate of real-world generalization.

**However — and this matters for interpreting the paper — that leakage
turns out to be moot in the current code, because of a second, independent
bug:**

### 4a. The "stage two" ensemble never actually trains anything (dead code + stale-variable bug)

Cell 76 builds a second `X`/`y` pair (`X2`, `y2 = ColumnFailure0`). Cell 77
splits it: `X_train, X_pred, y_train, y_test = train_test_split(X2, y2, ...)`.
**No model is ever `.fit()` or `.predict()`-ed on this split** — the
variable names (`X_pred` instead of `X_test`) suggest this was meant to
feed a new model, but that step is simply missing from the notebook.

Cell 78 then runs `results_df['E_col'] = y_pred`. Because no cell between
55 and 78 reassigns `y_pred`, this `y_pred` is still whatever cell 61 set
it to: `y_pred = results_df['sum_excluding_real']` — **the stage-one
OR-logic ensemble's prediction on the original `Machine failure` target**,
not a prediction from any model trained on `ColumnFailure0`.
`src/reproduce.py` reproduces this exactly and confirms it programmatically:

```
sum_excluding_real2 identical to sum_excluding_real for every row: True
```

Cell 80's `sum_excluding_real2` sums `SVM + RF + KNN + LR + GBM + E_col`
and thresholds at `> 0`. Since `E_col` is an exact copy of the already-OR'd
`sum_excluding_real` column, adding it changes the sum's magnitude but
never flips which rows are `>0` — so **`sum_excluding_real2` is
mathematically guaranteed to equal `sum_excluding_real` on every row**,
which the reproduction verifies (see row 14 of `results_reproduced.csv`:
identical TN/FP/FN/TP/accuracy/etc. to the OR-Logic row).

### 4b. Bonus finding: if someone "fixes" §4a, they will hit direct target leakage in `X2`

Cell 76's drop list for `X2` is copy-pasted from cell 32 and does **not**
drop `'risk factor'` or `'ColumnFailure0'`. So `X2` currently contains
`'ColumnFailure0'` as a **feature column** while `y2` is *also*
`'ColumnFailure0'` — i.e., the target is literally present in the feature
matrix. This is currently inert (because, per §4a, nothing is ever trained
on `X2`/`y2`), but it is a landmine: naively "completing" the missing
`.fit()`/`.predict()` step would produce a trivially perfect (and
meaningless) classifier unless `'ColumnFailure0'` (and arguably `'risk
factor'`, itself derived using the leaky envelope) are dropped from `X2`
first.

### 4c. What this means for the paper's "two-stage alert system" claims

The abstract states: *"a two-stage alert system was implemented: stage one
detects 99.5% of non-failure cases through ensemble prediction, while
stage two applies threshold analysis to operational parameters. Field
deployment showed 88% of failures occurred between alerts, enabling
preemptive shutdowns."* No code in `ML main.ipynb` computes a "field
deployment" statistic, and — per §4a — no code trains or evaluates a
distinct "stage two" model at all; `ColumnFailure0` is computed and then
effectively discarded (overwritten by a copy of the stage-one column
before any downstream metric is reported). Whatever produced the
abstract's stage-two/field-deployment numbers, it is not reproducible from
this notebook.

---

## 5. Evidence that the code's split parameters (not the paper's stated ones) produced the paper's actual numbers

To check whether the paper's Methods description (§3.2/§4.4: stratified
split, `random_state=42`) or the code's actual parameters
(`random_state=25`, no `stratify`) are what really generated the reported
test-set composition, we compared the test-set failure count under all
four combinations:

| `random_state` | `stratify` | Resulting test failures / non-failures |
|---|---|---|
| 42 | none | 72 / 2428 |
| 42 | `y` (stratified) | 85 / 2415 |
| **25** | **none** | **83 / 2417** |
| 25 | `y` (stratified) | 85 / 2415 |

The paper's Table 5 reports **83 failures / 2417 non-failures** in every
row (8+75, 69+14, 63+20, 37+46, 73+10 all sum to 83; 2417 appears as the
TN+FP total for every model). **Only `random_state=25` with no
stratification** — exactly what `ML main.ipynb` cells 16/34 do —
reproduces that split composition. Neither the paper's claimed
`random_state=42` nor genuine stratified sampling produce it. This
confirms the code in the notebook is the right artifact to audit against
the paper (same split composition), while also confirming that §3.2's
"stratified sampling" claim and §4.4's "random_state=42" claim are
inaccurate descriptions of what was actually run.

---

## 6. Why Gradient Boosting is the one exact match, and what that implies

GBM (modified-data) is the only model whose reproduced confusion matrix
(TP=63, FN=20, TN=2414, FP=3 — accuracy 99.08%, recall 75.90%) exactly
matches the paper's Table 5 (TP=63, FN=20, TN=2414, FP=3). It is also the
only model where the paper's stated §4.2 hyperparameters
(`n_estimators=100, learning_rate=0.1, max_depth=3`) happen to equal
`GradientBoostingClassifier()`'s sklearn defaults — i.e., the only model
where paper-Methods and actual-code genuinely agree. Combined with §5's
split-composition match, this is strong evidence that:

1. `src/reproduce.py` is a correct, faithful port of the notebook (same
   split, same row composition, same code path a GBM run would take), and
2. the *other* four models' mismatches (§3a/§3b) are attributable to the
   §2(a) hyperparameter discrepancies between the paper's Methods text and
   the code's actual hyperparameters — not to some unrelated reproduction
   error.

In other words: **the paper's Methods sections (§3.2, §4.2, §4.4) describe
a different pipeline than the one in `ML main.ipynb`**, and the numbers in
Tables 4/5/8/9 are consistent with that different (undocumented) pipeline,
not with the code currently in the repository.

---

## 7. Secondary observation: the notebook's own printed metrics use weighted averaging

Not a paper-vs-code discrepancy, but worth flagging: the notebook's own
`print("Precision:", ...)` / `print("Recall:", ...)` / `print("F1
Score:", ...)` calls (cells 37, 41, 45, 49, 53) all use
`average='weighted'`, e.g. `precision_score(y_test, y_pred,
average='weighted')`. Under 3.39% failure prevalence, a weighted average is
dominated by the negative class and will read far higher than the
positive-class numbers the paper actually reports (e.g. a model with 10%
positive-class recall but near-100% negative-class recall will print a
weighted recall near 90%+). `results_reproduced.csv` reports positive-class
metrics only, per the task's request; if the notebook's own printed
metrics are ever quoted directly, they should not be confused with the
paper's per-class figures.

---

## 8. Summary checklist

- [x] `requirements.txt`, `src/`, `data/`, `outputs/` created; venv used
      (`.venv`, existing repo venv, packages installed via
      `requirements.txt`).
- [x] `data/ai4i2020.csv` obtained from UCI (10,000 rows, matches paper's
      Table 3 stats).
- [x] `src/reproduce.py` runs end-to-end with `.venv/bin/python
      src/reproduce.py`, preserving exact hyperparameters, split, no
      scaling, hard VotingClassifier, OR-logic sum-of-5>0, and the
      stage-two `ColumnFailure0` logic as written (bugs included).
- [x] `outputs/results_reproduced.csv` has full confusion matrix +
      accuracy/precision/recall/F1/specificity/FNR for the positive class,
      for every model (original + modified feature sets) and both
      ensembles, plus the (bugged, identical) stage-two row.
- [x] Discrepancies found:
  - (a) Hyperparameters: SVM, RF, KNN, LR all differ from paper §4.2; only
    GBC matches (§2a).
  - (a) Split: no stratification in code vs. paper's stratified-sampling
    claim (§3.2/§4.4); `random_state=25` in code vs. paper's claimed `42`
    (§4.4) — and §5 shows the code's actual parameters, not the paper's
    stated ones, reproduce the paper's reported split composition.
  - (a) Feature count: code's `X` has 10 columns (`Type_encoded` leaks
    through an incomplete drop list) vs. paper's claimed 9 (§2c).
  - (b) Positive-class metrics: do **not** match the paper's Tables 4/5/9
    for any model except GBM on modified data, which matches exactly
    (§3a/§3b/§6). Table 8 has no corresponding code at all (§3c).
  - (c) `ColumnFailure0` envelope: **confirmed computed over the full
    dataset (train+test), not train-only — genuine leakage** (§4). It is
    currently inert due to a separate stale-variable bug that makes
    "stage two" an exact duplicate of stage one (§4a), and would leak the
    target directly into the feature matrix if that bug were naively
    fixed (§4b). The paper's "two-stage alert system" / "field deployment"
    claims in the abstract have no corresponding computation in the
    notebook at all (§4c).
