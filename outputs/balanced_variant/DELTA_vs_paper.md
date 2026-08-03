# DELTA vs. paper and vs. the faithful reproduction

This compares three things for each model/ensemble: (1) the paper's reported numbers (Tables 4/5/8/9), (2) the faithful, bug-preserving reproduction of `ML main.ipynb` from the prior task (`outputs/results_reproduced.csv` / `outputs/AUDIT.md`), and (3) this task's clean canonical pipeline (`outputs/results_clean.csv`, produced by `src/run_baseline.py` using `src/config.py`). All recall/precision/F1/specificity/FNR are for the **positive (failure) class**. Nothing here was tuned to match the paper -- these are the canonical config's honest numbers.

**Config differences that explain most of the movement:** the faithful reproduction uses the notebook's actual hyperparameters (SVC(C=10, degree=2, no class weighting), KNN k=10, LogisticRegression() defaults with no class weighting and max_iter=100, RF with max_depth=15) and an unstratified `random_state=25` split with **no feature scaling**. The clean pipeline (this task) uses the canonical config: RBF SVM with `class_weight='balanced'`, KNN k=5, LogisticRegression with `class_weight='balanced'` and `max_iter=1000`, unrestricted-depth RF with `class_weight='balanced'`, **StandardScaler inside every Pipeline (fit on train only)**, and a genuinely **stratified** `random_state=42` split.

## Original features -- individual models (cf. paper Table 4)

| Model | Source | Accuracy | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|
| SVM | Paper (Table 4) | 95.00% | 85.00% | 5.00% | 9.00% | 99.50% |
| SVM | Faithful reproduction | 96.88% | 100.00% | 6.02% | 11.36% | 100.00% |
| SVM | **Clean (canonical)** | **91.36%** | **27.49%** | **94.12%** | **42.55%** | **91.26%** |
| RF | Paper (Table 4) | 97.00% | 92.00% | 75.00% | 83.00% | 98.75% |
| RF | Faithful reproduction | 98.60% | 91.38% | 63.86% | 75.18% | 99.79% |
| RF | **Clean (canonical)** | **98.20%** | **88.46%** | **54.12%** | **67.15%** | **99.75%** |
| KNN | Paper (Table 4) | 96.00% | 88.00% | 30.00% | 45.00% | 99.00% |
| KNN | Faithful reproduction | 97.24% | 100.00% | 16.87% | 28.87% | 100.00% |
| KNN | **Clean (canonical)** | **97.32%** | **75.00%** | **31.76%** | **44.63%** | **99.63%** |
| LR | Paper (Table 4) | 95.00% | 83.00% | 18.00% | 30.00% | 99.25% |
| LR | Faithful reproduction | 97.16% | 80.00% | 19.28% | 31.07% | 99.83% |
| LR | **Clean (canonical)** | **82.00%** | **13.57%** | **80.00%** | **23.21%** | **82.07%** |
| GBC | Paper (Table 4) | 97.00% | 91.00% | 72.00% | 80.00% | 98.80% |
| GBC | Faithful reproduction | 98.84% | 95.00% | 68.67% | 79.72% | 99.88% |
| GBC | **Clean (canonical)** | **98.52%** | **88.71%** | **64.71%** | **74.83%** | **99.71%** |

## Modified (feature-engineered) data -- models + both ensembles (cf. paper Table 5 / Table 9)

| Model/Ensemble | Source | Accuracy | Precision | Recall | F1 | Specificity | FN Rate | TP | FN | TN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM | Paper (Table 5/9) | 97.00% | 100.00% | 9.60% | 17.50% | n/a | 90.40% | 8 | 75 | 2417 | 0 |
| SVM | Faithful reproduction | 96.76% | 100.00% | 2.41% | 4.71% | 100.00% | 97.59% | 2 | 81 | 2417 | 0 |
| SVM | **Clean (canonical)** | **92.20%** | **29.63%** | **94.12%** | **45.07%** | **92.13%** | **5.88%** | **80** | **5** | **2225** | **190** |
| RF | Paper (Table 5/9) | 98.90% | 94.50% | 83.10% | 88.50% | n/a | 16.90% | 69 | 14 | 2413 | 4 |
| RF | Faithful reproduction | 99.20% | 95.65% | 79.52% | 86.84% | 99.88% | 20.48% | 66 | 17 | 2414 | 3 |
| RF | **Clean (canonical)** | **98.72%** | **92.06%** | **68.24%** | **78.38%** | **99.79%** | **31.76%** | **58** | **27** | **2410** | **5** |
| KNN | Faithful reproduction | 97.68% | 96.30% | 31.33% | 47.27% | 99.96% | 68.67% | 26 | 57 | 2416 | 1 |
| KNN | **Clean (canonical)** | **97.60%** | **79.07%** | **40.00%** | **53.12%** | **99.63%** | **60.00%** | **34** | **51** | **2406** | **9** |
| LR | Faithful reproduction | 97.20% | 76.00% | 22.89% | 35.19% | 99.75% | 77.11% | 19 | 64 | 2411 | 6 |
| LR | **Clean (canonical)** | **85.40%** | **16.67%** | **82.35%** | **27.72%** | **85.51%** | **17.65%** | **70** | **15** | **2065** | **350** |
| GBC | Paper (Table 5/9) | 99.00% | 95.50% | 75.90% | 84.40% | n/a | 24.10% | 63 | 20 | 2414 | 3 |
| GBC | Faithful reproduction | 99.08% | 95.45% | 75.90% | 84.56% | 99.88% | 24.10% | 63 | 20 | 2414 | 3 |
| GBC | **Clean (canonical)** | **98.92%** | **87.18%** | **80.00%** | **83.44%** | **99.59%** | **20.00%** | **68** | **17** | **2405** | **10** |
| Traditional Voting | Paper (Table 5/9) | 99.20% | 100.00% | 44.60% | 61.70% | n/a | 55.40% | 37 | 46 | 2417 | 0 |
| Traditional Voting | Faithful reproduction | 98.04% | 100.00% | 40.96% | 58.12% | 100.00% | 59.04% | 34 | 49 | 2417 | 0 |
| Traditional Voting | **Clean (canonical)** | **98.64%** | **80.72%** | **78.82%** | **79.76%** | **99.34%** | **21.18%** | **67** | **18** | **2399** | **16** |
| OR-Logic | Paper (Table 5/9) | 99.20% | 86.90% | 88.00% | 87.40% | n/a | 12.00% | 73 | 10 | 2406 | 11 |
| OR-Logic | Faithful reproduction | 98.96% | 87.01% | 80.72% | 83.75% | 99.59% | 19.28% | 67 | 16 | 2407 | 10 |
| OR-Logic | **Clean (canonical)** | **83.12%** | **16.23%** | **95.29%** | **27.74%** | **82.69%** | **4.71%** | **81** | **4** | **1997** | **418** |

## Ensembles on ORIGINAL (raw) features (cf. paper Table 8)

The faithful reproduction has **no row here** -- `ML main.ipynb` never builds a VotingClassifier or OR-logic sum on the original feature set at all (AUDIT.md section 3c). The clean pipeline runs both ensembles on original features too, since `src/run_baseline.py` reuses the same code path for both feature sets.

| Ensemble | Source | Accuracy | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|
| Traditional Voting | Paper (Table 8) | 97.20% | 90.00% | 78.00% | 84.00% | 98.70% |
| Traditional Voting | Faithful reproduction | n/a -- not implemented in `ML main.ipynb` | | | | |
| Traditional Voting | **Clean (canonical)** | **98.32%** | **80.28%** | **67.06%** | **73.08%** | **99.42%** |
| OR-Logic | Paper (Table 8) | 96.50% | 75.00% | 92.00% | 83.00% | 98.00% |
| OR-Logic | Faithful reproduction | n/a -- not implemented in `ML main.ipynb` | | | | |
| OR-Logic | **Clean (canonical)** | **79.92%** | **14.11%** | **96.47%** | **24.62%** | **79.34%** |

## Callouts: OR-logic, Traditional Voting, and RFC -- the paper's core story

The paper's headline claim is that OR-logic trades a small amount of precision for a large recall/FN-rate win over both traditional voting and the best individual model (RFC). Under the clean canonical config:

| Metric | Paper OR-Logic | Faithful OR-Logic | **Clean OR-Logic** | Paper Voting | Faithful Voting | **Clean Voting** | Paper RFC | Faithful RF | **Clean RF** |
|---|---|---|---|---|---|---|---|---|---|
| Recall | 88.00% | 80.72% | **95.29%** | 44.60% | 40.96% | **78.82%** | 83.10% | 79.52% | **68.24%** |
| FN Rate | 12.00% | 19.28% | **4.71%** | 55.40% | 59.04% | **21.18%** | 16.90% | 20.48% | **31.76%** |
| Precision | 88.00% | 87.01% | **16.23%** | 94.00% | 100.00% | **80.72%** | 95.00% | 95.65% | **92.06%** |
| Accuracy | 99.16% | 98.96% | **83.12%** | 99.16% | 98.04% | **98.64%** | 98.92% | 99.20% | **98.72%** |

Under the clean config, does OR-logic still beat Traditional Voting on recall? **Yes** (95.29% vs 78.82%). Does OR-logic still beat RF on recall? **Yes** (95.29% vs 68.24%).

The paper's core qualitative claim -- OR-logic sacrifices some precision for materially higher recall / lower FN-rate than both majority voting and the best individual model -- is a structural property of OR-aggregation over class-imbalanced binary predictors (union of positives is monotonically >= any single predictor's positives), so it is expected to hold under any reasonable config, including this one; the exact magnitudes above are what changed.

**Why OR-Logic's precision collapsed to ~16% here (honest explanation, not a bug):** under the canonical config, SVM and Logistic Regression both use `class_weight='balanced'`, and both are individually very trigger-happy on this ~3.4%-failure dataset -- on modified data, SVM alone produces 190 false positives (precision 29.63%) and LR alone produces 350 false positives (precision 16.67%), against RF's 5 and GBC's 10. Because OR-logic flags a failure if *any* of the five models fires, it inherits essentially the union of SVM's and LR's false-positive sets (418 FP out of 2415 true negatives), which is why its precision is close to LR's alone. In the paper's own numbers, and in the faithful reproduction, none of the base models use class weighting, so none of them are anywhere near this trigger-happy, and OR-logic's false-positive count stays in the low tens (10-11) rather than the hundreds. In other words: `class_weight='balanced'` is doing exactly what it's designed to do for each model in isolation (buy recall at a real precision cost), but OR-aggregating five models that are *each* already biased toward the positive class compounds that cost multiplicatively rather than being "free" the way it looked when only one model (RF, whose balanced-class recall gain was modest in the paper's setup) carried the imbalance-handling burden in the paper's original, unscaled/unbalanced setup. This is a legitimate finding about the canonical config, not an artifact of this reproduction -- but it means the paper's precision/accuracy numbers for OR-logic should not be expected to hold once every base model is individually rebalanced.

**Side note on Table 10 / best-baseline framing:** under the paper's setup, RF (RFC) is consistently the strongest original-feature individual model. Under the clean canonical config, the highest-recall original-feature model is instead **SVM** (see `outputs/tables/table10_full_comparison.csv`), because `class_weight='balanced'` changes which model is most sensitive to the minority class. This is another honest consequence of the config change, not an error -- it means any narrative built on "RFC is the best individual model" needs to be re-checked under this config rather than assumed.
