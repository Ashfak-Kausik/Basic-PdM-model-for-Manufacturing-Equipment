# FINAL_NUMBERS.md

Single source of truth for every number the manuscript and response-to-reviewers
letter should cite. Every table is read verbatim from an already-committed
`outputs/` CSV produced by the LOCKED pipeline (`src/config.py`, SEED=42,
unbalanced base learners, StandardScaler, stratified 75/25 split) — no
retraining, no config change, nothing recomputed here except where noted.
Source filename is given under each table. Figures referenced below live in
`outputs/figures_final/`.

---

## Split composition

Source: `outputs/split_class_balance.txt`

Stratified train/test split, SEED=42, test_size=0.25, stratify=y (Machine failure).

| | rows | failures | failure rate |
|---|---|---|---|
| Train | 7500 | 254 | 3.387% |
| Test | 2500 | 85 | 3.400% |
| **Total** | **10000** | **339** | **3.390%** |

---

## Table A — Individual models + ensembles, modified data, SEED=42

Source: `outputs/results_clean.csv` (feature_set = "modified")

| Model | Accuracy | Precision | Recall | F1 | Specificity | FN rate | MCC | TP | FN | TN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM | 0.9756 | 0.9000 | 0.3176 | 0.4696 | 0.9988 | 0.6824 | 0.5266 | 27 | 58 | 2412 | 3 |
| RF | 0.9900 | 0.9167 | 0.7765 | 0.8408 | 0.9975 | 0.2235 | 0.8387 | 66 | 19 | 2409 | 6 |
| KNN | 0.9760 | 0.7907 | 0.4000 | 0.5313 | 0.9963 | 0.6000 | 0.5524 | 34 | 51 | 2406 | 9 |
| LR | 0.9688 | 0.6129 | 0.2235 | 0.3276 | 0.9950 | 0.7765 | 0.3579 | 19 | 66 | 2403 | 12 |
| GBC | 0.9892 | 0.8718 | 0.8000 | 0.8344 | 0.9959 | 0.2000 | 0.8296 | 68 | 17 | 2405 | 10 |
| Traditional Voting | 0.9828 | 0.9565 | 0.5176 | 0.6718 | 0.9992 | 0.4824 | 0.6969 | 44 | 41 | 2413 | 2 |
| **OR-Logic** | 0.9820 | 0.7083 | **0.8000** | 0.7514 | 0.9884 | 0.2000 | 0.7436 | 68 | 17 | 2387 | 28 |

Note: on this single SEED=42 split, OR-Logic and GBC land on *identical* recall
(0.800, both TP=68/FN=17) — the recall edge for OR-Logic only appears once you
average over seeds (Table B / Table C).

---

## Table B — Stability across 30 seeds (mean ± 95% CI)

Source: `outputs/experiments/stability/stability_summary.csv`

| Model | Recall | Precision | Specificity | FN rate | MCC |
|---|---|---|---|---|---|
| SVM | 0.339 [0.247, 0.424] | 0.873 [0.796, 0.960] | 0.9982 [0.9971, 0.9996] | 0.661 [0.576, 0.753] | 0.534 [0.453, 0.608] |
| RF | 0.736 [0.632, 0.803] | 0.907 [0.857, 0.956] | 0.9973 [0.9957, 0.9988] | 0.264 [0.197, 0.368] | 0.811 [0.754, 0.854] |
| KNN | 0.371 [0.293, 0.445] | 0.782 [0.689, 0.886] | 0.9963 [0.9945, 0.9985] | 0.629 [0.555, 0.707] | 0.527 [0.463, 0.613] |
| LR | 0.239 [0.158, 0.309] | 0.718 [0.598, 0.854] | 0.9967 [0.9949, 0.9986] | 0.761 [0.691, 0.842] | 0.403 [0.297, 0.490] |
| GBC | 0.739 [0.661, 0.827] | 0.893 [0.831, 0.954] | 0.9968 [0.9945, 0.9988] | 0.261 [0.173, 0.339] | 0.806 [0.754, 0.846] |
| Traditional Voting | 0.469 [0.382, 0.568] | 0.930 [0.872, 0.978] | 0.9987 [0.9974, 0.9996] | 0.531 [0.432, 0.618] | 0.651 [0.589, 0.723] |
| **OR-Logic** | **0.776 [0.703, 0.839]** | 0.745 [0.685, 0.821] | 0.9905 [0.9870, 0.9941] | 0.224 [0.161, 0.297] | 0.751 [0.703, 0.796] |

Figure: `outputs/figures_final/recall_forest.{png,pdf}` (recall panel only, sorted, OR-Logic/Voting highlighted).

---

## Table C — Paired significance (30 seeds)

Source: `outputs/experiments/stability/paired_tests.csv` (Wilcoxon signed-rank, paired per seed) + `outputs/experiments/stability/NOTES.md`

| Comparison | Metric | Mean diff | Wilcoxon p | Effect size |
|---|---|---|---|---|
| OR-Logic vs Traditional Voting | Recall | +0.3078 (+30.8 pp) | 1.72e-06 | 1.0 |
| OR-Logic vs Traditional Voting | FN rate | −0.3078 (−30.8 pp) | 1.72e-06 | −1.0 |
| OR-Logic vs GBC | Recall | +0.0376 (+3.8 pp) | 2.45e-06 | 1.0 |
| OR-Logic vs GBC | FN rate | −0.0376 (−3.8 pp) | 2.53e-06 | −1.0 |

Plain-language reading of each row:
- **OR-Logic vs Voting, recall**: OR-Logic beats Traditional Voting on recall in all 30/30 seeds — a large, unambiguous effect (+30.8 pp mean).
- **OR-Logic vs Voting, FN rate**: mirror image of the above — OR-Logic misses 30.8 pp fewer actual failures than Voting, on average.
- **OR-Logic vs GBC, recall**: OR-Logic beats GBC's recall in 29/30 seeds (1 tie), a real and consistent but *much smaller* effect than the Voting comparison (+3.8 pp, not +30.8 pp) — an honest, marginal practical gain.
- **OR-Logic vs GBC, FN rate**: same result restated as missed failures — OR-Logic misses 3.8 pp fewer failures than GBC on average, consistently but modestly.

This paired per-seed test supersedes the single-split McNemar test
(`outputs/experiments/stability/mcnemar_results.csv`) as the basis for the
paper's recall claim, since it isolates the recall/FNR trade-off directly
across 30 resamples rather than testing overall single-split correctness.

---

## Table D — Original vs modified features (recall/accuracy deltas), base models

Source: `outputs/tables/table6_orig_vs_modified.csv`

| Model | Accuracy (orig) | Recall (orig) | Accuracy (modified) | Recall (modified) | Δ Accuracy (pp) | Δ Recall (pp) |
|---|---|---|---|---|---|---|
| SVM | 0.9720 | 0.2235 | 0.9756 | 0.3176 | +0.36 | +9.41 |
| RF | 0.9840 | 0.6353 | 0.9900 | 0.7765 | +0.60 | +14.12 |
| KNN | 0.9732 | 0.3176 | 0.9760 | 0.4000 | +0.28 | +8.24 |
| LR | 0.9696 | 0.1647 | 0.9688 | 0.2235 | −0.08 | +5.88 |
| GBC | 0.9852 | 0.6471 | 0.9892 | 0.8000 | +0.40 | +15.29 |

Ensemble rows (not in `table6...csv`, which covers only the 5 base models —
computed here as a plain subtraction of the two already-published
`results_clean.csv` rows per ensemble, the same operation `table6` already
applies to the base models; source: `outputs/results_clean.csv`):

| Model | Accuracy (orig) | Recall (orig) | Accuracy (modified) | Recall (modified) | Δ Accuracy (pp) | Δ Recall (pp) |
|---|---|---|---|---|---|---|
| Traditional Voting | 0.9752 | 0.3294 | 0.9828 | 0.5176 | +0.76 | +18.82 |
| OR-Logic | 0.9824 | 0.7176 | 0.9820 | 0.8000 | −0.04 | +8.24 |

---

## Table E — Imbalance strategies, OR-Logic and GBC

Source: `outputs/experiments/imbalance/imbalance_results.csv` + `outputs/experiments/imbalance/NOTES.md`

| Strategy | Model | Recall | Precision | Specificity | FN rate | MCC | TP/FN/TN/FP |
|---|---|---|---|---|---|---|---|
| none (locked) | OR-Logic | 0.800 | 0.708 | 0.988 | 0.200 | 0.744 | 68/17/2387/28 |
| SMOTE | OR-Logic | 0.965 | 0.155 | 0.815 | 0.035 | 0.346 | 82/3/1969/446 |
| class_weight=balanced (parked) | OR-Logic | 0.953 | 0.162 | 0.827 | 0.047 | 0.354 | 81/4/1997/418 |
| RandomUnderSampler | OR-Logic | 0.976 | 0.122 | 0.753 | 0.024 | 0.297 | 83/2/1819/596 |
| none (locked) | GBC | 0.800 | 0.872 | 0.996 | 0.200 | 0.830 | 68/17/2405/10 |
| SMOTE | GBC | 0.906 | 0.348 | 0.940 | 0.094 | 0.540 | 77/8/2271/144 |
| class_weight=balanced (parked)* | GBC | 0.800 | 0.872 | 0.996 | 0.200 | 0.830 | 68/17/2405/10 |
| RandomUnderSampler | GBC | 0.929 | 0.294 | 0.921 | 0.071 | 0.498 | 79/6/2225/190 |

\* GBC's `class_weight=balanced` row is identical to "none" by construction:
`GradientBoostingClassifier` has no `class_weight` parameter, so GBC itself was
never rebalanced (only SVM/RF/LR were, in the parked balanced-variant run).

Plain-language reading: leaving the base models unbalanced (the locked config)
is the right choice for an OR-union ensemble — because OR-Logic's overall
false-positive rate is bounded below by the union of its five members' FPRs,
any per-model rebalancing (SMOTE, class-weighting, undersampling) that trades
precision for recall degrades **every** member simultaneously, and OR-Logic
compounds all five degradations at once. All three rebalancing strategies buy
higher raw recall (0.95–0.98) at a steep specificity/precision cost (e.g.
RandomUnderSampler: specificity 0.988 → 0.753, precision 0.708 → 0.122).

Figure: `outputs/figures_final/imbalance_tradeoff.{png,pdf}`.

---

## Table F — Automatic vs hand-engineered features

Source: `outputs/experiments/autofeat/or_logic_gbc_comparison.csv`, `outputs/experiments/autofeat/autofeat_results.csv`, `outputs/experiments/autofeat/NOTES.md`

| Model | Feature set | Recall | Precision | Specificity | FN rate | AUC |
|---|---|---|---|---|---|---|
| OR-Logic | hand-engineered (9 features) | 0.8000 | 0.7083 | 0.9884 | 0.2000 | 0.8980 |
| OR-Logic | auto-polynomial (20 features) | 0.7765 | 0.7416 | 0.9905 | 0.2235 | 0.8851 |
| GBC | hand-engineered (9 features) | 0.8000 | 0.8718 | 0.9959 | 0.2000 | 0.9763 |
| GBC | auto-polynomial (20 features) | 0.7412 | 0.8630 | 0.9959 | 0.2588 | 0.9658 |

Verdict (from NOTES.md): hand-engineered domain features (RelationTemperature,
Power, WearRPM, ToolWearTorque — 4 added features) beat the automatically
generated `PolynomialFeatures(degree=2)` expansion (20 added features) on
**both** OR-Logic and GBC, on both recall and AUC, despite the automatic set
having 5x more added columns. Hand-engineered features are also far more
interpretable (a threshold on RelationTemperature has physical meaning; a
threshold on "Air temperature × Torque" does not). TSFresh/wavelet baselines
are not applicable — the AI4I 2020 dataset is cross-sectional (no per-machine
timestamp or ID linking rows over time), so there is no time series for those
methods to operate on.

---

## Table G — Computational cost

Source: `outputs/experiments/cost/cost_summary.csv`, `outputs/experiments/cost/system_info.txt`

Measuring CPU: **13th Gen Intel(R) Core(TM) i5-13400F**, 16 logical CPUs, all
models run single-threaded (no `n_jobs>1` anywhere in `src/config.py`), so
these latency numbers are single-thread CPU-comparable across models.

| Model | Fit time (s) | Per-sample predict latency (µs) | Model size (KB) |
|---|---|---|---|
| SVM | 0.440 | 10.95 | 48.2 |
| RF | 0.928 | 4.03 | 1575.6 |
| KNN | 0.006 | 21.56 | 1218.2 |
| LR | 0.015 | 0.47 | 2.0 |
| GBC | 1.158 | 0.70 | 136.7 |
| Traditional Voting | 2.548 (= sum of 5 members) | 38.05 | 2980.6 (= sum of 5 members) |
| OR-Logic | 2.548 (= sum of 5 members) | 38.05 | 2980.6 (= sum of 5 members) |

Voting and OR-Logic share identical cost rows because both ensembles run the
same five base models and differ only in the (near-zero-cost) aggregation
rule applied to their outputs.

---

## SHAP top-5

Source: `outputs/experiments/explain/gbc_shap_ranking.csv`, `outputs/experiments/explain/rf_shap_ranking.csv`

**GBC** (* = engineered feature):
1. Rotational speed [rpm] — 0.3125
2. RelationTemperature* — 0.2645
3. WearRPM* — 0.2639
4. Tool wear [min] — 0.2636
5. Power (W)* — 0.2084

**RF** (* = engineered feature):
1. RelationTemperature* — 0.01625
2. Rotational speed [rpm] — 0.01544
3. WearRPM* — 0.01345
4. Torque [Nm] — 0.01274
5. Power (W)* — 0.01220

Plain-language verdict: engineered features cluster strongly near the top of
both rankings (3 of GBC's top 5, 3 of RF's top 5 are engineered) but do **not**
uniformly dominate rank #1 — RF's top feature is engineered
(RelationTemperature), but GBC's top feature is a raw sensor input (Rotational
speed [rpm]). Honest claim: "engineered features are consistently among the
most important, but the single most important feature depends on the model."

Figure: `outputs/figures_final/shap_gbc.{png,pdf}` (beeswarm + mean|SHAP| bar, engineered features marked with `*`).

---

## Headline sentences for the abstract

- **OR-Logic recall vs Voting, single split (SEED=42)**: 0.800 vs 0.518 (+28.2 pp). *[Table A, `results_clean.csv`]*
- **OR-Logic recall vs Voting, 30-seed CI**: 0.776 [0.703, 0.839] vs 0.469 [0.382, 0.568]; paired mean diff +30.8 pp, Wilcoxon p = 1.7e-6, OR-Logic higher in 30/30 seeds. *[Table B/C, `stability_summary.csv` / `paired_tests.csv`]*
- **OR-Logic vs GBC, paired result — the honest small margin**: on the single SEED=42 split the two are *tied* (both recall = 0.800); across 30 seeds OR-Logic's mean recall (0.776) exceeds GBC's (0.739) by +3.8 pp, Wilcoxon p = 2.5e-6, OR-Logic higher in 29/30 seeds (1 tie) — real and statistically consistent, but a modest practical gain, not a large one. *[Table A/C]*
- **Specificity retained**: OR-Logic keeps 30-seed mean specificity at 0.9905 [0.9870, 0.9941] — a ~0.6–0.8 pp give-up versus GBC (0.9968) and Voting (0.9987), but still comfortably above 99%. *[Table B]*
- **Feature engineering vs auto verdict**: hand-engineered domain features (9 total) beat an automatic 20-feature polynomial expansion on both recall and AUC for OR-Logic and GBC alike, despite having far fewer columns. *[Table F]*

---

## MLP baseline (comparison only, not in ensembles)

Source: `outputs/experiments/mlp/mlp_results.csv`, `mlp_stability.csv`,
`mlp_vs_gbc_paired.csv`, `NOTES.md`.

Answers R1.3 / R1.11 ("why no neural network?") with one empirical row: a
lightly-tuned `MLPClassifier((64, 32), relu, alpha=1e-4, early_stopping=True,
random_state=42)`, fit on the same locked modified nine-feature set, no
class weighting. **Not** added to `src/config.py`'s five base learners and
**not** included in Traditional Voting or OR-Logic — a standalone comparison
baseline only. All numbers above this section are unaffected.

| Model | Accuracy | Precision | Recall | F1 | Specificity | FN rate | MCC | AUC | TP | FN | TN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.9784 | 0.7925 | 0.4941 | 0.6087 | 0.9954 | 0.5059 | 0.6159 | 0.9768 | 42 | 43 | 2404 | 11 |
| GBC | 0.9892 | 0.8718 | 0.8000 | 0.8344 | 0.9959 | 0.2000 | 0.8296 | 0.9763 | 68 | 17 | 2405 | 10 |
| OR-Logic | 0.9820 | 0.7083 | 0.8000 | 0.7514 | 0.9884 | 0.2000 | 0.7436 | 0.8980 | 68 | 17 | 2387 | 28 |

*Source: `mlp_results.csv` (SEED=42 locked split; GBC/OR-Logic rows read verbatim from `results_clean.csv` + `autofeat_results.csv`).*

30-seed stability (mean, 95% CI): MLP recall = 0.481 [0.236, 0.684], vs GBC's
already-committed 0.739 [0.661, 0.827] (`stability_summary.csv`).
*Source: `mlp_stability.csv`.*

Paired Wilcoxon signed-rank, MLP vs GBC recall, across the same 30 seeds:
GBC higher in 30/30 seeds, mean diff = −0.258 (−25.8 pp), Wilcoxon
p = 1.73e-6, effect size = −1.0. *Source: `mlp_vs_gbc_paired.csv`.*

Verdict (from `NOTES.md`): the MLP **clearly underperforms GBC on recall**,
consistently across all 30 resamples, despite near-identical ROC AUC
(0.977 vs 0.976) — a discrepancy flagged as a default-threshold artifact
(the MLP ranks failures about as well as GBC but is more conservative at the
default 0.5 cutoff on this ~3.4%-positive class), not a genuine ranking-
quality gap. The architecture was not tuned, and no result here was used to
adjust the config after the fact.

---

## Needs re-export

None. Every figure and table above was buildable from arrays/CSVs already on
disk in `outputs/`. Two figures (`roc_modified`, `shap_gbc`) needed raw
per-point arrays (ROC fpr/tpr; per-sample SHAP values) that were never
persisted to disk — only the old rendered PNGs and summary CSVs were — so
`scripts/build_final_figures.py` re-derives them by calling the exact same
locked, deterministic functions (`src.experiments.common.locked_seed42_split`,
`src.models.build_models`, SEED=42 throughout) already used by
`src/run_baseline.py` and `src/experiments/explain.py`. Both re-derivations
are checked by assertion against already-committed numbers before the figure
is saved (ROC AUCs vs. `autofeat_results.csv`'s `hand_engineered_9feat` column;
SHAP mean|value| vs. `gbc_shap_ranking.csv`) and matched to floating-point
precision — confirming reproduction, not new analysis.
