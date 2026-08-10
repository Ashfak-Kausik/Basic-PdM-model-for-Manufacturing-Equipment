# CHANGES_RESULTS.md

Every numeric/claim change made to the Results and Analysis section, old -> new,
with its source file. All new numbers are read verbatim from the cited CSV;
none were recomputed or tuned.

## Editorial decision requiring author confirmation

- **"Best Baseline" (raw data) changed from RFC to GBC.** Not explicitly listed
  in the task's table specs, but `results_clean.csv` (feature_set=original)
  shows GBC beats RFC on *every* metric on raw data (accuracy 0.9852 vs 0.9840,
  recall 0.6471 vs 0.6353, precision 0.8871 vs 0.8571, F1 0.7483 vs 0.7297).
  Kept RFC as "best raw baseline" would contradict the source data, so
  `final_comparison`'s "Best Baseline" row and the baseline-paragraph's "best
  individual" language were both updated to GBC. Flag if this is not intended.

## Table baseline_performance (source: results_clean.csv, feature_set=original)

| Model | Metric | Old | New |
|---|---|---|---|
| SVM | Accuracy | 0.9500 | 0.9720 |
| SVM | Precision | 0.85 | 0.8261 |
| SVM | Recall | 0.05 | 0.2235 |
| SVM | F1 | 0.09 | 0.3519 |
| SVM | Specificity | 0.9950 | 0.9983 |
| RFC | Accuracy | 0.9700 | 0.9840 |
| RFC | Precision | 0.92 | 0.8571 |
| RFC | Recall | 0.75 | 0.6353 |
| RFC | F1 | 0.83 | 0.7297 |
| RFC | Specificity | 0.9875 | 0.9963 |
| KNN | Accuracy | 0.9600 | 0.9732 |
| KNN | Precision | 0.88 | 0.7500 |
| KNN | Recall | 0.30 | 0.3176 |
| KNN | F1 | 0.45 | 0.4463 |
| KNN | Specificity | 0.9900 | 0.9963 |
| LR | Accuracy | 0.9500 | 0.9696 |
| LR | Precision | 0.83 | 0.7368 |
| LR | Recall | 0.18 | 0.1647 |
| LR | F1 | 0.30 | 0.2692 |
| LR | Specificity | 0.9925 | 0.9979 |
| GBC | Accuracy | 0.9700 | 0.9852 |
| GBC | Precision | 0.91 | 0.8871 |
| GBC | Recall | 0.72 | 0.6471 |
| GBC | F1 | 0.80 | 0.7483 |
| GBC | Specificity | 0.9880 | 0.9971 |

Prose: "SVM catastrophically low recall (5%)" and "RF/GBC strongest at 75%/72%"
corrected to SVM 22.35%, GBC 64.71% (now best), RF 63.53%. Wrapped in blue
(claim reframed: best-raw-individual changes from RF to GBC).

## Confusion-matrix paragraph, original data (source: results_clean.csv)

| Old claim | New value |
|---|---|
| TN > 2,400/2,417 (>99.4%) | TN > 2,406/2,415 (>99.6%) |
| FP range 1-5 | FP range 4-9 |
| TP range 4 (SVM) - 62 (RFC) | TP range 14 (LR) - 55 (GBC) |
| FN range 21 (RFC) - 79 (SVM) | FN range 30 (GBC) - 71 (LR) |

Total test set is 2,500 rows / 85 failures / 2,415 non-failures (from
`outputs/FINAL_NUMBERS.md` split table), not 2,417 as previously implied.
Wrapped in blue.

## Confusion-matrix paragraph, modified data (source: results_clean.csv, feature_set=modified)

| Old claim | New value |
|---|---|
| TN > 99.8% (TN > 2,412) | TN > 99.5% (TN > 2,403) |
| RF highest recall 83.1% (69 TP, 14 FN) | GBC highest recall 80.00% (68 TP, 17 FN) |
| GBC 75.9% (63 TP, 20 FN) | RF 77.65% (66 TP, 19 FN) |
| KNN 37.3% (31 TP, 52 FN) | KNN 40.00% (34 TP, 51 FN) |
| LR 25.3% (21 TP, 62 FN) | SVM 31.76% (27 TP, 58 FN) |
| SVM 9.6% (8 TP, 75 FN) | LR 22.35% (19 TP, 66 FN) |

Ranking order changes: GBC now #1 (was RF), and SVM now beats LR (was reversed).
Wrapped in blue.

## Ensemble confusion paragraph (Voting vs OR-Logic, modified data)

| Old | New | Source |
|---|---|---|
| Voting: 2,417 TN, 0 FP, 37/83 TP (44.6%), 46 FN | Voting: 2,413 TN, 2 FP, 44/85 TP (51.76%), 41 FN | results_clean.csv |
| OR-Logic: 2,406 TN (99.5%), 11 FP (0.5%), 73/83 TP (88.0%), 10 FN | OR-Logic: 2,387 TN (98.84%), 28 FP (1.16%), 68/85 TP (80.00%), 17 FN | results_clean.csv |
| "11 additional FP -> 36 additional TP, 97% improvement" | "26 additional FP -> 24 additional TP, 54.5% relative improvement" | derived |

Wrapped in blue.

## Table detailed_metrics (source: results_clean.csv, feature_set=modified)

| Metric | Model | Old | New |
|---|---|---|---|
| Accuracy | SVM/RFC/GBC/Voting/OR-Logic | 97.0/98.9/99.0/99.2/99.2% | 97.6/99.0/98.9/98.3/98.2% |
| Precision | same order | 100/94.5/95.5/100/86.9% | 90.0/91.7/87.2/95.7/70.8% |
| Recall | same order | 9.6/83.1/75.9/44.6/88.0% | 31.8/77.6/80.0/51.8/80.0% |
| F1 | same order | 17.5/88.5/84.4/61.7/87.4% | 47.0/84.1/83.4/67.2/75.1% |
| TP | same order | 8/69/63/37/73 | 27/66/68/44/68 |
| FN | same order | 75/14/20/46/10 | 58/19/17/41/17 |
| TN | same order | 2417/2413/2414/2417/2406 | 2412/2409/2405/2413/2387 |
| FP | same order | 0/4/3/0/11 | 3/6/10/2/28 |
| FN Rate | same order | 90.4/16.9/24.1/55.4/12.0% | 68.24/22.35/20.00/48.24/20.00% |
| FP Rate | same order | 0.0/0.17/0.12/0.0/0.46% | 0.12/0.25/0.41/0.08/1.16% |

Note: GBC and OR-Logic now **tie** on recall (80.0%) and FN rate (20.0%) --
previously OR-Logic was shown as uniquely best. Paragraph rewritten (blue) to
state the tie plainly and move the "OR-Logic wins" claim to the cross-seed
result (+3.8pp, Wilcoxon p<0.001, `outputs/experiments/stability/paired_tests.csv`).

## Table feature_comparison (source: outputs/tables/table6_orig_vs_modified.csv)

| Model | Old (Orig Acc/Recall, Mod Acc/Recall) | New |
|---|---|---|
| SVM | 0.95/0.05, 0.97/0.096 | 0.9720/0.2235, 0.9756/0.3176 |
| RFC | 0.97/0.75, 0.9892/0.831 | 0.9840/0.6353, 0.9900/0.7765 |
| KNN | 0.96/0.30, 0.9728/0.373 | 0.9732/0.3176, 0.9760/0.4000 |
| LR | 0.95/0.18, 0.9732/0.253 | 0.9696/0.1647, 0.9688/0.2235 |
| GBC | 0.97/0.72, 0.9896/0.759 | 0.9852/0.6471, 0.9892/0.8000 |

Note: modified-set accuracy for GBC corrected from 0.9896 (not in
results_clean.csv) to 0.9892 (actual value).

## Table improvement_analysis (source: outputs/tables/table6_orig_vs_modified.csv)

Column semantics changed from relative-% improvement to percentage-point (pp)
deltas, per task instruction. Headers renamed "Acc. (%)"/"Recall (%)" ->
"Acc. (pp)"/"Recall (pp)".

| Model | Old Acc Improvement | New Acc Δpp | Old Recall Improvement | New Recall Δpp |
|---|---|---|---|---|
| SVM | +2.1% | +0.36 | +92.0% | +9.41 |
| RFC | +2.0% | +0.60 | +10.8% | +14.12 |
| KNN | +1.3% | +0.28 | +24.3% | +8.24 |
| LR | +2.4% | **-0.08** | +40.6% | +5.88 |
| GBC | +2.0% | +0.40 | +5.4% | +15.29 |

LR's accuracy delta flips sign (was shown as a gain, is actually a -0.08pp
decrease). Paragraph corrected (blue): removed false "all models gain
accuracy" and false "GBC has the smallest recall gain" claims -- GBC now has
the *largest* recall gain (+15.29pp). Also corrected false "precision
improved or preserved for all models" claim: LR (73.68%->61.29%) and GBC
(88.71%->87.18%) both show a precision decrease (results_clean.csv).

## Table ensemble_raw (source: results_clean.csv, feature_set=original, ensemble rows)

| Metric | Old Voting/OR-Logic | New Voting/OR-Logic |
|---|---|---|
| Accuracy | 0.9720/0.9650 | 0.9752/0.9824 |
| Precision | 0.90/0.75 | 0.8485/0.7531 |
| Recall | 0.78/0.92 | 0.3294/0.7176 |
| F1 | 0.84/0.83 | 0.4746/0.7349 |
| Specificity | 0.9870/0.9800 | 0.9979/0.9917 |

Notable reversal: OR-Logic's accuracy is now *higher* than Voting's (98.24%
vs 97.52%), not lower as the old text claimed ("cost of ... accuracy").
Paragraph rewritten (blue).

## Table ensemble_modified (source: results_clean.csv, feature_set=modified)

| Metric | Old Voting/OR-Logic/BestIndiv(RFC) | New Voting/OR-Logic/BestIndiv(GBC) |
|---|---|---|
| Accuracy | 0.9916/0.9916/0.9892 | 0.9828/0.9820/0.9892 |
| Precision | 0.94/0.88/0.95 | 0.9565/0.7083/0.8718 |
| Recall | 0.446/0.880/0.831 | 0.5176/0.8000/0.8000 |
| F1 | 0.60/0.88/0.89 | 0.6718/0.7514/0.8344 |
| Specificity | 1.0000/0.9950/0.9983 | 0.9992/0.9884/0.9959 |
| MCC | 0.63/0.88/0.87 | 0.6969/0.7436/0.8296 |
| FN Rate | 0.554/0.120/0.169 | 0.4824/0.2000/0.2000 |

Best Individual model relabeled RFC -> GBC. Both paragraphs after this table
fully rewritten (blue) to state the SEED=42 tie between OR-Logic and GBC
honestly, and cite the cross-seed +3.8pp result
(`outputs/experiments/stability/paired_tests.csv`) as the real basis for any
"OR-Logic beats best individual" claim.

## ROC/AUC paragraph (source: outputs/experiments/autofeat/autofeat_results.csv, feature_set=hand_engineered_9feat)

| Old | New |
|---|---|
| GBC & RFC AUC = 0.99 | RFC = 0.983, GBC = 0.976 |
| SVM & LR AUC = 0.89 | SVM = 0.973, LR = 0.904 |
| KNN AUC = 0.85 (weakest) | KNN = 0.891 (weakest) |

SVM's AUC is far higher than previously stated (0.973 vs 0.89) despite its
low recall -- flagged in the rewritten paragraph (blue) as a
threshold/ranking distinction.

## Table final_comparison (source: results_clean.csv + paired_tests.csv, all deltas derived)

| Row | Metric | Old | New |
|---|---|---|---|
| Best Baseline | label | RFC, raw | **GBC, raw** (see decision note above) |
| Best Baseline | Accuracy/Recall/FN Rate | 97.0%/75.0%/25.0% | 98.52%/64.71%/35.29% |
| Best Individual | label | RFC, modified | **GBC, modified** |
| Best Individual | Accuracy/Recall/FN Rate/Improvement | 98.9%/83.1%/16.9%/+8.1% recall | 98.92%/80.00%/20.00%/+15.3pp recall |
| Traditional Voting | Accuracy/Recall/FN Rate/Improvement | 99.2%/44.6%/55.4%/-38.5% recall | 98.28%/51.76%/48.24%/-28.2pp recall |
| Proposed Ensemble | Accuracy/Recall/FN Rate/Improvement | 99.2%/88.0%/12.0%/+13.0% recall | 98.20%/80.00%/20.00%/+28.2pp recall |
| vs. Baseline | Acc/Recall/FNRate | +2.2%/+13.0%/-52% | -0.32pp/+15.3pp/-15.3pp |
| vs. Best Individual | Acc/Recall/FNRate | +0.3%/+4.9%/-29% | -0.72pp/**+0.0pp**\*/+0.0pp\* |
| vs. Traditional Voting | Acc/Recall/FNRate | 0.0%/+43.4%/-78% | -0.08pp/+28.2pp/-28.2pp |

\* Footnote added (blue): tied on this split; +3.8pp mean recall gain over
GBC across 30 seeds, paired Wilcoxon p<0.001 (paired_tests.csv).

Note the accuracy sign flips: the proposed ensemble is *not* higher-accuracy
than the raw baseline (-0.32pp) or the best individual (-0.72pp) -- both were
previously shown as positive. Final paragraph rewritten (blue) to state this
plainly alongside the honest recall framing.

## Flagged but NOT fixed (per instructions)

1. `Detailed performance analysis and impact assessment are presented in
   Section [Results].` -- broken placeholder reference, left as-is.
2. `\label{fig:confusion_modified}` is used twice: once (correctly) for the
   modified-data confusion-matrix figure reference, and again for the
   correlation-matrix figure (`fig30.png`) later in the section. Left as-is;
   will produce a duplicate-label warning/wrong-figure-number in LaTeX.
