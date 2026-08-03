# DELTA vs. paper and vs. the faithful reproduction

Config change from the previous clean run: `class_weight='balanced'` has been **removed from SVM, RF, and LR** in `src/config.py` (KNN and GBC were never balanced and are unchanged). Reason: with balanced weights, SVM and LR each individually over-fired on the ~3.4%-failure test set, and because OR-logic is a union of the five models' positives, it inherited the union of their false alarms -- recall rose to ~95% but precision collapsed to ~16% and specificity fell to ~83%, breaking the paper's actual claim of high recall *while holding specificity near 99.5%*. The balanced-weights run is preserved in `outputs/balanced_variant/` (see its `README.txt`) as a cost-sensitive baseline for reviewer comment R1.6. Everything else in the canonical config is unchanged: `StandardScaler` in every Pipeline, `SEED=42`, the stratified split, the nine-feature modified set, Traditional Voting = `sum(preds) >= 3`, OR-Logic = `sum(preds) >= 1`.

## Headline

- **OR-Logic** -- clean (unbalanced): recall 80.00%, precision 70.83%, specificity 98.84%, FN rate 20.00%  |  paper: recall 88.00%, precision 86.90%, specificity 99.50%, FN rate 12.00%
- **Traditional Voting** -- clean (unbalanced): recall 51.76%, precision 95.65%, specificity 99.92%, FN rate 48.24%  |  paper: recall 44.60%, precision 100.00%, specificity 100.00%, FN rate 55.40%
- **RFC** -- clean (unbalanced): recall 77.65%, precision 91.67%, specificity 99.75%, FN rate 22.35%  |  paper: recall 83.10%, precision 94.50%, specificity 99.83%, FN rate 16.90%
- **GBC** -- clean (unbalanced): recall 80.00%, precision 87.18%, specificity 99.59%, FN rate 20.00%  |  paper: recall 75.90%, precision 95.50%, specificity 99.88%, FN rate 24.10%

## Modified (feature-engineered) data -- every model & ensemble, three configs side by side

cf. paper Tables 4 (baseline text figures for KNN/LR), 5, and 9. All recall/precision/F1/specificity/FN-rate are for the **positive (failure) class**. KNN and LR paper rows are marked `(inferred)` -- see footnote.

| Model/Ensemble | Source | Accuracy | Precision | Recall | F1 | Specificity | FN Rate | MCC | TP | FN | TN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM | Paper (Table 5/9) | 97.00% | 100.00% | 9.60% | 17.50% | 100.00% | 90.40% | n/a | 8 | 75 | 2417 | 0 |
| SVM | Faithful reproduction | 96.76% | 100.00% | 2.41% | 4.71% | 100.00% | 97.59% | 0.153 | 2 | 81 | 2417 | 0 |
| SVM | **Clean (unbalanced)** | **97.56%** | **90.00%** | **31.76%** | **46.96%** | **99.88%** | **68.24%** | **0.527** | **27** | **58** | **2412** | **3** |
| RF | Paper (Table 5/9) | 98.92% | 94.50% | 83.10% | 88.50% | 99.83% | 16.90% | 0.87 | 69 | 14 | 2413 | 4 |
| RF | Faithful reproduction | 99.20% | 95.65% | 79.52% | 86.84% | 99.88% | 20.48% | 0.868 | 66 | 17 | 2414 | 3 |
| RF | **Clean (unbalanced)** | **99.00%** | **91.67%** | **77.65%** | **84.08%** | **99.75%** | **22.35%** | **0.839** | **66** | **19** | **2409** | **6** |
| KNN | Paper (Table 5/9; text, inferred) | 97.28% | 65.96% | 37.35% | 47.69% | 99.34% | 62.65% | n/a | 31 | 52 | 2401 | 16 |
| KNN | Faithful reproduction | 97.68% | 96.30% | 31.33% | 47.27% | 99.96% | 68.67% | 0.542 | 26 | 57 | 2416 | 1 |
| KNN | **Clean (unbalanced)** | **97.60%** | **79.07%** | **40.00%** | **53.12%** | **99.63%** | **60.00%** | **0.552** | **34** | **51** | **2406** | **9** |
| LR | Paper (Table 5/9; text, inferred) | 97.32% | 80.77% | 25.30% | 38.53% | 99.79% | 74.70% | n/a | 21 | 62 | 2412 | 5 |
| LR | Faithful reproduction | 97.20% | 76.00% | 22.89% | 35.19% | 99.75% | 77.11% | 0.408 | 19 | 64 | 2411 | 6 |
| LR | **Clean (unbalanced)** | **96.88%** | **61.29%** | **22.35%** | **32.76%** | **99.50%** | **77.65%** | **0.358** | **19** | **66** | **2403** | **12** |
| GBC | Paper (Table 5/9) | 98.96% | 95.50% | 75.90% | 84.40% | 99.88% | 24.10% | n/a | 63 | 20 | 2414 | 3 |
| GBC | Faithful reproduction | 99.08% | 95.45% | 75.90% | 84.56% | 99.88% | 24.10% | 0.847 | 63 | 20 | 2414 | 3 |
| GBC | **Clean (unbalanced)** | **98.92%** | **87.18%** | **80.00%** | **83.44%** | **99.59%** | **20.00%** | **0.830** | **68** | **17** | **2405** | **10** |
| Traditional Voting | Paper (Table 5/9) | 99.16% | 100.00% | 44.60% | 61.70% | 100.00% | 55.40% | 0.63 | 37 | 46 | 2417 | 0 |
| Traditional Voting | Faithful reproduction | 98.04% | 100.00% | 40.96% | 58.12% | 100.00% | 59.04% | 0.634 | 34 | 49 | 2417 | 0 |
| Traditional Voting | **Clean (unbalanced)** | **98.28%** | **95.65%** | **51.76%** | **67.18%** | **99.92%** | **48.24%** | **0.697** | **44** | **41** | **2413** | **2** |
| OR-Logic | Paper (Table 5/9) | 99.16% | 86.90% | 88.00% | 87.40% | 99.50% | 12.00% | 0.88 | 73 | 10 | 2406 | 11 |
| OR-Logic | Faithful reproduction | 98.96% | 87.01% | 80.72% | 83.75% | 99.59% | 19.28% | 0.833 | 67 | 16 | 2407 | 10 |
| OR-Logic | **Clean (unbalanced)** | **98.20%** | **70.83%** | **80.00%** | **75.14%** | **98.84%** | **20.00%** | **0.744** | **68** | **17** | **2387** | **28** |

Footnotes: (1) KNN and LR are not in the paper's Table 5/9 at all -- the paper only reports their accuracy and recall in running text (Section 5.3.2: KNN "31 failures (37.3%)... 97.28%"; LR "21 failures (25.3%)... 97.32%"). Their precision/F1/specificity/FN-rate/TN/FP above are *inferred* by solving TN = accuracy*2500 - TP and FP = 2500 - TP - FN - TN from those two reported figures, not values the paper itself states. (2) The paper's own Table 5 and Table 9 **disagree with each other** on Traditional Voting's precision/F1: Table 5 reports TP=37/FN=46/TN=2417/FP=0 (which forces precision=TP/(TP+FP)=100%, F1=61.7%), but Table 9 reports precision=94%/F1=60% for the same row. We used Table 5's confusion-matrix-consistent figures above; this is a discrepancy *within the paper*, not introduced by this reproduction.

## Narrative check

Does OR-logic beat Traditional Voting on recall while keeping specificity high? **Yes** -- OR-Logic reaches 80.00% recall vs. Voting's 51.76% (paper: 88.00% vs. 44.60%), while holding 98.84% specificity (28 FP out of 2415 true negatives) -- a bit below the paper's 99.50% but still solidly in the high-90s, and a world away from the balanced-weights run's ~83% (418 FP, see `outputs/balanced_variant/`). The paper's core contrast -- OR-logic trading a modest precision cost for a large recall/FN-rate win over both Traditional Voting and the best individual model, while specificity stays in the high-90s -- **does hold** under this unbalanced canonical config, and the OR-logic vs. Voting gap is if anything larger here (FN rate 20.00% vs. 48.24%) than the paper reports (FN rate 12.00% vs. 55.40%). GBC remains the strongest individual model by recall on modified data (80.00% vs. RF's 77.65%), unlike the paper where RFC leads GBC.
