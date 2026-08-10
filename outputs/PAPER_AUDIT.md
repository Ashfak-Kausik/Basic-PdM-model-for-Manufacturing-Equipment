# PAPER_AUDIT.md

Read-only numeric verification of `main.tex` against `outputs/FINAL_NUMBERS.md` and the
CSVs it cites (plus the raw dataset for a handful of claims that have no committed CSV).
Nothing in the repo was modified during the original audit pass.

**Update:** Items #1–#11 (the `dataset_stats` table cells and all 5 correlation-value
occurrences) have since been corrected in `main.tex`, per the values in this report.
Items #12–#17 (`tab:stability`, `tab:cost`) are still open.

Scope note: literature-citation numbers (other papers' reported accuracies, historical
maintenance-optimization percentages, etc.) are out of scope — they aren't checkable
against this repo's own outputs and aren't flagged below.

---

## 1. MISMATCHES

| # | Location | Manuscript value | Correct value | Source | Status |
|---|---|---|---|---|---|
| 1 | Table `dataset_stats` (line 586), Air temperature min | 294 | **295.3** | `data/ai4i2020.csv` (recomputed: `df["Air temperature [K]"].min()`) | ✅ Fixed |
| 2 | Table `dataset_stats` (line 586), Air temperature max | 304.9 | **304.5** | `data/ai4i2020.csv` | ✅ Fixed |
| 3 | Table `dataset_stats` (line 587), Process temperature max | 315.2 | **313.8** | `data/ai4i2020.csv` | ✅ Fixed |
| 4 | Table `dataset_stats` (line 588), Rotational speed mean | 1539.4 | **1538.8** | `data/ai4i2020.csv` | ✅ Fixed |
| 5 | Table `dataset_stats` (line 588), Rotational speed std | 182.1 | **179.3** | `data/ai4i2020.csv` | ✅ Fixed |
| 6 | Table `dataset_stats` (line 588), Rotational speed min | 1150 | **1168** | `data/ai4i2020.csv` | ✅ Fixed |
| 7 | §3.4 Feature Engineering Approach (line 409), Air/Process temp correlation | 0.86 | **0.88** (0.8761) | `data/ai4i2020.csv` (recomputed Pearson r) | ✅ Fixed |
| 8 | §3.4 (line 409), Rotational speed / Torque correlation | -0.86 | **-0.88** (-0.8750) | `data/ai4i2020.csv` | ✅ Fixed |
| 9 | §4.1 Dataset Description (line 574), same two correlations repeated | 0.86 / -0.86 | **0.88 / -0.88** | `data/ai4i2020.csv` (duplicate of #7/#8) | ✅ Fixed |
| 10 | §5.3.3 Feature Importance Analysis (line 919), Torque–Power correlation | 0.96 | **0.98** (0.9788) | `data/ai4i2020.csv` | ✅ Fixed |
| 11 | §5.3.3 (line 919), Rotational speed–Torque correlation (repeated again) | -0.86 | **-0.88** | `data/ai4i2020.csv` (3rd occurrence of #8) | ✅ Fixed |
| 12 | Table `tab:stability` (line 1003), Traditional Voting specificity CI low | 0.998 | **0.997** | `outputs/experiments/stability/stability_summary.csv` | Open |
| 13 | Table `tab:stability` (line 1003), Traditional Voting MCC (mean [CI]) | 0.63 [0.55, 0.71] | **0.65 [0.59, 0.72]** | `stability_summary.csv` | Open |
| 14 | Table `tab:stability` (line 1004), GBC specificity (mean [CI]) | 0.996 [0.993, 0.998] | **0.997 [0.995, 0.999]** | `stability_summary.csv` | Open |
| 15 | Table `tab:stability` (line 1004), GBC MCC CI high | 0.86 | **0.85** | `stability_summary.csv` | Open |
| 16 | Table `tab:stability` (line 1005), OR-Logic MCC (mean [CI]) | 0.74 [0.68, 0.79] | **0.75 [0.70, 0.80]** | `stability_summary.csv` | Open |
| 17 | Table `tab:cost` (line 1058), OR-Logic ensemble model size | 2,900 KB | **2,980.6 KB** | `outputs/experiments/cost/cost_summary.csv` (sum of 5 members = 2980.6416 KB) | Open |

**Table `tab:stability` is the biggest concentration of errors**: every Recall cell is correct, but every Specificity and MCC cell for GBC, OR-Logic, and Traditional Voting is off (mostly by 0.01, i.e. one part in a hundred) — consistent with the table having been populated from a different/earlier run than the currently-committed `stability_summary.csv`, or hand-transcribed with a rounding slip. Recommend regenerating this table directly from the CSV rather than re-checking cell by cell.

The three correlation-value mismatches (#7, #8, #9, #10, #11 — all the same two underlying numbers, 0.86→0.88 and 0.96→0.98) recur three times each across three different subsections; each occurrence should be fixed together since they're the same claim restated, not independent errors.

---

## 2. UNVERIFIABLE (no source found in the cited ground truth)

| # | Location | Claim | Why unverifiable |
|---|---|---|---|
| 1 | §3.4 (line 429), RelationTemperature "optimal" range | "-12.1 K to -7.6 K" | No committed CSV states this range. Recomputed directly from `data/ai4i2020.csv` as a sanity check (not part of the official ground-truth file list, so still flagged unverifiable against that list) — **and the check raises a real concern**: -12.1 and -7.6 are literally the global min and max of RelationTemperature across the *entire* 10,000-row dataset (failures and non-failures alike). 99.9% of all rows already fall inside this "range," so a claim that failures cluster *outside* it is close to vacuous as worded — there's almost no "outside" for anything to be in. Recommend re-deriving this threshold or rewording the claim. |
| 2 | §3.4 (line 429), WearRPM "optimal" threshold | "values below 0.174" | Same issue: 99.9% of *all* rows (not just non-failures) have WearRPM < 0.174 (verified against raw data). This threshold does not discriminate failures from non-failures in any visible way; the claim as worded is not supported. `src/experiments/explain.py` (line 206) itself flags this exact number as an unverified paper claim awaiting a PDP-plot sanity check, not a confirmed empirical result. |
| 3 | §3.4 (line 429), Power "low-failure-rate" range | "3515.22 W to 8998.49 W" | No committed CSV. Recomputed directly: 99% of all rows fall in this range, and the failure rate *inside* it (≈2.5%) is far lower than the rate *outside* it (≈95%) — so, unlike items 1–2 above, this claim is directionally well-supported by a quick recompute, just not backed by any file in the official ground-truth list. |
| 4 | §3.4 (line 429), ToolWearTorque threshold | "below 0.26253 minimizing machine failure probability" | No committed CSV. Recomputed directly: failure rate below the threshold (≈2.9%) is somewhat lower than above it (≈3.4%) — directionally consistent but a modest effect, not backed by a ground-truth file. |
| 5 | §7 Discussion (line 1099) | Failure cost "\$10,000--\$250,000"; inspection cost "\$100--\$1,000" | No source file in the repo; these appear to be assumed/illustrative figures, not computed from the pipeline. |
| 6 | §7 Discussion (line 1101) | MLP has "higher inference cost" than the tree ensemble | `outputs/experiments/mlp/` contains accuracy/recall/AUC results but no latency or model-size measurement for the MLP, so this specific comparison (cost, not accuracy) has no source to check. |

---

## 3. INTERNAL INCONSISTENCIES (same quantity, different values in different places)

| # | Quantity | Occurrence A | Occurrence B | Note |
|---|---|---|---|---|
| 1 | OR-Logic vs. Voting relative reduction in missed failures | "a **58%** reduction" — §3.6.4 Architecture Diagram (line 552); also Conclusion (line 1174) | "a **58.5%** relative reduction" — §5.5.2 (line 986); §5.5.3 closing paragraph (line 1096) | Same underlying number ((48.24−20.00)/48.24 = 58.51%). 58.5% is the more precise/correct rounding (58.51% rounds to 59% at zero decimals, not 58%, but is consistently written "58.5%" elsewhere) — the two "58%" instances are the outliers and should be updated to match the "58.5%" used in three other places, or all four should be reconciled to one consistent rounding. |

---

## 4. ABSTRACT — flagged, not detailed (per instructions; pending a dedicated abstract pass)

The abstract (lines 229–231) still reports **pre-correction** numbers that no longer match any table in the current Results and Analysis section:

- "The modified ensemble achieved **99.16%** accuracy" — current OR-Logic (modified) accuracy is **98.20%** (`results_clean.csv`).
- "reducing false negatives from **55.4%** to **12%** compared to traditional voting methods" — current values are Traditional Voting FN rate **48.24%** → OR-Logic FN rate **20.00%** (`results_clean.csv`). Both endpoints are stale (55.4%/12% match neither the current single-split nor the 30-seed numbers anywhere else in the paper).

These two numbers appear to be untouched holdovers from before the numeric-correction pass applied to the rest of the paper this session. Flagging only, as requested — not fixing here.

---

## 5. Known-sensitive items — explicitly checked, confirmed correct

- **OR-Logic vs GBC tie / +3.8pp**: consistently and correctly stated everywhere it appears (Results §5.5.2/§5.5.3, Statistical Robustness §5.6, Discussion, Conclusion) — tied 0.800 recall on seed-42, +3.8pp mean over 30 seeds, Wilcoxon p<0.001. Matches `results_clean.csv` and `paired_tests.csv` exactly.
- **GBC as best individual model**: no remaining instance of RFC being called the best individual anywhere in the manuscript (checked every RFC/Random Forest occurrence outside of table rows/captions).
- **Voting recall 0.518 / 0.469, OR-Logic 0.800 / 0.776**: all four values match `results_clean.csv` (single-split) and `stability_summary.csv` (30-seed) exactly.
- **Five ROC AUCs on modified data** (RFC 0.983, GBC 0.976, SVM 0.973, LR 0.904, KNN 0.891): all match `autofeat_results.csv` (`hand_engineered_9feat` rows) exactly.
- **Cost table / sum-of-members property**: Train time, latency, and size for all five individual models match `cost_summary.csv` exactly; the ensemble train time (2.548s) and latency (38µs) correctly equal the sum of the five members. Only the ensemble **size** cell is wrong (2,900 vs correct 2,980.6 — see Mismatch #17).

---

## 6. Other issues noticed (not numeric, flagged per your catch-all instruction)

- **`\label{fig:confusion_modified}` is defined three times**, not twice: once on the real SVM/RF/KNN/LR confusion-matrix figure (line 634) and once more on the GBC confusion-matrix figure (line 648) — both placed oddly inside §4 "Experimental Setup" rather than §5 "Results and Analysis" — and a third time on the correlation-matrix figure `fig30.png` (line 880) inside Results and Analysis. LaTeX will compile with a "multiply defined label" warning and `\ref{fig:confusion_modified}` will resolve to whichever definition compiles last (typically the correlation-matrix one), meaning the sentence at line 767 ("Figure \ref{fig:confusion_modified} presents confusion matrices for models trained on modified data...") will likely link to the wrong figure even though the *real* confusion-matrix figures do exist in the document.
- **"Section [Results]" placeholder** (line 823) is still an unresolved literal placeholder rather than a `\ref`.

Both were previously identified and fixed in `outputs/results_section.tex` during an earlier pass this session, but `main.tex` currently does not reflect that fix (it's a separate copy). Flagged only, per the read-only scope of this task.

---

## 7. Summary counts

- **Tables fully verified as MATCH** (every cell, no exceptions): `baseline_performance`, `detailed_metrics`, `feature_comparison`, `improvement_analysis`, `tab:autofeat`, `tab:ensemble_raw`, `tab:ensemble_modified`, `tab:imbalance`, `tab:final_comparison`, `sample_data` (all 10 rows × 4 engineered-feature columns, spot-checked against recomputation from `data/ai4i2020.csv`), and the `Machine failure`/`Torque`/`Tool wear` rows of `dataset_stats`.
- **Tables with mismatches**: `dataset_stats` (6 cells, Air temp / Process temp / Rotational speed rows only), `tab:stability` (6 cells across 3 rows), `tab:cost` (1 cell).
- **Prose numeric claims checked**: ~60+ individual figures across Abstract, Methodology, Results, Discussion, Limitations, and Conclusion; all MATCH except the correlation values (5 occurrences of 2 underlying numbers), the abstract (flagged separately), and the "58% vs 58.5%" internal inconsistency.
- Total distinct mismatches: **17** (see §1). Total unverifiable claims: **6** (see §2). Internal inconsistencies: **1** (see §3, appearing 4 times in the text).
