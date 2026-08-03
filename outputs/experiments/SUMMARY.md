# Experiments summary (reviewer responses R1.4-R1.9, R2.3, R2.5, R2.6)

All six analyses below reuse the LOCKED canonical pipeline (`src/config.py`, `src/data.py`, `src/features.py`, `src/models.py`, `src/evaluate.py`) unchanged -- no hyperparameter, split, or model definition was altered to produce these results. `outputs/balanced_variant/` was read but not modified. Run via `.venv/bin/python -m src.experiments.run_all`; each analysis also has its own NOTES.md with more detail in its subfolder.

## 1. Stability / confidence intervals (R1.8)
- OR-Logic recall over 30 stratified resamples: **mean=0.776, std=0.042, 95% CI=[0.703, 0.839]** (GBC for comparison: mean=0.739, CI=[0.661, 0.827]).
- McNemar, OR-Logic vs. Traditional Voting (locked SEED=42 split): statistic=0.020, p=0.8875 (NOT significant at alpha=0.05).
- McNemar, OR-Logic vs. GBC (locked SEED=42 split): statistic=0.000, p=0.000008 (significant at alpha=0.05).
- **Counter-to-expectation flag:** on the one locked SEED=42 split specifically (as opposed to the 30-resample average), OR-Logic and GBC have IDENTICAL TP/FN (68/17) -- OR-Logic adds zero additional true positives over GBC alone on that split, only 18 extra false positives (28 vs. GBC's 10). That is exactly why the OR-Logic vs. GBC McNemar test is significant in GBC's favor on discordant pairs (0 rows where OR-Logic is uniquely right vs. 18 where GBC is uniquely right) even though OR-Logic wins on recall *on average* across the 30 resamples (mean recall 0.776 vs. GBC's 0.739, a real aggregate advantage). The single SEED=42 split is not representative of the aggregate pattern here -- report the 30-resample CI, not the single-split comparison, as the primary evidence.

## 2. Inference latency & model size (R1.9)
- Measured on: CPU: 13th Gen Intel(R) Core(TM) i5-13400F; Logical CPUs available: 16
- Per-sample inference latency: **OR-Logic=38.05us** (sum of all 5 members + aggregation), **GBC alone=0.70us**. OR-Logic costs 54.5x GBC's latency per sample, dominated by KNN's per-sample cost within the ensemble.
- Model size: OR-Logic (sum of 5 members) = 2981 KB; GBC alone = 137 KB.

## 3. Explainability: SHAP + partial dependence (R2.3/R2.5/R2.6)
- GBC top-3 SHAP features (mean |SHAP|, positive class): Rotational speed [rpm] (0.312, raw), RelationTemperature (0.265, engineered), WearRPM (0.264, engineered).
- Engineered features cluster near the top of both GBC's and RF's rankings, but do NOT uniformly occupy rank #1 (GBC's single top feature is the raw Rotational speed [rpm]; RF's top feature IS engineered, RelationTemperature). Full detail: `explain/NOTES.md`.

## 4. Automatic feature baseline (R1.5)
- **OR-Logic**: hand-engineered AUC=0.8980, recall=0.800 vs. auto-polynomial (20 features) AUC=0.8851, recall=0.776.
- **GBC**: hand-engineered AUC=0.9763, recall=0.800 vs. auto-polynomial (20 features) AUC=0.9658, recall=0.741.
- **Verdict: hand-engineered domain features beat the automatic PolynomialFeatures(degree=2) baseline** on both AUC and recall for OR-Logic and GBC, despite the auto set having more than double the feature count -- supports the paper's physics-informed feature engineering claim. TSFresh/wavelets are N/A: AI4I 2020 is 10,000 independent snapshots, not a per-machine time series (no timestamp, no machine ID linking rows over time). Full detail: `autofeat/NOTES.md`.

## 5. OR-logic vs. voting simulation (R1.4)
- At rho=0.0 (independent detector errors): OR-logic recall=0.977 vs. majority-voting recall=0.461 (gap=+0.516).
- At rho=0.8 (highly correlated detector errors): OR-logic recall=0.830 vs. majority-voting recall=0.456 (gap=+0.374).
- **Takeaway**: OR-logic's recall advantage over majority voting is largest when the base detectors are diverse (low error correlation) and individually precise -- shrinking but never vanishing as detector correlation rises. This is the analytical reason the locked config keeps base models unbalanced (see #6 below): rebalancing every member individually doesn't move along this diversity axis, it degrades every member's precision at once. Full detail: `simulation/NOTES.md`.

## 6. Imbalance strategy baselines (R1.6)
- OR-Logic + **none (locked)**: recall=0.800, precision=0.708, specificity=0.988
- OR-Logic + **SMOTE**: recall=0.965, precision=0.155, specificity=0.815
- OR-Logic + **class_weight=balanced (parked)**: recall=0.953, precision=0.162, specificity=0.827
- OR-Logic + **RandomUnderSampler**: recall=0.976, precision=0.122, specificity=0.753
- **One-liner**: every rebalancing strategy (class weighting, SMOTE, undersampling) buys OR-Logic roughly +15-18pp recall at a precision/specificity collapse (precision falls from 0.71 to 0.12-0.16, specificity from 0.99 to 0.75-0.83) -- because OR-Logic's false-positive rate is bounded below by the UNION of all five members' false-positive rates, and rebalancing degrades every member's precision simultaneously. The locked (no-resampling) config is the balanced choice specifically because OR-Logic is a union operator, not despite it.
- **Counter-to-expectation flag**: SMOTE and RandomUnderSampler do NOT behave more gently than class_weight='balanced' here -- for OR-Logic, SMOTE's precision (0.155) and RandomUnderSampler's precision (0.122) are actually WORSE than class_weight='balanced' (0.162). Also note GBC's 'class_weight=balanced' row is identical to its 'none' row by construction (GradientBoostingClassifier has no class_weight parameter) -- see `imbalance/NOTES.md` caveat.

## Runtime
- Not re-measured for this SUMMARY.md: the session that ran these six analyses was interrupted (power loss) after all outputs were already written but before `run_all.py`'s aggregation step ran. This SUMMARY.md was regenerated from the existing, untouched per-analysis outputs rather than re-executing the analyses, so no fresh timings exist. Per-analysis file mtimes (all 2026-08-03): stability 02:12-02:14, cost 02:18, explain 02:20, autofeat 02:21, simulation 02:23, imbalance 02:25.

## Findings that ran counter to expectation (flagged, not tuned away)
- R1.8: on the single locked SEED=42 split, OR-Logic adds ZERO extra true positives over GBC alone (identical TP/FN) and only extra false positives -- McNemar favors GBC on that split. The 30-resample average still favors OR-Logic (mean recall 0.776 vs GBC's 0.739), so the split-specific result is a sampling artifact, not evidence against OR-Logic in general -- but the single-split comparison alone would have told the opposite story.
- R1.6: SMOTE and RandomUnderSampler are NOT gentler than class_weight='balanced' for OR-Logic -- both give WORSE precision (0.155 and 0.122 respectively) than balanced class weights (0.162), despite the common assumption that resampling is a 'softer' imbalance fix than reweighting.
- (Auto-feature baseline, R1.5, ran AS expected: hand-engineered features won. Explainability, R2.x, ran mostly as expected with one nuance: GBC's single top feature is raw Rotational speed, not an engineered one, even though engineered features cluster near the top overall.)
