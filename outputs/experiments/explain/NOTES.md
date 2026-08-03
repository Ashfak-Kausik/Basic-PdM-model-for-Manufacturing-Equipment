# Explainability notes (R2.3 / R2.5 / R2.6)

## SHAP ranking -- GBC (best individual model)
Top 3 by mean |SHAP|: Rotational speed [rpm] (0.3125), RelationTemperature (0.2645), WearRPM (0.2639).
Engineered features occupy ranks [2, 3, 5, 8] out of 9.

## SHAP ranking -- RF
Top 3 by mean |SHAP|: RelationTemperature (0.0163), Rotational speed [rpm] (0.0154), WearRPM (0.0135).
Engineered features occupy ranks [1, 3, 5, 8] out of 9.

## Plain-language verdict
For GBC, the single top-ranked feature is **Rotational speed [rpm]** (a RAW sensor input, not an engineered one), but 3 of the top 5 features are engineered (ranks [2, 3, 5]...). For RF, the top-ranked feature IS engineered (**RelationTemperature**), and 3 of its top 5 are engineered. So: the engineered features cluster strongly near the top of both rankings and collectively out-rank most raw sensor inputs, but they do **not uniformly dominate rank #1** -- for GBC specifically, plain Rotational speed [rpm] edges out every engineered feature. This is a more qualified claim than "engineered features are the most important"; the honest statement is "engineered features are consistently among the most important, but the single most important feature depends on the model."

See `gbc_shap_ranking.csv` / `rf_shap_ranking.csv` for the exact ordering and
magnitudes (columns: rank, feature, mean_abs_shap, is_engineered).

## Partial dependence / ICE (GBC)
`pdp_ice_gbc.png` shows partial dependence (thick line) + individual conditional
expectation curves (thin lines, 60-row subsample) for the four engineered features
(RelationTemperature, Power (W), WearRPM, ToolWearTorque) and Torque [Nm], on the
locked SEED=42 training data. Look for: (i) whether the PDP curves are monotonic or
threshold-shaped (the paper's Section 3.3 claims specific threshold values, e.g.
WearRPM < 0.174, ToolWearTorque < 0.26253 -- this plot is the direct empirical check
of whether the fitted GBC actually learned anything resembling those thresholds),
and (ii) whether ICE curves fan out (heterogeneous effects / interactions) or stay
tight around the PDP line (a purely additive effect).
