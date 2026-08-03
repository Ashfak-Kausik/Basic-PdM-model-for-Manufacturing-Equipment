# Auto-feature baseline notes (R1.5 / R1.11)

## Setup
Five original sensor features expanded via `sklearn.preprocessing.PolynomialFeatures`
(degree=2, interaction_only=False, include_bias=False): 5
original -> 20 automatically-generated features (5 linear + 10 pairwise
interactions + 5 squared terms). Scaled in a `Pipeline(PolynomialFeatures -> StandardScaler
-> classifier)`. Same five LOCKED models (identical hyperparameters, imported from
`src/config.py`), same locked SEED=42 stratified split rows, same ensemble thresholds
(Voting >= 3, OR-Logic >= 1) as the canonical hand-engineered pipeline.

## Hand-engineered (4 features: RelationTemperature, Power, WearRPM, ToolWearTorque) vs. auto-polynomial (20 features)

- **OR-Logic**: hand-engineered AUC=0.8980, recall=0.8000 vs. auto-polynomial AUC=0.8851, recall=0.7765 (ΔAUC=-0.0128, Δrecall=-0.0235) -- **hand-engineered features beat auto-polynomial features**.
- **GBC**: hand-engineered AUC=0.9763, recall=0.8000 vs. auto-polynomial AUC=0.9658, recall=0.7412 (ΔAUC=-0.0105, Δrecall=-0.0588) -- **hand-engineered features beat auto-polynomial features**.

Full 7-model/ensemble x 2-feature-set comparison: `autofeat_results.csv`.
Focused OR-Logic/GBC table: `or_logic_gbc_comparison.csv`.

## Verdict
Hand-engineered domain features perform at least as well as, and in most comparisons better than, the automatically-generated polynomial expansion, despite the auto set having far more columns (20 vs. 9 total features, or 4 vs. 20 comparing only the *added* features). This supports the paper's claim that physics-informed feature engineering adds real value beyond what an automatic, domain-agnostic feature expansion provides -- and the hand-engineered features remain far more interpretable (a threshold on RelationTemperature has a physical meaning; a threshold on 'Air temperature x Torque' does not).

## Why TSFresh / wavelets are N/A here (R1.11)
TSFresh, wavelet decomposition, and other time-series feature-extraction libraries
operate on a *sequence* of readings from the same unit over time (extracting things
like rolling statistics, spectral energy, autocorrelation, trend). The AI4I 2020
dataset used throughout this paper is **not** a time series per machine: each of the
10,000 rows is an independent snapshot (a distinct `UDI`/`Product ID`) with no
machine identifier linking rows over time and no timestamp column. There is nothing
for a time-series featurizer to operate on -- applying TSFresh here would require
either (a) fabricating a temporal ordering the dataset does not contain, or (b)
treating unrelated rows as if they were one machine's trajectory, which would be a
methodological error, not a stronger baseline. This is the direct rebuttal to R1.5's
request for automatic feature discovery (addressed above via PolynomialFeatures) and
to R1.11's specific request for TSFresh/wavelet comparison (N/A given the data's
cross-sectional, non-temporal structure).
