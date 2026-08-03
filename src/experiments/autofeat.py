"""
Reviewer R1.5 -- automatic feature baseline (honest tabular comparison, NOT
time-series featurization).

Builds an alternative feature set from the FIVE original features only,
expanded automatically via sklearn PolynomialFeatures(degree=2,
interaction_only=False, include_bias=False) -- 5 linear + 10 pairwise
interaction + 5 squared = 20 auto-generated features, scaled in a Pipeline
(Poly -> StandardScaler -> classifier). Runs the same five LOCKED models
(same hyperparameters, from src/config.py, imported directly -- not
redefined) plus both locked ensembles on this auto-feature set, under the
SAME SEED=42 stratified split rows as the locked hand-engineered pipeline.

This directly answers "why didn't you just let an algorithm generate
features instead of hand-engineering four of them" (R1.5), and documents
why TSFresh/wavelet-style time-series featurization is not applicable to
this dataset at all (R1.11): AI4I 2020 is 10,000 independent per-unit
snapshots, not per-machine time series -- there is no within-machine
temporal ordering to extract time-domain/frequency-domain features from.

Outputs:
  outputs/experiments/autofeat/autofeat_results.csv          (7 rows: hand-engineered vs 7 rows: auto-poly, both feature sets)
  outputs/experiments/autofeat/or_logic_gbc_comparison.csv   (focused OR-Logic/GBC hand vs auto table)
  outputs/experiments/autofeat/NOTES.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from src import config
from src.data import load_raw, split_indices, train_test_frames
from src.experiments.common import BASE_MODEL_NAMES, EXPERIMENTS_DIR, locked_seed42_split, ensembles_from_preds
from src.evaluate import positive_class_metrics

OUT_DIR = EXPERIMENTS_DIR / "autofeat"


def build_poly_models() -> dict:
    """Same locked hyperparameters as src/models.py's build_models(), just
    with PolynomialFeatures ahead of the scaler instead of the hand-
    engineered feature columns. Hyperparameters are imported from
    src/config.py unchanged -- nothing here is tuned."""

    def poly_pipeline(clf):
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    return {
        "SVM": poly_pipeline(SVC(**config.SVM_PARAMS)),
        "RF": poly_pipeline(RandomForestClassifier(**config.RF_PARAMS)),
        "KNN": poly_pipeline(KNeighborsClassifier(**config.KNN_PARAMS)),
        "LR": poly_pipeline(LogisticRegression(**config.LR_PARAMS)),
        "GBC": poly_pipeline(GradientBoostingClassifier(**config.GBC_PARAMS)),
    }


def run_pipeline(models: dict, X_train, y_train, X_test, y_test, feature_set_label):
    rows = []
    preds, probas = {}, {}
    for name in BASE_MODEL_NAMES:
        pipe = models[name]
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds[name] = pred
        probas[name] = proba
        row = positive_class_metrics(y_test, pred, name, feature_set_label)
        row["auc"] = roc_auc_score(y_test, proba)
        rows.append(row)

    voting_pred, or_pred = ensembles_from_preds(preds, X_test.index)
    # ensemble "score" for AUC purposes: fraction of the 5 members voting failure
    pred_df = pd.DataFrame(preds, index=X_test.index)[BASE_MODEL_NAMES]
    ensemble_score = pred_df.sum(axis=1) / len(BASE_MODEL_NAMES)

    row = positive_class_metrics(y_test, voting_pred, "Traditional Voting", feature_set_label)
    row["auc"] = roc_auc_score(y_test, ensemble_score)
    rows.append(row)

    row = positive_class_metrics(y_test, or_pred, "OR-Logic", feature_set_label)
    row["auc"] = roc_auc_score(y_test, ensemble_score)
    rows.append(row)

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- hand-engineered (locked) feature set: reuse the locked pipeline verbatim ---
    X_train_hand, X_test_hand, y_train, y_test = locked_seed42_split()
    from src.models import build_models

    hand_rows = run_pipeline(build_models(), X_train_hand, y_train, X_test_hand, y_test, "hand_engineered_9feat")
    print(f"Hand-engineered (9-feature) pipeline: {len(hand_rows)} rows")

    # --- auto-polynomial feature set: SAME split rows, but built from the 5
    # original features expanded via PolynomialFeatures(degree=2) ---
    df = load_raw()
    train_idx, test_idx = split_indices(df)  # identical row split as locked_seed42_split()
    X_original_full = df[config.ORIGINAL_FEATURES]
    y_full = df[config.TARGET]
    X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_frames(X_original_full, y_full, train_idx, test_idx)
    assert y_train_auto.equals(y_train) and y_test_auto.equals(y_test), "split mismatch between hand and auto feature runs"

    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    n_auto_features = poly.fit(X_train_auto).n_output_features_
    print(f"Auto-polynomial feature set: {len(config.ORIGINAL_FEATURES)} original -> {n_auto_features} polynomial features")

    auto_rows = run_pipeline(build_poly_models(), X_train_auto, y_train_auto, X_test_auto, y_test_auto, "auto_polynomial_20feat")
    print(f"Auto-polynomial pipeline: {len(auto_rows)} rows")

    all_rows = hand_rows + auto_rows
    df_out = pd.DataFrame(all_rows)
    col_order = [
        "feature_set", "model_or_ensemble", "n_test", "TP", "FN", "TN", "FP",
        "accuracy", "precision", "recall", "f1", "specificity", "fnr", "fpr", "mcc", "auc",
    ]
    df_out = df_out[col_order]
    results_path = OUT_DIR / "autofeat_results.csv"
    df_out.to_csv(results_path, index=False)
    print(f"Wrote {results_path}")

    # --- focused comparison table: OR-Logic and GBC, hand vs auto ---
    focus_rows = []
    for name in ["OR-Logic", "GBC"]:
        hand = df_out[(df_out["feature_set"] == "hand_engineered_9feat") & (df_out["model_or_ensemble"] == name)].iloc[0]
        auto = df_out[(df_out["feature_set"] == "auto_polynomial_20feat") & (df_out["model_or_ensemble"] == name)].iloc[0]
        for label, r in [("hand_engineered (9 features)", hand), ("auto_polynomial (20 features)", auto)]:
            focus_rows.append(
                {
                    "model_or_ensemble": name,
                    "feature_set": label,
                    "recall": r["recall"],
                    "precision": r["precision"],
                    "specificity": r["specificity"],
                    "fnr": r["fnr"],
                    "auc": r["auc"],
                }
            )
    focus_df = pd.DataFrame(focus_rows)
    focus_path = OUT_DIR / "or_logic_gbc_comparison.csv"
    focus_df.to_csv(focus_path, index=False)
    print(f"Wrote {focus_path}")
    print(focus_df.to_string(index=False))

    # --- NOTES.md ---
    def get(name, fs):
        return df_out[(df_out["feature_set"] == fs) & (df_out["model_or_ensemble"] == name)].iloc[0]

    or_hand, or_auto = get("OR-Logic", "hand_engineered_9feat"), get("OR-Logic", "auto_polynomial_20feat")
    gbc_hand, gbc_auto = get("GBC", "hand_engineered_9feat"), get("GBC", "auto_polynomial_20feat")

    def verdict_line(name, hand, auto):
        auc_diff = auto["auc"] - hand["auc"]
        recall_diff = auto["recall"] - hand["recall"]
        winner = "auto-polynomial features BEAT hand-engineered features" if auc_diff > 0.005 else (
            "hand-engineered features beat auto-polynomial features" if auc_diff < -0.005 else
            "hand-engineered and auto-polynomial features are roughly tied"
        )
        return (
            f"- **{name}**: hand-engineered AUC={hand['auc']:.4f}, recall={hand['recall']:.4f} vs. "
            f"auto-polynomial AUC={auto['auc']:.4f}, recall={auto['recall']:.4f} "
            f"(ΔAUC={auc_diff:+.4f}, Δrecall={recall_diff:+.4f}) -- **{winner}**."
        )

    or_line = verdict_line("OR-Logic", or_hand, or_auto)
    gbc_line = verdict_line("GBC", gbc_hand, gbc_auto)

    any_auto_wins = (or_auto["auc"] - or_hand["auc"] > 0.005) or (gbc_auto["auc"] - gbc_hand["auc"] > 0.005)

    notes = f"""# Auto-feature baseline notes (R1.5 / R1.11)

## Setup
Five original sensor features expanded via `sklearn.preprocessing.PolynomialFeatures`
(degree=2, interaction_only=False, include_bias=False): {len(config.ORIGINAL_FEATURES)}
original -> {n_auto_features} automatically-generated features (5 linear + 10 pairwise
interactions + 5 squared terms). Scaled in a `Pipeline(PolynomialFeatures -> StandardScaler
-> classifier)`. Same five LOCKED models (identical hyperparameters, imported from
`src/config.py`), same locked SEED=42 stratified split rows, same ensemble thresholds
(Voting >= 3, OR-Logic >= 1) as the canonical hand-engineered pipeline.

## Hand-engineered (4 features: RelationTemperature, Power, WearRPM, ToolWearTorque) vs. auto-polynomial (20 features)

{or_line}
{gbc_line}

Full 7-model/ensemble x 2-feature-set comparison: `autofeat_results.csv`.
Focused OR-Logic/GBC table: `or_logic_gbc_comparison.csv`.

## Verdict
{"**Counter to expectation: the auto-generated polynomial features are at least competitive with, or beat, the hand-engineered physics-informed features** on this metric for at least one of OR-Logic/GBC -- see the deltas above. This does not mean domain knowledge was wasted (the engineered features are far more *interpretable* -- a threshold on RelationTemperature has a physical meaning; a threshold on 'Air temperature x Torque' does not), but it means the raw predictive-power claim ('domain features are necessary for strong performance') is not fully supported by this baseline and should be softened in the paper." if any_auto_wins else "Hand-engineered domain features perform at least as well as, and in most comparisons better than, the automatically-generated polynomial expansion, despite the auto set having far more columns (20 vs. 9 total features, or 4 vs. 20 comparing only the *added* features). This supports the paper's claim that physics-informed feature engineering adds real value beyond what an automatic, domain-agnostic feature expansion provides -- and the hand-engineered features remain far more interpretable (a threshold on RelationTemperature has a physical meaning; a threshold on 'Air temperature x Torque' does not)."}

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
"""
    (OUT_DIR / "NOTES.md").write_text(notes)
    print(f"Wrote {OUT_DIR / 'NOTES.md'}")
    print("\n" + notes)


if __name__ == "__main__":
    main()
