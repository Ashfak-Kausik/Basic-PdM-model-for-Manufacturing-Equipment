"""
Reviewers R2.3 / R2.5 / R2.6 -- explainability: SHAP + partial dependence / ICE.

Uses the LOCKED models (src/models.py, src/config.py -- unchanged) fit on the
LOCKED SEED=42 train split, modified nine-feature set.

(a) SHAP: TreeExplainer on the fitted RandomForest and GradientBoosting
    classifiers (fast + exact for tree ensembles). Since each model is a
    Pipeline([('scaler', StandardScaler()), ('clf', ...)]), SHAP values are
    computed on the SCALED inputs the tree actually sees (that's what
    TreeExplainer needs), but displayed against the ORIGINAL (unscaled)
    feature values for interpretability -- scaling is a per-feature
    monotonic transform, so it doesn't change SHAP's importance ranking,
    only the numeric magnitude of the plotted feature axis.
(b) Partial dependence + ICE for GBC on the four engineered features plus
    Torque [Nm], using sklearn's PartialDependenceDisplay directly on the
    fitted Pipeline (so the x-axis is in original, unscaled units).

Outputs:
  outputs/experiments/explain/gbc_shap_ranking.csv
  outputs/experiments/explain/rf_shap_ranking.csv
  outputs/experiments/explain/shap_beeswarm_gbc.png
  outputs/experiments/explain/shap_beeswarm_rf.png
  outputs/experiments/explain/shap_bar_gbc_rf.png
  outputs/experiments/explain/pdp_ice_gbc.png
  outputs/experiments/explain/NOTES.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay

from src import config
from src.experiments.common import EXPERIMENTS_DIR, locked_seed42_split
from src.models import build_models

OUT_DIR = EXPERIMENTS_DIR / "explain"
PDP_FEATURES = ["RelationTemperature", "Power (W)", "WearRPM", "ToolWearTorque", "Torque [Nm]"]


def positive_class_shap_values(explainer, X_scaled) -> np.ndarray:
    """Normalizes across shap/sklearn version differences in TreeExplainer's
    return shape for binary classifiers -- always returns a (n_samples,
    n_features) array of SHAP values for the POSITIVE (failure) class."""
    raw = explainer.shap_values(X_scaled)
    if isinstance(raw, list):  # older shap: list of per-class arrays
        return np.asarray(raw[1])
    raw = np.asarray(raw)
    if raw.ndim == 3:  # newer shap: (n_samples, n_features, n_classes)
        return raw[:, :, 1]
    return raw  # already (n_samples, n_features) -- single-output binary model


def shap_for_model(name, pipe, X_test):
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.TreeExplainer(clf)
    shap_values = positive_class_shap_values(explainer, X_test_scaled)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", ranking.index + 1)
    ranking["is_engineered"] = ranking["feature"].isin(config.ENGINEERED_FEATURES)

    explanation = shap.Explanation(
        values=shap_values,
        data=X_test.to_numpy(),  # original, unscaled values for display
        feature_names=list(X_test.columns),
    )
    return ranking, explanation


def plot_beeswarm(explanation, title, out_path):
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_combined_bar(gbc_ranking, rf_ranking, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    for ax, ranking, name in zip(axes, [gbc_ranking, rf_ranking], ["GBC", "RF"]):
        r = ranking.sort_values("mean_abs_shap")
        colors = ["#c44e52" if e else "#4c72b0" for e in r["is_engineered"]]
        ax.barh(r["feature"], r["mean_abs_shap"], color=colors)
        ax.set_title(f"{name}: mean |SHAP| (positive class)")
        ax.set_xlabel("mean |SHAP value|")
    fig.suptitle("Feature importance ranking (red = engineered feature)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pdp_ice(gbc_pipe, X_train, out_path):
    fig, ax = plt.subplots(figsize=(15, 9))
    PartialDependenceDisplay.from_estimator(
        gbc_pipe,
        X_train,
        features=PDP_FEATURES,
        kind="both",
        subsample=60,
        random_state=config.SEED,
        n_cols=3,
        ax=ax,
        n_jobs=1,
    )
    fig = plt.gcf()
    fig.suptitle("Partial dependence + ICE (GBC) -- engineered features + Torque", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = locked_seed42_split()
    models = build_models()

    gbc_pipe = models["GBC"]
    gbc_pipe.fit(X_train, y_train)
    rf_pipe = models["RF"]
    rf_pipe.fit(X_train, y_train)

    print("Computing SHAP values (TreeExplainer) for GBC and RF...")
    gbc_ranking, gbc_explanation = shap_for_model("GBC", gbc_pipe, X_test)
    rf_ranking, rf_explanation = shap_for_model("RF", rf_pipe, X_test)

    gbc_ranking.to_csv(OUT_DIR / "gbc_shap_ranking.csv", index=False)
    rf_ranking.to_csv(OUT_DIR / "rf_shap_ranking.csv", index=False)
    print(f"Wrote {OUT_DIR / 'gbc_shap_ranking.csv'} and rf_shap_ranking.csv")

    plot_beeswarm(gbc_explanation, "SHAP beeswarm -- GBC (best individual model)", OUT_DIR / "shap_beeswarm_gbc.png")
    plot_beeswarm(rf_explanation, "SHAP beeswarm -- RF", OUT_DIR / "shap_beeswarm_rf.png")
    plot_combined_bar(gbc_ranking, rf_ranking, OUT_DIR / "shap_bar_gbc_rf.png")
    print("Wrote SHAP beeswarm + bar figures")

    print("Computing partial dependence + ICE for GBC...")
    plot_pdp_ice(gbc_pipe, X_train, OUT_DIR / "pdp_ice_gbc.png")
    print(f"Wrote {OUT_DIR / 'pdp_ice_gbc.png'}")

    # --- NOTES.md ---
    gbc_top3 = gbc_ranking.head(3)
    rf_top3 = rf_ranking.head(3)
    gbc_engineered_ranks = gbc_ranking[gbc_ranking["is_engineered"]]["rank"].tolist()
    rf_engineered_ranks = rf_ranking[rf_ranking["is_engineered"]]["rank"].tolist()
    n_features = len(gbc_ranking)
    gbc_top1_engineered = bool(gbc_ranking.iloc[0]["is_engineered"])
    rf_top1_engineered = bool(rf_ranking.iloc[0]["is_engineered"])
    gbc_top5_eng_count = int(gbc_ranking.head(5)["is_engineered"].sum())
    rf_top5_eng_count = int(rf_ranking.head(5)["is_engineered"].sum())

    verdict = (
        f"For GBC, the single top-ranked feature is **{gbc_ranking.iloc[0]['feature']}** "
        f"({'an engineered feature' if gbc_top1_engineered else 'a RAW sensor input, not an engineered one'}), "
        f"but {gbc_top5_eng_count} of the top 5 features are engineered "
        f"(ranks {gbc_engineered_ranks[:3]}...). For RF, the top-ranked feature IS "
        f"engineered (**{rf_ranking.iloc[0]['feature']}**), and {rf_top5_eng_count} of "
        f"its top 5 are engineered. So: the engineered features cluster strongly near "
        f"the top of both rankings and collectively out-rank most raw sensor inputs, "
        f"but they do **not uniformly dominate rank #1** -- for GBC specifically, plain "
        f"Rotational speed [rpm] edges out every engineered feature. This is a more "
        f"qualified claim than \"engineered features are the most important\"; the "
        f"honest statement is \"engineered features are consistently among the most "
        f"important, but the single most important feature depends on the model.\""
    )

    notes = f"""# Explainability notes (R2.3 / R2.5 / R2.6)

## SHAP ranking -- GBC (best individual model)
Top 3 by mean |SHAP|: {', '.join(f"{r.feature} ({r.mean_abs_shap:.4f})" for r in gbc_top3.itertuples())}.
Engineered features occupy ranks {gbc_engineered_ranks} out of {n_features}.

## SHAP ranking -- RF
Top 3 by mean |SHAP|: {', '.join(f"{r.feature} ({r.mean_abs_shap:.4f})" for r in rf_top3.itertuples())}.
Engineered features occupy ranks {rf_engineered_ranks} out of {n_features}.

## Plain-language verdict
{verdict}

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
"""
    (OUT_DIR / "NOTES.md").write_text(notes)
    print(f"Wrote {OUT_DIR / 'NOTES.md'}")
    print("\n" + notes)


if __name__ == "__main__":
    main()
