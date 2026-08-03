"""
Reviewer R1.8 -- stability / confidence intervals.

Repeats the full modified-data pipeline over N=30 stratified resamples,
varying ONLY the train/test split's random_state (test_size and stratify
stay locked at config.TEST_SIZE / y). Reuses the locked model factory
(src/models.py) unchanged -- every seed gets freshly-fit copies of the same
five Pipelines with the same hyperparameters.

Also runs a paired McNemar test (statsmodels) on the SEED=42 locked split:
OR-Logic vs Traditional Voting, and OR-Logic vs GBC (the best individual
model), to check whether OR-Logic's edge is statistically significant on
this dataset (not just apparent).

Outputs:
  outputs/experiments/stability/stability_raw.csv       (30 seeds x 7 models/ensembles, per-metric)
  outputs/experiments/stability/stability_summary.csv    (one row per model/ensemble x metric: mean/std/ci_low/ci_high)
  outputs/experiments/stability/mcnemar_results.csv      (the two paired tests)
  outputs/experiments/stability/recall_forest.png        (forest plot of recall with 95% CI, one row per model/ensemble)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.experiments.common import (
    BASE_MODEL_NAMES,
    EXPERIMENTS_DIR,
    load_modified_xy,
    resample_split,
    all_rows_for_split,
    locked_seed42_split,
)

OUT_DIR = EXPERIMENTS_DIR / "stability"
METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "fnr", "mcc"]
MODEL_ORDER = BASE_MODEL_NAMES + ["Traditional Voting", "OR-Logic"]

# Fixed list of 30 seeds for the resampling loop (distinct from the locked
# config.SEED=42, which is reserved for the canonical single-split run and
# for the McNemar tests below).
SEEDS = list(range(1, 31))


def run_resamples():
    X, y, _ = load_modified_xy()
    all_rows = []
    for i, seed in enumerate(SEEDS):
        X_train, X_test, y_train, y_test = resample_split(X, y, seed)
        rows, _ = all_rows_for_split(X_train, y_train, X_test, y_test)
        for r in rows:
            r["seed"] = seed
        all_rows.extend(rows)
        print(f"  seed {seed} ({i + 1}/{len(SEEDS)}) done")
    return pd.DataFrame(all_rows)


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for name in MODEL_ORDER:
        sub = raw_df[raw_df["model_or_ensemble"] == name]
        for metric in METRICS:
            vals = sub[metric].to_numpy(dtype=float)
            mean = vals.mean()
            std = vals.std(ddof=1)
            ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
            out_rows.append(
                {
                    "model_or_ensemble": name,
                    "metric": metric,
                    "n_seeds": len(vals),
                    "mean": mean,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(out_rows)


def paired_mcnemar(y_true, pred_a, pred_b, label_a, label_b):
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(pred_a) == y_true)
    correct_b = (np.asarray(pred_b) == y_true)
    table = np.array(
        [
            [int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],
            [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))],
        ]
    )
    # Exact binomial test when the discordant-pair count is small (statsmodels
    # convention), else chi-square with continuity correction.
    n_discordant = table[0, 1] + table[1, 0]
    use_exact = n_discordant < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "both_correct": table[0, 0],
        f"{label_a}_correct_only": table[0, 1],
        f"{label_b}_correct_only": table[1, 0],
        "both_wrong": table[1, 1],
        "n_discordant": int(n_discordant),
        "test_type": "exact_binomial" if use_exact else "chi2_corrected",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def run_mcnemar():
    X_train, X_test, y_train, y_test = locked_seed42_split()
    rows, artifacts = all_rows_for_split(X_train, y_train, X_test, y_test)
    preds = artifacts["preds"]
    or_pred = artifacts["or_pred"]
    voting_pred = artifacts["voting_pred"]
    gbc_pred = preds["GBC"]

    results = [
        paired_mcnemar(y_test, or_pred, voting_pred, "OR-Logic", "Traditional Voting"),
        paired_mcnemar(y_test, or_pred, gbc_pred, "OR-Logic", "GBC"),
    ]
    return pd.DataFrame(results)


def plot_recall_forest(summary_df: pd.DataFrame, out_path):
    sub = summary_df[summary_df["metric"] == "recall"].set_index("model_or_ensemble").loc[MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(MODEL_ORDER))
    means = sub["mean"].to_numpy()
    lo = means - sub["ci_low"].to_numpy()
    hi = sub["ci_high"].to_numpy() - means
    colors = ["#4c72b0"] * len(BASE_MODEL_NAMES) + ["#dd8452", "#c44e52"]

    ax.errorbar(means, y_pos, xerr=[lo, hi], fmt="o", capsize=4, color="black", ecolor="grey", markersize=0)
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter([m], [i], color=c, s=80, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Recall (positive class)")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"Recall across {len(SEEDS)} stratified resamples (mean, 95% CI)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(SEEDS)} stratified resamples over modified data...")
    raw_df = run_resamples()
    raw_path = OUT_DIR / "stability_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path}")

    summary_df = summarize(raw_df)
    summary_path = OUT_DIR / "stability_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    plot_recall_forest(summary_df, OUT_DIR / "recall_forest.png")
    print(f"Wrote {OUT_DIR / 'recall_forest.png'}")

    print("Running McNemar tests on the locked SEED=42 split...")
    mcnemar_df = run_mcnemar()
    mcnemar_path = OUT_DIR / "mcnemar_results.csv"
    mcnemar_df.to_csv(mcnemar_path, index=False)
    print(f"Wrote {mcnemar_path}")
    print(mcnemar_df.to_string(index=False))

    or_recall = summary_df[(summary_df["model_or_ensemble"] == "OR-Logic") & (summary_df["metric"] == "recall")].iloc[0]
    print(
        f"\nOR-Logic recall over {len(SEEDS)} resamples: mean={or_recall['mean']:.3f}, "
        f"std={or_recall['std']:.3f}, 95% CI=[{or_recall['ci_low']:.3f}, {or_recall['ci_high']:.3f}]"
    )


if __name__ == "__main__":
    main()
