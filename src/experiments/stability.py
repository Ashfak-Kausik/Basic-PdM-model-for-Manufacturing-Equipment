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

In addition, runs a paired Wilcoxon signed-rank test across the 30
stratified resamples themselves (not the single locked split): for each of
the 30 seeds, OR-Logic's recall/fnr is paired against Traditional Voting's
and against GBC's on that SAME seed's split, then the 30 paired differences
are tested. This is the primary evidence for the paper's cross-seed recall
claims (main.tex Sec. 5.6): it directly measures the recall/FNR trade-off
across repeated partitions, which the single-split McNemar test above
cannot. (This closes a gap where outputs/experiments/stability/paired_tests.csv
existed as a committed artifact but had no generating script in src/ --
this function is that script.)

Outputs:
  outputs/experiments/stability/stability_raw.csv       (30 seeds x 7 models/ensembles, per-metric)
  outputs/experiments/stability/stability_summary.csv    (one row per model/ensemble x metric: mean/std/ci_low/ci_high)
  outputs/experiments/stability/mcnemar_results.csv      (the two single-split paired tests)
  outputs/experiments/stability/paired_tests.csv         (the two 30-seed paired Wilcoxon tests, recall + fnr)
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
from scipy.stats import rankdata, ttest_rel, wilcoxon
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


def paired_wilcoxon_30seed(raw_df: pd.DataFrame, name_a: str, name_b: str, metric: str) -> dict:
    """Pairs name_a's and name_b's per-seed `metric` values (matched by seed,
    same 30-seed resamples raw_df.run_resamples() already produced) and runs
    a paired Wilcoxon signed-rank test, plus a paired t-test and a rank-based
    effect size for context. Mirrors src/experiments/mlp.py's paired_wilcoxon
    (same statistics, same effect-size formula), applied here to the two
    ensemble comparisons instead of MLP vs GBC."""
    a = raw_df[raw_df["model_or_ensemble"] == name_a].set_index("seed")[metric]
    b = raw_df[raw_df["model_or_ensemble"] == name_b].set_index("seed")[metric]
    a, b = a.align(b, join="inner")
    assert len(a) == len(SEEDS), f"expected {len(SEEDS)} matched seeds for {name_a} vs {name_b}/{metric}, got {len(a)}"

    diffs = (a - b).to_numpy()
    stat, p = wilcoxon(a.to_numpy(), b.to_numpy())
    t_p = ttest_rel(a.to_numpy(), b.to_numpy()).pvalue

    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        ranks = rankdata(np.abs(nonzero))
        w_pos = ranks[nonzero > 0].sum()
        w_neg = ranks[nonzero < 0].sum()
        effect_size = (w_pos - w_neg) / (w_pos + w_neg)
    else:
        effect_size = 0.0

    return {
        "comparison": f"{name_a} vs {name_b}",
        "metric": metric,
        "n": len(diffs),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "ttest_p": float(t_p),
        "effect_size": float(effect_size),
        "significant_at_0.05": bool(p < 0.05),
    }


def run_paired_tests(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The 30-seed paired Wilcoxon tests behind main.tex's Sec. 5.6 claims:
    OR-Logic vs Traditional Voting and OR-Logic vs GBC, on both recall and fnr."""
    rows = []
    for other in ["Traditional Voting", "GBC"]:
        for metric in ["recall", "fnr"]:
            rows.append(paired_wilcoxon_30seed(raw_df, "OR-Logic", other, metric))
    return pd.DataFrame(rows)


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

    print("Running paired Wilcoxon signed-rank tests across the 30 resamples "
          "(OR-Logic vs Traditional Voting, OR-Logic vs GBC)...")
    paired_df = run_paired_tests(raw_df)
    paired_path = OUT_DIR / "paired_tests.csv"
    paired_df.to_csv(paired_path, index=False)
    print(f"Wrote {paired_path}")
    print(paired_df.to_string(index=False))

    or_recall = summary_df[(summary_df["model_or_ensemble"] == "OR-Logic") & (summary_df["metric"] == "recall")].iloc[0]
    print(
        f"\nOR-Logic recall over {len(SEEDS)} resamples: mean={or_recall['mean']:.3f}, "
        f"std={or_recall['std']:.3f}, 95% CI=[{or_recall['ci_low']:.3f}, {or_recall['ci_high']:.3f}]"
    )


if __name__ == "__main__":
    main()
