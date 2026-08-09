"""
Reviewers R1.3 / R1.11 -- "why no neural network?" answered with one
empirical MLP row, not just reasoning.

STANDALONE COMPARISON BASELINE ONLY. This script does NOT touch
src/config.py, does NOT add MLP to the five locked base learners
(src/models.py), and does NOT feed MLP predictions into Traditional Voting
or OR-Logic. Every already-committed number in outputs/ is read verbatim
here, never recomputed. The MLP is one reasonable, lightly-tuned
architecture (see build_mlp() below) reported as-is -- no architecture
search, no per-metric tuning to force a particular outcome.

Fit on the SAME modified nine-feature set as the locked models
(src.experiments.common.load_modified_xy / locked_seed42_split), with the
scaler inside the Pipeline (fit on the training fold only, matching
src/models.py's convention). No class_weight / rebalancing, matching the
locked base learners' unbalanced setup -- see config.py's SVM_PARAMS
comment for why that's the fair comparison point.

Runs:
1. SEED=42 locked split -- full positive-class metrics + ROC AUC for the
   MLP. GBC and OR-Logic rows are read verbatim from outputs/results_clean.csv
   (feature_set == "modified") for side-by-side context; their AUC values
   are read verbatim from outputs/experiments/autofeat/autofeat_results.csv
   (feature_set == "hand_engineered_9feat", the same SEED=42 modified split)
   -- neither is refit here.
2. Stability -- resample_split() over the SAME 30 seeds (1..30) used by
   outputs/experiments/stability/stability_raw.csv, refitting the MLP each
   time; mean + 95% CI (2.5/97.5 percentile, matching
   src/experiments/stability.py's summarize()) for recall, precision,
   specificity, fnr, mcc.
3. Paired Wilcoxon signed-rank test, MLP vs GBC, on recall across those same
   30 seeds. GBC's per-seed recall is read from
   outputs/experiments/stability/stability_raw.csv (identical seed list,
   split function, and hyperparameters already computed there) rather than
   refit.

Outputs:
  outputs/experiments/mlp/mlp_results.csv          (MLP SEED=42 row + GBC/OR-Logic context rows)
  outputs/experiments/mlp/mlp_stability_raw.csv     (30 seeds x MLP metrics, per-seed)
  outputs/experiments/mlp/mlp_stability.csv         (30-seed mean + 95% CI, MLP only)
  outputs/experiments/mlp/mlp_vs_gbc_paired.csv     (Wilcoxon signed-rank result)
  outputs/experiments/mlp/NOTES.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluate import positive_class_metrics
from src.experiments.common import EXPERIMENTS_DIR, load_modified_xy, locked_seed42_split, resample_split

OUT_DIR = EXPERIMENTS_DIR / "mlp"
RESULTS_CLEAN_CSV = config.OUTPUTS_DIR / "results_clean.csv"
AUTOFEAT_RESULTS_CSV = EXPERIMENTS_DIR / "autofeat" / "autofeat_results.csv"
STABILITY_RAW_CSV = EXPERIMENTS_DIR / "stability" / "stability_raw.csv"

# Identical seed list to src/experiments/stability.py's SEEDS -- distinct
# from the locked config.SEED=42, which is reserved for the single-split run.
SEEDS = list(range(1, 31))
CI_METRICS = ["recall", "precision", "specificity", "fnr", "mcc"]
RESULT_COLS = ["feature_set", "model_or_ensemble", "n_test", "TP", "FN", "TN", "FP",
               "accuracy", "precision", "recall", "f1", "specificity", "fnr", "mcc", "auc"]

MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=1e-4,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=config.SEED,
)


def build_mlp() -> Pipeline:
    """One reasonable, lightly-tuned small-tabular MLP -- a standard two
    hidden-layer (64, 32) ReLU network with early stopping. NOT a member of
    config.py's five locked base learners; NOT included in Traditional
    Voting or OR-Logic. No class_weight, matching the locked base learners'
    unbalanced setup."""
    return Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(**MLP_PARAMS))])


def run_seed42() -> pd.DataFrame:
    X_train, X_test, y_train, y_test = locked_seed42_split()
    pipe = build_mlp()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    row = positive_class_metrics(y_test, pred, "MLP", "modified")
    row["auc"] = roc_auc_score(y_test, proba)
    mlp_row = pd.DataFrame([row])[RESULT_COLS]

    clean = pd.read_csv(RESULTS_CLEAN_CSV)
    context = clean[(clean["feature_set"] == "modified") & (clean["model_or_ensemble"].isin(["GBC", "OR-Logic"]))].copy()

    auc_lookup = pd.read_csv(AUTOFEAT_RESULTS_CSV)
    auc_lookup = auc_lookup[auc_lookup["feature_set"] == "hand_engineered_9feat"].set_index("model_or_ensemble")["auc"]
    context["auc"] = context["model_or_ensemble"].map(auc_lookup)
    context = context[RESULT_COLS]

    return pd.concat([mlp_row, context], ignore_index=True)


def run_stability_raw() -> pd.DataFrame:
    X, y, _ = load_modified_xy()
    rows = []
    for i, seed in enumerate(SEEDS):
        X_train, X_test, y_train, y_test = resample_split(X, y, seed)
        pipe = build_mlp()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = positive_class_metrics(y_test, pred, "MLP", "modified")
        row["seed"] = seed
        rows.append(row)
        print(f"  seed {seed} ({i + 1}/{len(SEEDS)}) done")
    return pd.DataFrame(rows)


def summarize_stability(raw_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for metric in CI_METRICS:
        vals = raw_df[metric].to_numpy(dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1)
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        out_rows.append(
            {
                "model_or_ensemble": "MLP",
                "metric": metric,
                "n_seeds": len(vals),
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(out_rows)


def paired_wilcoxon(mlp_recall: np.ndarray, gbc_recall: np.ndarray) -> pd.DataFrame:
    diffs = mlp_recall - gbc_recall
    stat, p = wilcoxon(mlp_recall, gbc_recall)
    t_p = ttest_rel(mlp_recall, gbc_recall).pvalue

    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        ranks = rankdata(np.abs(nonzero))
        w_pos = ranks[nonzero > 0].sum()
        w_neg = ranks[nonzero < 0].sum()
        effect_size = (w_pos - w_neg) / (w_pos + w_neg)
    else:
        effect_size = 0.0

    row = {
        "comparison": "MLP vs GBC",
        "metric": "recall",
        "n": len(diffs),
        "n_mlp_higher": int(np.sum(diffs > 0)),
        "n_gbc_higher": int(np.sum(diffs < 0)),
        "n_tied": int(np.sum(diffs == 0)),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "ttest_p": float(t_p),
        "effect_size": float(effect_size),
        "significant_at_0.05": bool(p < 0.05),
    }
    return pd.DataFrame([row])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Run 1/3: SEED=42 locked split...")
    results_df = run_seed42()
    results_df.to_csv(OUT_DIR / "mlp_results.csv", index=False)

    print("Run 2/3: 30-seed stability...")
    raw_df = run_stability_raw()
    raw_df.to_csv(OUT_DIR / "mlp_stability_raw.csv", index=False)
    summary_df = summarize_stability(raw_df)
    summary_df.to_csv(OUT_DIR / "mlp_stability.csv", index=False)

    print("Run 3/3: paired Wilcoxon, MLP vs GBC recall...")
    gbc_raw = pd.read_csv(STABILITY_RAW_CSV)
    gbc_raw = gbc_raw[gbc_raw["model_or_ensemble"] == "GBC"][["seed", "recall"]].rename(columns={"recall": "gbc_recall"})
    mlp_raw = raw_df[["seed", "recall"]].rename(columns={"recall": "mlp_recall"})
    merged = mlp_raw.merge(gbc_raw, on="seed", how="inner").sort_values("seed")
    assert len(merged) == len(SEEDS), f"expected {len(SEEDS)} matched seeds, got {len(merged)}"

    paired_df = paired_wilcoxon(merged["mlp_recall"].to_numpy(), merged["gbc_recall"].to_numpy())
    paired_df.to_csv(OUT_DIR / "mlp_vs_gbc_paired.csv", index=False)

    print("\n--- mlp_results.csv ---")
    print(results_df.to_string(index=False))
    print("\n--- mlp_stability.csv ---")
    print(summary_df.to_string(index=False))
    print("\n--- mlp_vs_gbc_paired.csv ---")
    print(paired_df.to_string(index=False))


if __name__ == "__main__":
    main()
