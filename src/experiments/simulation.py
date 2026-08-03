"""
Reviewer R1.4 -- Monte Carlo simulation: why does OR-logic beat majority
voting on recall, independent of this specific dataset?

Simulates M=5 base detectors on a rare-positive population (prevalence
~3.4%, matching AI4I 2020's actual failure rate). Each detector's per-trial
recall/false-positive-rate is drawn from a Normal distribution centered on
that model's EMPIRICAL mean/std from the stability experiment
(outputs/experiments/stability/stability_summary.csv, task 1 -- reused, not
re-estimated here), so the simulation is anchored to what we actually
observed, not arbitrary numbers. A Gaussian-copula construction gives the
five detectors a controllable pairwise error-correlation rho (compound
symmetry): rho=0 means detector errors are independent; rho close to 1
means detectors tend to succeed/fail together (e.g. all trained on
correlated features, or all fooled by the same hard cases).

For each rho in [0, 0.8], many trials are simulated and OR-logic (fire if
any detector fires) is compared against majority voting (fire if >=3 of 5)
on expected recall and specificity.

Outputs:
  outputs/experiments/simulation/sim_summary.csv
  outputs/experiments/simulation/recall_vs_rho.png
  outputs/experiments/simulation/recall_vs_specificity.png
  outputs/experiments/simulation/NOTES.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.experiments.common import BASE_MODEL_NAMES, EXPERIMENTS_DIR

OUT_DIR = EXPERIMENTS_DIR / "simulation"
STABILITY_SUMMARY = EXPERIMENTS_DIR / "stability" / "stability_summary.csv"

PREVALENCE = 0.034  # matches AI4I 2020's ~3.39% failure rate
N_PER_TRIAL = 20_000
N_TRIALS = 40
RHOS = np.round(np.arange(0.0, 0.81, 0.1), 2)
RNG_SEED = 12345  # simulation-only seed; independent of config.SEED (the modeling pipeline's locked seed)


def load_base_detector_stats() -> dict:
    """Anchors the simulation's per-detector recall/FPR to the empirical
    values measured in the stability experiment (task 1), falling back to
    hardcoded values transcribed from that run if it hasn't been run yet in
    this environment."""
    if STABILITY_SUMMARY.exists():
        df = pd.read_csv(STABILITY_SUMMARY)
        stats = {}
        for name in BASE_MODEL_NAMES:
            r = df[(df["model_or_ensemble"] == name) & (df["metric"] == "recall")].iloc[0]
            s = df[(df["model_or_ensemble"] == name) & (df["metric"] == "specificity")].iloc[0]
            stats[name] = dict(
                recall_mean=float(r["mean"]), recall_std=float(r["std"]),
                fpr_mean=float(1 - s["mean"]), fpr_std=float(s["std"]),
            )
        return stats
    # Fallback (values transcribed from outputs/experiments/stability/stability_summary.csv)
    return {
        "SVM": dict(recall_mean=0.339, recall_std=0.052, fpr_mean=0.0018, fpr_std=0.0007),
        "RF": dict(recall_mean=0.736, recall_std=0.047, fpr_mean=0.0027, fpr_std=0.0009),
        "KNN": dict(recall_mean=0.371, recall_std=0.047, fpr_mean=0.0037, fpr_std=0.0012),
        "LR": dict(recall_mean=0.239, recall_std=0.047, fpr_mean=0.0033, fpr_std=0.0011),
        "GBC": dict(recall_mean=0.739, recall_std=0.044, fpr_mean=0.0032, fpr_std=0.0013),
    }


def simulate_trial(rng, rho, detector_stats):
    m = len(detector_stats)
    n = N_PER_TRIAL

    y = rng.random(n) < PREVALENCE

    # per-trial jitter around each detector's empirical recall/FPR
    recalls = np.clip(
        [rng.normal(detector_stats[name]["recall_mean"], detector_stats[name]["recall_std"]) for name in BASE_MODEL_NAMES],
        0.01, 0.99,
    )
    fprs = np.clip(
        [rng.normal(detector_stats[name]["fpr_mean"], detector_stats[name]["fpr_std"]) for name in BASE_MODEL_NAMES],
        0.0001, 0.5,
    )

    # Gaussian copula: compound-symmetry correlation rho among the 5 detectors' latent scores
    sigma = np.full((m, m), rho)
    np.fill_diagonal(sigma, 1.0)
    sigma += np.eye(m) * 1e-9  # numerical safety for Cholesky at rho close to 1
    L = np.linalg.cholesky(sigma)
    z = rng.standard_normal((n, m)) @ L.T

    preds = np.zeros((n, m), dtype=int)
    pos_thr = norm.ppf(1 - recalls)  # P(z > thr | positive) = recall
    neg_thr = norm.ppf(1 - fprs)     # P(z > thr | negative) = fpr
    for j in range(m):
        preds[y, j] = (z[y, j] > pos_thr[j]).astype(int)
        preds[~y, j] = (z[~y, j] > neg_thr[j]).astype(int)

    row_sum = preds.sum(axis=1)
    or_pred = row_sum >= 1
    majority_pred = row_sum >= 3

    def recall_spec(pred):
        recall = pred[y].mean() if y.any() else float("nan")
        specificity = (~pred[~y]).mean() if (~y).any() else float("nan")
        return recall, specificity

    or_recall, or_spec = recall_spec(or_pred)
    maj_recall, maj_spec = recall_spec(majority_pred)
    return or_recall, or_spec, maj_recall, maj_spec


def run_sweep():
    detector_stats = load_base_detector_stats()
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for rho in RHOS:
        or_recalls, or_specs, maj_recalls, maj_specs = [], [], [], []
        for _ in range(N_TRIALS):
            or_r, or_s, maj_r, maj_s = simulate_trial(rng, rho, detector_stats)
            or_recalls.append(or_r)
            or_specs.append(or_s)
            maj_recalls.append(maj_r)
            maj_specs.append(maj_s)

        for rule, recalls, specs in [("OR-logic", or_recalls, or_specs), ("Majority (>=3)", maj_recalls, maj_specs)]:
            recalls, specs = np.array(recalls), np.array(specs)
            rows.append(
                {
                    "rho": rho,
                    "rule": rule,
                    "recall_mean": recalls.mean(),
                    "recall_std": recalls.std(ddof=1),
                    "specificity_mean": specs.mean(),
                    "specificity_std": specs.std(ddof=1),
                }
            )
        print(f"rho={rho:.1f}  OR recall={np.mean(or_recalls):.3f}  Majority recall={np.mean(maj_recalls):.3f}")

    return pd.DataFrame(rows), detector_stats


def plot_recall_vs_rho(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for rule, color in [("OR-logic", "#c44e52"), ("Majority (>=3)", "#4c72b0")]:
        sub = df[df["rule"] == rule].sort_values("rho")
        ax.plot(sub["rho"], sub["recall_mean"], marker="o", label=rule, color=color)
        ax.fill_between(sub["rho"], sub["recall_mean"] - sub["recall_std"], sub["recall_mean"] + sub["recall_std"], alpha=0.15, color=color)
    ax.set_xlabel("Pairwise error correlation (rho) among the 5 base detectors")
    ax.set_ylabel("Ensemble recall (mean +/- 1 std over trials)")
    ax.set_title(f"Simulated ensemble recall vs. detector correlation (prevalence={PREVALENCE:.1%})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_recall_vs_specificity(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for rule, color, marker in [("OR-logic", "#c44e52", "o"), ("Majority (>=3)", "#4c72b0", "s")]:
        sub = df[df["rule"] == rule].sort_values("rho")
        sc = ax.scatter(sub["specificity_mean"], sub["recall_mean"], c=sub["rho"], cmap="viridis", marker=marker, s=70, label=rule, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Specificity (mean over trials)")
    ax.set_ylabel("Recall (mean over trials)")
    ax.set_title("Recall vs. specificity trade-off across rho (color = rho)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("rho")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {len(RHOS)} rho values x {N_TRIALS} trials x n={N_PER_TRIAL} (prevalence={PREVALENCE:.1%})...")
    df, detector_stats = run_sweep()

    out_path = OUT_DIR / "sim_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    plot_recall_vs_rho(df, OUT_DIR / "recall_vs_rho.png")
    plot_recall_vs_specificity(df, OUT_DIR / "recall_vs_specificity.png")
    print("Wrote recall_vs_rho.png and recall_vs_specificity.png")

    rho0 = df[df["rho"] == RHOS[0]]
    rho_hi = df[df["rho"] == RHOS[-1]]
    or_gap_low_rho = rho0[rho0["rule"] == "OR-logic"]["recall_mean"].iloc[0] - rho0[rho0["rule"] == "Majority (>=3)"]["recall_mean"].iloc[0]
    or_gap_hi_rho = rho_hi[rho_hi["rule"] == "OR-logic"]["recall_mean"].iloc[0] - rho_hi[rho_hi["rule"] == "Majority (>=3)"]["recall_mean"].iloc[0]

    detector_lines = "\n".join(
        f"  {name}: recall~{s['recall_mean']:.3f} (+/-{s['recall_std']:.3f}), FPR~{s['fpr_mean']:.4f} (+/-{s['fpr_std']:.4f})"
        for name, s in detector_stats.items()
    )

    notes = f"""# Simulation notes (R1.4)

## Setup
5 simulated base detectors, population prevalence {PREVALENCE:.1%} (matches AI4I 2020's
actual ~3.39% failure rate). Each detector's per-trial recall/FPR is drawn from a Normal
distribution centered on its EMPIRICAL mean/std from the stability experiment (task 1):

{detector_lines}

A Gaussian copula gives the 5 detectors' errors a controllable pairwise correlation rho
(compound symmetry), swept from 0 (independent errors) to 0.8 (highly correlated errors).
{N_TRIALS} trials of n={N_PER_TRIAL} simulated instances per rho value.

## Result
- At rho=0 (independent detectors): OR-logic recall exceeds majority-voting recall by
  {or_gap_low_rho:+.3f} (see `recall_vs_rho.png`).
- At rho={RHOS[-1]:.1f} (highly correlated detectors): the OR-logic vs. majority recall
  gap is {or_gap_hi_rho:+.3f}.

## Takeaway
OR-logic's recall advantage over majority voting is **largest when the base detectors are
diverse (low pairwise error correlation) and individually reasonably precise** -- exactly
the setting the empirical detector stats above describe (each base model on the real data
has specificity/FPR in a similar narrow band, and the five models use structurally
different algorithms -- SVM, tree ensembles, instance-based, and linear -- so their errors
are not tightly coupled). As detectors become more correlated (rho -> 0.8), OR-logic's
extra recall shrinks toward whatever the majority rule already captures, because a
"union of five near-identical detectors" behaves like one detector.

This links directly back to why `class_weight='balanced'` was removed from SVM/RF/LR in
the locked config (see `outputs/balanced_variant/README.txt`): making every base model
individually MORE willing to fire (lower precision, higher FPR) does not just move along
this same simulated curve -- it pushes every detector's FPR up simultaneously, and
OR-logic's false-positive count scales with the union of all five FPRs regardless of rho.
The simulation here isolates the *diversity* dimension (rho) holding each detector's own
recall/FPR fixed; the balanced-weights experiment (imbalance/, R1.6) isolates the
*individual detector precision* dimension. Both point the same direction: OR-logic works
best over diverse, individually precise detectors, not over detectors that have each been
independently pushed toward high recall/low precision.
"""
    (OUT_DIR / "NOTES.md").write_text(notes)
    print(f"Wrote {OUT_DIR / 'NOTES.md'}")
    print("\n" + notes)


if __name__ == "__main__":
    main()
