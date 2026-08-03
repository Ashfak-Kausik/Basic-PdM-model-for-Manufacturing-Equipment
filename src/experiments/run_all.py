"""
Runs all six R1/R2 reviewer-response analyses in order and writes
outputs/experiments/SUMMARY.md aggregating their key results.

Single command:
    .venv/bin/python -m src.experiments.run_all

Each analysis is also independently runnable, e.g.:
    .venv/bin/python -m src.experiments.stability

None of these scripts modify src/config.py, src/data.py, src/features.py,
src/models.py, src/evaluate.py, src/run_baseline.py, or
outputs/balanced_variant/ -- they only read the locked pipeline and write to
their own outputs/experiments/<name>/ subfolder.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.experiments.common import EXPERIMENTS_DIR
from src.experiments import stability, cost, explain, autofeat, simulation, imbalance

MODULES = [
    ("1. Stability / confidence intervals (R1.8)", stability),
    ("2. Inference latency & model size (R1.9)", cost),
    ("3. Explainability: SHAP + partial dependence (R2.3/R2.5/R2.6)", explain),
    ("4. Automatic feature baseline (R1.5)", autofeat),
    ("5. OR-logic vs. voting simulation (R1.4)", simulation),
    ("6. Imbalance strategy baselines (R1.6)", imbalance),
]


def run_experiments():
    timings = []
    for label, mod in MODULES:
        print(f"\n{'=' * 78}\nRunning {label}\n{'=' * 78}")
        t0 = time.perf_counter()
        mod.main()
        elapsed = time.perf_counter() - t0
        timings.append((label, elapsed))
        print(f"-- {label} done in {elapsed:.1f}s --")
    return timings


def build_summary(timings):
    stab_dir = EXPERIMENTS_DIR / "stability"
    cost_dir = EXPERIMENTS_DIR / "cost"
    explain_dir = EXPERIMENTS_DIR / "explain"
    autofeat_dir = EXPERIMENTS_DIR / "autofeat"
    sim_dir = EXPERIMENTS_DIR / "simulation"
    imbalance_dir = EXPERIMENTS_DIR / "imbalance"

    lines = ["# Experiments summary (reviewer responses R1.4-R1.9, R2.3, R2.5, R2.6)", ""]
    lines.append(
        "All six analyses below reuse the LOCKED canonical pipeline "
        "(`src/config.py`, `src/data.py`, `src/features.py`, `src/models.py`, "
        "`src/evaluate.py`) unchanged -- no hyperparameter, split, or model "
        "definition was altered to produce these results. `outputs/balanced_variant/` "
        "was read but not modified. Run via `.venv/bin/python -m src.experiments.run_all`; "
        "each analysis also has its own NOTES.md with more detail in its subfolder."
    )
    lines.append("")

    counter_findings = []

    # --- 1. Stability ---
    lines.append("## 1. Stability / confidence intervals (R1.8)")
    summary_df = pd.read_csv(stab_dir / "stability_summary.csv")
    or_recall = summary_df[(summary_df.model_or_ensemble == "OR-Logic") & (summary_df.metric == "recall")].iloc[0]
    gbc_recall = summary_df[(summary_df.model_or_ensemble == "GBC") & (summary_df.metric == "recall")].iloc[0]
    mcnemar_df = pd.read_csv(stab_dir / "mcnemar_results.csv")
    mn_vote = mcnemar_df[mcnemar_df["comparison"] == "OR-Logic vs Traditional Voting"].iloc[0]
    mn_gbc = mcnemar_df[mcnemar_df["comparison"] == "OR-Logic vs GBC"].iloc[0]
    lines.append(
        f"- OR-Logic recall over 30 stratified resamples: **mean={or_recall['mean']:.3f}, "
        f"std={or_recall['std']:.3f}, 95% CI=[{or_recall['ci_low']:.3f}, {or_recall['ci_high']:.3f}]** "
        f"(GBC for comparison: mean={gbc_recall['mean']:.3f}, CI=[{gbc_recall['ci_low']:.3f}, {gbc_recall['ci_high']:.3f}])."
    )
    lines.append(
        f"- McNemar, OR-Logic vs. Traditional Voting (locked SEED=42 split): "
        f"statistic={mn_vote['statistic']:.3f}, p={mn_vote['p_value']:.4f} "
        f"({'significant' if mn_vote['significant_at_0.05'] else 'NOT significant'} at alpha=0.05)."
    )
    lines.append(
        f"- McNemar, OR-Logic vs. GBC (locked SEED=42 split): "
        f"statistic={mn_gbc['statistic']:.3f}, p={mn_gbc['p_value']:.6f} "
        f"({'significant' if mn_gbc['significant_at_0.05'] else 'NOT significant'} at alpha=0.05)."
    )
    lines.append(
        "- **Counter-to-expectation flag:** on the one locked SEED=42 split specifically "
        "(as opposed to the 30-resample average), OR-Logic and GBC have IDENTICAL "
        "TP/FN (68/17) -- OR-Logic adds zero additional true positives over GBC alone on "
        "that split, only 18 extra false positives (28 vs. GBC's 10). That is exactly why "
        "the OR-Logic vs. GBC McNemar test is significant in GBC's favor on discordant "
        "pairs (0 rows where OR-Logic is uniquely right vs. 18 where GBC is uniquely "
        "right) even though OR-Logic wins on recall *on average* across the 30 resamples "
        "(mean recall 0.776 vs. GBC's 0.739, a real aggregate advantage). The single "
        "SEED=42 split is not representative of the aggregate pattern here -- report the "
        "30-resample CI, not the single-split comparison, as the primary evidence."
    )
    counter_findings.append(
        "R1.8: on the single locked SEED=42 split, OR-Logic adds ZERO extra true positives over GBC alone "
        "(identical TP/FN) and only extra false positives -- McNemar favors GBC on that split. The 30-resample "
        "average still favors OR-Logic (mean recall 0.776 vs GBC's 0.739), so the split-specific result is a "
        "sampling artifact, not evidence against OR-Logic in general -- but the single-split comparison alone "
        "would have told the opposite story."
    )
    lines.append("")

    # --- 2. Cost ---
    lines.append("## 2. Inference latency & model size (R1.9)")
    cost_df = pd.read_csv(cost_dir / "cost_summary.csv")
    or_cost = cost_df[cost_df.model_or_ensemble == "OR-Logic"].iloc[0]
    gbc_cost = cost_df[cost_df.model_or_ensemble == "GBC"].iloc[0]
    sysinfo_first_lines = (cost_dir / "system_info.txt").read_text().splitlines()[2:4]
    lines.append(f"- Measured on: {'; '.join(sysinfo_first_lines)}")
    lines.append(
        f"- Per-sample inference latency: **OR-Logic={or_cost['per_sample_latency_us_mean']:.2f}us** "
        f"(sum of all 5 members + aggregation), **GBC alone={gbc_cost['per_sample_latency_us_mean']:.2f}us**. "
        f"OR-Logic costs {or_cost['per_sample_latency_us_mean'] / gbc_cost['per_sample_latency_us_mean']:.1f}x "
        f"GBC's latency per sample, dominated by KNN's per-sample cost within the ensemble."
    )
    lines.append(f"- Model size: OR-Logic (sum of 5 members) = {or_cost['model_size_kb']:.0f} KB; GBC alone = {gbc_cost['model_size_kb']:.0f} KB.")
    lines.append("")

    # --- 3. Explainability ---
    lines.append("## 3. Explainability: SHAP + partial dependence (R2.3/R2.5/R2.6)")
    gbc_rank = pd.read_csv(explain_dir / "gbc_shap_ranking.csv")
    top3 = gbc_rank.head(3)
    lines.append(
        "- GBC top-3 SHAP features (mean |SHAP|, positive class): "
        + ", ".join(f"{r.feature} ({r.mean_abs_shap:.3f}, {'engineered' if r.is_engineered else 'raw'})" for r in top3.itertuples())
        + "."
    )
    lines.append(
        "- Engineered features cluster near the top of both GBC's and RF's rankings, but do NOT "
        "uniformly occupy rank #1 (GBC's single top feature is the raw Rotational speed [rpm]; "
        "RF's top feature IS engineered, RelationTemperature). Full detail: `explain/NOTES.md`."
    )
    lines.append("")

    # --- 4. Auto-feature baseline ---
    lines.append("## 4. Automatic feature baseline (R1.5)")
    focus = pd.read_csv(autofeat_dir / "or_logic_gbc_comparison.csv")
    for name in ["OR-Logic", "GBC"]:
        sub = focus[focus.model_or_ensemble == name]
        hand = sub[sub.feature_set.str.contains("hand")].iloc[0]
        auto = sub[sub.feature_set.str.contains("auto")].iloc[0]
        lines.append(
            f"- **{name}**: hand-engineered AUC={hand['auc']:.4f}, recall={hand['recall']:.3f} vs. "
            f"auto-polynomial (20 features) AUC={auto['auc']:.4f}, recall={auto['recall']:.3f}."
        )
    lines.append(
        "- **Verdict: hand-engineered domain features beat the automatic PolynomialFeatures(degree=2) "
        "baseline** on both AUC and recall for OR-Logic and GBC, despite the auto set having more than "
        "double the feature count -- supports the paper's physics-informed feature engineering claim. "
        "TSFresh/wavelets are N/A: AI4I 2020 is 10,000 independent snapshots, not a per-machine time "
        "series (no timestamp, no machine ID linking rows over time). Full detail: `autofeat/NOTES.md`."
    )
    lines.append("")

    # --- 5. Simulation ---
    lines.append("## 5. OR-logic vs. voting simulation (R1.4)")
    sim_df = pd.read_csv(sim_dir / "sim_summary.csv")
    rho_lo = sim_df.rho.min()
    rho_hi = sim_df.rho.max()
    or_lo = sim_df[(sim_df.rho == rho_lo) & (sim_df.rule == "OR-logic")].iloc[0]
    maj_lo = sim_df[(sim_df.rho == rho_lo) & (sim_df.rule == "Majority (>=3)")].iloc[0]
    or_hi = sim_df[(sim_df.rho == rho_hi) & (sim_df.rule == "OR-logic")].iloc[0]
    maj_hi = sim_df[(sim_df.rho == rho_hi) & (sim_df.rule == "Majority (>=3)")].iloc[0]
    lines.append(
        f"- At rho={rho_lo:.1f} (independent detector errors): OR-logic recall={or_lo['recall_mean']:.3f} vs. "
        f"majority-voting recall={maj_lo['recall_mean']:.3f} (gap={or_lo['recall_mean'] - maj_lo['recall_mean']:+.3f})."
    )
    lines.append(
        f"- At rho={rho_hi:.1f} (highly correlated detector errors): OR-logic recall={or_hi['recall_mean']:.3f} vs. "
        f"majority-voting recall={maj_hi['recall_mean']:.3f} (gap={or_hi['recall_mean'] - maj_hi['recall_mean']:+.3f})."
    )
    lines.append(
        "- **Takeaway**: OR-logic's recall advantage over majority voting is largest when the base "
        "detectors are diverse (low error correlation) and individually precise -- shrinking but never "
        "vanishing as detector correlation rises. This is the analytical reason the locked config keeps "
        "base models unbalanced (see #6 below): rebalancing every member individually doesn't move along "
        "this diversity axis, it degrades every member's precision at once. Full detail: `simulation/NOTES.md`."
    )
    lines.append("")

    # --- 6. Imbalance ---
    lines.append("## 6. Imbalance strategy baselines (R1.6)")
    imb_df = pd.read_csv(imbalance_dir / "imbalance_results.csv")
    or_rows = imb_df[imb_df.model_or_ensemble == "OR-Logic"].set_index("strategy")
    for strat in ["none (locked)", "SMOTE", "class_weight=balanced (parked)", "RandomUnderSampler"]:
        r = or_rows.loc[strat]
        lines.append(f"- OR-Logic + **{strat}**: recall={r['recall']:.3f}, precision={r['precision']:.3f}, specificity={r['specificity']:.3f}")
    lines.append(
        "- **One-liner**: every rebalancing strategy (class weighting, SMOTE, undersampling) buys OR-Logic "
        "roughly +15-18pp recall at a precision/specificity collapse (precision falls from 0.71 to "
        "0.12-0.16, specificity from 0.99 to 0.75-0.83) -- because OR-Logic's false-positive rate is "
        "bounded below by the UNION of all five members' false-positive rates, and rebalancing degrades "
        "every member's precision simultaneously. The locked (no-resampling) config is the balanced choice "
        "specifically because OR-Logic is a union operator, not despite it."
    )
    lines.append(
        "- **Counter-to-expectation flag**: SMOTE and RandomUnderSampler do NOT behave more gently than "
        "class_weight='balanced' here -- for OR-Logic, SMOTE's precision (0.155) and RandomUnderSampler's "
        "precision (0.122) are actually WORSE than class_weight='balanced' (0.162). Also note GBC's "
        "'class_weight=balanced' row is identical to its 'none' row by construction (GradientBoostingClassifier "
        "has no class_weight parameter) -- see `imbalance/NOTES.md` caveat."
    )
    counter_findings.append(
        "R1.6: SMOTE and RandomUnderSampler are NOT gentler than class_weight='balanced' for OR-Logic -- "
        "both give WORSE precision (0.155 and 0.122 respectively) than balanced class weights (0.162), "
        "despite the common assumption that resampling is a 'softer' imbalance fix than reweighting."
    )
    lines.append("")

    # --- Timing ---
    lines.append("## Runtime")
    for label, elapsed in timings:
        lines.append(f"- {label}: {elapsed:.1f}s")
    lines.append("")

    # --- Counter-to-expectation roundup ---
    lines.append("## Findings that ran counter to expectation (flagged, not tuned away)")
    for f in counter_findings:
        lines.append(f"- {f}")
    lines.append(
        "- (Auto-feature baseline, R1.5, ran AS expected: hand-engineered features won. Explainability, "
        "R2.x, ran mostly as expected with one nuance: GBC's single top feature is raw Rotational speed, "
        "not an engineered one, even though engineered features cluster near the top overall.)"
    )

    text = "\n".join(lines) + "\n"
    out_path = EXPERIMENTS_DIR / "SUMMARY.md"
    out_path.write_text(text)
    print(f"\nWrote {out_path}")
    return text


def main():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    timings = run_experiments()
    summary_text = build_summary(timings)
    print("\n" + "=" * 78)
    print(summary_text)


if __name__ == "__main__":
    main()
