"""
Publication-quality figure rebuild for outputs/figures_final/.

Reads exclusively from the already-committed outputs/*.csv produced by the
LOCKED pipeline (src/config.py, src/models.py, src/data.py -- untouched).
No config, model hyperparameter, split, or metric is changed here.

Two figures (roc_modified, shap_gbc) need raw arrays that were never persisted
to disk (ROC fpr/tpr points; per-sample SHAP values for the beeswarm) -- for
those two ONLY, this script re-runs the exact same locked, deterministic
functions already used by src/run_baseline.py and src/experiments/explain.py
(same SEED=42 split via src.experiments.common.locked_seed42_split(), same
build_models()) to regenerate those arrays. This is not new analysis: it is
byte-for-byte reproduction of numbers already on disk (verified below against
outputs/experiments/autofeat/autofeat_results.csv AUCs and
outputs/experiments/explain/gbc_shap_ranking.csv), done only because the raw
arrays themselves were never saved -- only the rendered PNGs and summary CSVs.

Usage: .venv/bin/python scripts/build_final_figures.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sk_auc

OUT_DIR = REPO_ROOT / "outputs" / "figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = REPO_ROOT / "outputs"

# ----------------------------------------------------------------------------
# Colorblind-safe categorical palette (dataviz skill reference palette,
# fixed slot order -- validated for adjacent-pair use on bars/lines/legends).
# ----------------------------------------------------------------------------
MODEL_COLORS = {
    "SVM": "#2a78d6",                # slot 1 blue
    "RF": "#eb6834",                 # slot 2 orange
    "KNN": "#1baf7a",                # slot 3 aqua
    "LR": "#eda100",                 # slot 4 yellow
    "GBC": "#e87ba4",                # slot 5 magenta
    "Traditional Voting": "#008300", # slot 6 green
    "Voting": "#008300",
    "OR-Logic": "#4a3aa7",           # slot 7 violet
}
MUTED_GREY = "#898781"
INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]  # light->dark

BASE_MODEL_ORDER = ["SVM", "RF", "KNN", "LR", "GBC"]
ALL_ORDER = BASE_MODEL_ORDER + ["Traditional Voting", "OR-Logic"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED_GREY,
    "axes.linewidth": 0.7,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "axes.grid": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed as real text, not curves, in the vector PDF
})


def save(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(REPO_ROOT)} + {pdf.name}")


# ============================================================================
# 1. roc_modified -- five base models, modified data, SEED=42 (restyle only;
#    raw fpr/tpr/AUC re-derived via the locked split + locked models, since
#    those points were never persisted -- only the old PNG was).
# ============================================================================
def fig_roc_modified():
    from src.experiments.common import locked_seed42_split, fit_predict_all

    X_train, X_test, y_train, y_test = locked_seed42_split()
    _, probas, _ = fit_predict_all(X_train, y_train, X_test)

    # Sanity check against already-committed AUCs for the same 9-feature
    # (hand_engineered) modified set, same locked split.
    autofeat = pd.read_csv(OUTPUTS_DIR / "experiments" / "autofeat" / "autofeat_results.csv")
    ref = autofeat[autofeat["feature_set"] == "hand_engineered_9feat"].set_index("model_or_ensemble")["auc"]

    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for name in BASE_MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        roc_auc = sk_auc(fpr, tpr)
        if name in ref.index:
            assert abs(roc_auc - ref[name]) < 1e-9, f"{name} AUC drift: {roc_auc} vs {ref[name]}"
        ax.plot(fpr, tpr, lw=1.6, color=MODEL_COLORS[name], label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], lw=0.8, ls="--", color=MUTED_GREY, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC -- base models, modified features (SEED=42)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "roc_modified")


# ============================================================================
# 2. confusion_grid_modified -- 5 models + Voting + OR-Logic, SEED=42, counts
#    + row-normalized percentages. Source: outputs/results_clean.csv (modified rows).
# ============================================================================
def fig_confusion_grid_modified():
    df = pd.read_csv(OUTPUTS_DIR / "results_clean.csv")
    df = df[df["feature_set"] == "modified"].set_index("model_or_ensemble")

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.0))
    axes_flat = axes.flatten()

    for i, name in enumerate(ALL_ORDER):
        ax = axes_flat[i]
        row = df.loc[name]
        tp, fn, tn, fp = row["TP"], row["FN"], row["TN"], row["FP"]
        mat = np.array([[tp, fn], [fp, tn]], dtype=float)
        row_pct = mat / mat.sum(axis=1, keepdims=True)

        im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred. fail", "Pred. OK"])
        ax.set_yticklabels(["Actual fail", "Actual OK"])
        for r in range(2):
            for c in range(2):
                count = int(mat[r, c])
                pct = row_pct[r, c] * 100
                txt_color = "white" if row_pct[r, c] > 0.6 else INK
                ax.text(c, r, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                         fontsize=7, color=txt_color)
        title_color = MODEL_COLORS.get(name, INK)
        ax.set_title(name, fontsize=8.5, color=title_color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    # 8th cell: shared colorbar + note, since there are only 7 models/ensembles
    ax8 = axes_flat[7]
    ax8.axis("off")
    cax = ax8.inset_axes([0.08, 0.2, 0.1, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("row %", fontsize=7, pad=6)
    cb.ax.tick_params(labelsize=6)
    ax8.text(0.42, 0.47, "Cell = count\n(row-normalized %)\nrow 1 = actual failures\nrow 2 = actual OK",
              fontsize=7, color=MUTED_GREY, va="center", transform=ax8.transAxes)

    fig.suptitle("Confusion matrices -- modified features, SEED=42", fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, "confusion_grid_modified")


# ============================================================================
# 3. recall_forest -- per-model/ensemble recall, 95% CI whiskers, sorted,
#    OR-Logic + Voting highlighted. Source: stability_summary.csv (30 seeds).
# ============================================================================
def fig_recall_forest():
    df = pd.read_csv(OUTPUTS_DIR / "experiments" / "stability" / "stability_summary.csv")
    df = df[df["metric"] == "recall"].copy()
    df = df.sort_values("mean", ascending=True).reset_index(drop=True)

    ensembles = {"Traditional Voting", "OR-Logic"}
    y_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, row in df.iterrows():
        name = row["model_or_ensemble"]
        is_ens = name in ensembles
        color = MODEL_COLORS.get(name, MUTED_GREY)
        lo, hi, mean = row["ci_low"], row["ci_high"], row["mean"]
        ax.hlines(i, lo, hi, color=color, lw=2.2 if is_ens else 1.3,
                   alpha=1.0 if is_ens else 0.75, zorder=2)
        marker = "D" if is_ens else "o"
        size = 42 if is_ens else 26
        ax.scatter(mean, i, color=color, marker=marker, s=size,
                    zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    labels = df["model_or_ensemble"].tolist()
    ax.set_yticklabels(labels)
    for tick, name in zip(ax.get_yticklabels(), labels):
        if name in ensembles:
            tick.set_fontweight("bold")
            tick.set_color(MODEL_COLORS.get(name, INK))

    ax.set_xlabel("Recall (mean +/- 95% CI, 30 seeds)")
    ax.set_xlim(0, 1)
    ax.set_title("Recall stability across 30 seeds")
    ax.grid(axis="x", zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "recall_forest")


# ============================================================================
# 4. imbalance_tradeoff -- recall vs precision, OR-Logic & GBC, four
#    strategies. Source: imbalance_results.csv.
# ============================================================================
def fig_imbalance_tradeoff():
    df = pd.read_csv(OUTPUTS_DIR / "experiments" / "imbalance" / "imbalance_results.csv")
    strategies = ["none (locked)", "SMOTE", "class_weight=balanced (parked)", "RandomUnderSampler"]
    strategy_labels = ["None\n(locked)", "SMOTE", "class_weight\n=balanced", "Random\nUnderSampler"]
    models = ["OR-Logic", "GBC"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    x = np.arange(len(strategies))
    width = 0.32

    for ax, model in zip(axes, models):
        sub = df[df["model_or_ensemble"] == model].set_index("strategy").loc[strategies]
        color = MODEL_COLORS[model]
        b1 = ax.bar(x - width / 2, sub["recall"], width, label="Recall",
                     color=color, alpha=1.0)
        b2 = ax.bar(x + width / 2, sub["precision"], width, label="Precision",
                     color=color, alpha=0.45, hatch="//", edgecolor=color, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_labels, fontsize=6.5)
        ax.set_title(model, fontsize=9, color=color, fontweight="bold")
        ax.set_ylim(0, 1.05)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right", frameon=False, ncol=1)
    fig.suptitle("Recall/precision tradeoff across imbalance strategies", fontsize=10)
    fig.tight_layout()
    save(fig, "imbalance_tradeoff")


# ============================================================================
# 5. shap_gbc -- SHAP beeswarm + mean|SHAP| bar, GBC, engineered features
#    marked with *. Raw SHAP array re-derived via the same locked GBC pipeline
#    + locked split used by src/experiments/explain.py (values were never
#    persisted -- only gbc_shap_ranking.csv (mean |SHAP|) and the old PNG were).
# ============================================================================
def fig_shap_gbc():
    import shap
    from src import config
    from src.experiments.common import locked_seed42_split
    from src.models import build_models

    X_train, X_test, y_train, y_test = locked_seed42_split()
    models = build_models()
    gbc_pipe = models["GBC"]
    gbc_pipe.fit(X_train, y_train)

    scaler = gbc_pipe.named_steps["scaler"]
    clf = gbc_pipe.named_steps["clf"]
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X_test_scaled)
    if isinstance(raw, list):
        shap_values = np.asarray(raw[1])
    else:
        raw = np.asarray(raw)
        shap_values = raw[:, :, 1] if raw.ndim == 3 else raw

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["is_engineered"] = ranking["feature"].isin(config.ENGINEERED_FEATURES)

    # Sanity check against the already-committed ranking csv.
    ref = pd.read_csv(OUTPUTS_DIR / "experiments" / "explain" / "gbc_shap_ranking.csv")
    ref_sorted = ref.set_index("feature")["mean_abs_shap"]
    for feat, val in ranking.set_index("feature")["mean_abs_shap"].items():
        assert abs(val - ref_sorted[feat]) < 1e-9, f"SHAP drift on {feat}: {val} vs {ref_sorted[feat]}"

    explanation = shap.Explanation(
        values=shap_values,
        data=X_test.to_numpy(),
        feature_names=[f"{f}*" if f in config.ENGINEERED_FEATURES else f for f in X_test.columns],
    )

    fig = plt.figure(figsize=(9.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])

    ax_bee = fig.add_subplot(gs[0, 0])
    plt.sca(ax_bee)
    shap.plots.beeswarm(explanation, show=False, color_bar=True, max_display=9)
    ax_bee.set_title("SHAP beeswarm -- GBC", fontsize=9)
    ax_bee.tick_params(labelsize=6.5)
    ax_bee.xaxis.label.set_fontsize(7.5)
    # shap.plots.beeswarm hardcodes larger font sizes for its own colorbar
    # (a separate axes appended to the current figure) -- rescale to match.
    for a in fig.axes:
        if a not in (ax_bee,):
            a.tick_params(labelsize=6)
            a.yaxis.label.set_fontsize(6.5)
            for t in a.texts:
                t.set_fontsize(6)

    ax_bar = fig.add_subplot(gs[0, 1])
    r = ranking.sort_values("mean_abs_shap")
    labels = [f"{f}*" if eng else f for f, eng in zip(r["feature"], r["is_engineered"])]
    colors = [MODEL_COLORS["GBC"] if eng else MUTED_GREY for eng in r["is_engineered"]]
    ax_bar.barh(labels, r["mean_abs_shap"], color=colors)
    ax_bar.set_xlabel("mean |SHAP value|")
    ax_bar.set_title("Mean |SHAP| ranking (* = engineered)", fontsize=9)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.tick_params(labelsize=6.5)

    fig.tight_layout()
    save(fig, "shap_gbc")


# ============================================================================
# 6. sim_or_vs_voting -- OR-logic vs majority-voting recall, rho 0->0.8.
#    Source: sim_summary.csv.
# ============================================================================
def fig_sim_or_vs_voting():
    df = pd.read_csv(OUTPUTS_DIR / "experiments" / "simulation" / "sim_summary.csv")

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    rule_colors = {"OR-logic": MODEL_COLORS["OR-Logic"], "Majority (>=3)": MODEL_COLORS["Traditional Voting"]}
    for rule, color in rule_colors.items():
        sub = df[df["rule"] == rule].sort_values("rho")
        ax.plot(sub["rho"], sub["recall_mean"], color=color, lw=1.8, marker="o", markersize=3.5, label=rule)
        ax.fill_between(sub["rho"], sub["recall_mean"] - sub["recall_std"],
                         sub["recall_mean"] + sub["recall_std"], color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Correlated-failure parameter (rho)")
    ax.set_ylabel("Recall (simulated)")
    ax.set_title("OR-logic vs majority-voting recall vs correlation")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "sim_or_vs_voting")


if __name__ == "__main__":
    fig_roc_modified()
    fig_confusion_grid_modified()
    fig_recall_forest()
    fig_imbalance_tradeoff()
    fig_shap_gbc()
    fig_sim_or_vs_voting()
    print("done")
