"""
Generates the manuscript figures main.tex actually references by filename
(fig23, fig30-fig43, fig51, fig52) at 300 DPI into outputs/figures_final/,
using a colorblind-safe, consistent per-model palette (shared with
scripts/build_final_figures.py's MODEL_COLORS).

Reads exclusively from the locked pipeline (src/config.py, src/data.py,
src/features.py, src/models.py -- untouched) via
src.experiments.common.locked_seed42_split() / fit_predict_all() / build_models(),
i.e. the exact SEED=42 split and hyperparameters src/run_baseline.py uses.
No config, hyperparameter, split, or metric is changed here.

shap_gbc.png and recall_forest.png are already produced at 300 DPI by
scripts/build_final_figures.py with the exact filenames main.tex expects --
this script does not duplicate them.

Usage: .venv/bin/python scripts/build_manuscript_figures.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc

from src.data import load_raw
from src.features import add_engineered_features
from src.experiments.common import locked_seed42_split, fit_predict_all, ensembles_from_preds

OUT_DIR = REPO_ROOT / "outputs" / "figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = REPO_ROOT / "outputs"

# ----------------------------------------------------------------------------
# Shared colorblind-safe palette (same slots as scripts/build_final_figures.py)
# ----------------------------------------------------------------------------
MODEL_COLORS = {
    "SVM": "#2a78d6",
    "RF": "#eb6834",
    "KNN": "#1baf7a",
    "LR": "#eda100",
    "GBC": "#e87ba4",
    "Traditional Voting": "#008300",
    "OR-Logic": "#4a3aa7",
}
MODEL_LABELS = {  # paper's display names
    "SVM": "Support Vector Machine",
    "RF": "Random Forest",
    "KNN": "K-Nearest Neighbors",
    "LR": "Logistic Regression",
    "GBC": "Gradient Boosting",
}
MUTED_GREY = "#898781"
INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"
FAIL_COLOR = "#c44e52"      # consistent "failure" color across fig51
NOFAIL_COLOR = "#4c72b0"    # consistent "no failure" color across fig51

BASE_MODEL_ORDER = ["SVM", "RF", "KNN", "LR", "GBC"]

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
    "pdf.fonttype": 42,
})


def save(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(REPO_ROOT)} + {pdf.name}")


# ============================================================================
# Shared data / model fit (once)
# ============================================================================
print("Loading data and fitting the five locked models on the SEED=42 split...")
df_raw = load_raw()
df_fe = add_engineered_features(df_raw)

X_train, X_test, y_train, y_test = locked_seed42_split()
preds, probas, fitted_models = fit_predict_all(X_train, y_train, X_test)
voting_pred, or_pred = ensembles_from_preds(preds, X_test.index)

ORIGINAL_FEATURES = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]",
]
ENGINEERED_FEATURES = ["RelationTemperature", "Power (W)", "WearRPM", "ToolWearTorque"]
MODIFIED_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES
TARGET = "Machine failure"

FEATURE_SHORT = {
    "Air temperature [K]": "Air temp. [K]",
    "Process temperature [K]": "Process temp. [K]",
    "Rotational speed [rpm]": "Rot. speed [rpm]",
    "Torque [Nm]": "Torque [Nm]",
    "Tool wear [min]": "Tool wear [min]",
    "RelationTemperature": "RelationTemp",
    "Power (W)": "Power (W)",
    "WearRPM": "WearRPM",
    "ToolWearTorque": "ToolWearTorque",
    "Machine failure": "Failure",
}


# ============================================================================
# fig23 -- normal-distribution curves of the five raw operational parameters
# ============================================================================
def fig23_normal_dist():
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()
    for i, feat in enumerate(ORIGINAL_FEATURES):
        ax = axes[i]
        x = df_raw[feat].to_numpy()
        mu, sigma = x.mean(), x.std(ddof=1)
        ax.hist(x, bins=40, density=True, color="#9ec5f4", edgecolor="white", linewidth=0.3, alpha=0.9)
        xs = np.linspace(x.min(), x.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="#0d366b", lw=1.8)
        ax.set_title(FEATURE_SHORT[feat], fontsize=9)
        ax.set_ylabel("Density")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[5].axis("off")
    fig.suptitle("Distribution of operational parameters with fitted normal curves", fontsize=11)
    fig.tight_layout()
    save(fig, "fig23")


# ============================================================================
# fig30 -- correlation matrix, original 5 features + target
# fig31 -- correlation matrix, 9 modified features + target
# ============================================================================
def _plot_corr(cols, out_name, title):
    corr = df_fe[cols + [TARGET]].rename(columns=FEATURE_SHORT).corr()
    fig, ax = plt.subplots(figsize=(0.85 * len(corr.columns) + 2, 0.85 * len(corr.columns) + 1))
    # Same diverging palette as src/evaluate.py's plot_correlation (green =
    # positive, red = negative) so fig30/fig31 match the rest of the repo's
    # correlation-matrix figures instead of an unrelated blue/maroon scale.
    cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9, as_cmap=True)
    sns.heatmap(
        corr, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt=".2f",
        square=True, linewidths=0.4, linecolor="white",
        annot_kws={"fontsize": 7}, cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title(title, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    save(fig, out_name)


def fig30_corr_original():
    _plot_corr(ORIGINAL_FEATURES, "fig30", "Correlation matrix -- original features")


def fig31_corr_modified():
    _plot_corr(MODIFIED_FEATURES, "fig31", "Correlation matrix -- original + engineered features")


# ============================================================================
# fig32/34/36/38/40 -- individual confusion matrices, modified data, per model
# fig33/35/37/39/41 -- individual ROC curves, modified data, per model
# ============================================================================
CM_FIG_BY_MODEL = {"SVM": "fig32", "RF": "fig34", "KNN": "fig36", "LR": "fig38", "GBC": "fig40"}
ROC_FIG_BY_MODEL = {"SVM": "fig33", "RF": "fig35", "KNN": "fig37", "LR": "fig39", "GBC": "fig41"}


def fig_confusion_single(name, out_name):
    cm = confusion_matrix(y_test, preds[name], labels=[0, 1])
    color = MODEL_COLORS[name]
    cmap = sns.light_palette(color, as_cmap=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False,
        xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"],
        linewidths=0.6, linecolor="white", annot_kws={"fontsize": 10}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(MODEL_LABELS[name], fontsize=9.5, color=color, fontweight="bold")
    fig.tight_layout()
    save(fig, out_name)


def fig_roc_single(name, out_name):
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    roc_auc = sk_auc(fpr, tpr)
    color = MODEL_COLORS[name]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(fpr, tpr, lw=2.0, color=color, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED_GREY, lw=0.9, linestyle="--", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(MODEL_LABELS[name], fontsize=9.5, color=color, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, out_name)


# ============================================================================
# fig42/43 -- confusion matrices, Traditional Voting / OR-Logic ensembles
# ============================================================================
def fig_confusion_ensemble(pred, name, color_key, out_name):
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    color = MODEL_COLORS[color_key]
    cmap = sns.light_palette(color, as_cmap=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False,
        xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"],
        linewidths=0.6, linecolor="white", annot_kws={"fontsize": 10}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name, fontsize=9.5, color=color, fontweight="bold")
    fig.tight_layout()
    save(fig, out_name)


# ============================================================================
# fig51 -- scatter/strip plots: engineered features vs. machine failure,
# with reference lines (Sec. 3.4 thresholds) overlaid. Line positions are the
# exact values reported in the manuscript; only their displayed legend text
# is rounded for readability -- the lines themselves are drawn at full
# precision (see THRESHOLD_LINES below), so plot geometry is unaffected by
# the rounded labels below it.
# ============================================================================
THRESHOLD_LINES = {
    "RelationTemperature": [-12.1, -7.6],
    "Power (W)": [3515.22, 8998.49],
    "WearRPM": [0.174],
    "ToolWearTorque": [0.26253],
}
THRESHOLD_LEGEND = {
    "RelationTemperature": "Observed range (non-failure): -12.1 to -7.6 K",
    "Power (W)": "Lower-failure-rate band: 3515 to 8998 W",
    "WearRPM": "Observed limit (non-failure): 0.174",
    "ToolWearTorque": "Reference value: 0.263",
}
PANEL_TITLES = {
    "RelationTemperature": "RelationTemperature [K]",
    "Power (W)": "Power [W]",
    "WearRPM": "WearRPM",
    "ToolWearTorque": "ToolWearTorque",
}


def fig51_scatter_engineered():
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    fail = df_fe[TARGET] == 1
    for i, feat in enumerate(ENGINEERED_FEATURES):
        ax = axes[i]
        x_ok = 0 + rng.uniform(-0.18, 0.18, size=(~fail).sum())
        x_fail = 1 + rng.uniform(-0.18, 0.18, size=fail.sum())
        ax.scatter(x_ok, df_fe.loc[~fail, feat], s=6, alpha=0.35, color=NOFAIL_COLOR, label="No Failure", linewidths=0)
        ax.scatter(x_fail, df_fe.loc[fail, feat], s=10, alpha=0.85, color=FAIL_COLOR, label="Failure", linewidths=0)
        for ln in THRESHOLD_LINES[feat]:
            ax.axhline(ln, color=INK, lw=0.8, linestyle=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Failure", "Failure"])
        ax.set_ylabel(feat)
        ax.set_title(PANEL_TITLES[feat], fontsize=8.5)
        # Reference-line caption goes below the x-axis tick labels (one
        # collapsed label per panel, even for panels with two dashed lines)
        # rather than an in-plot legend box -- with these scatter/strip
        # panels the data fills the full width at nearly every y-value
        # (including right at the dashed lines themselves, since two of the
        # four thresholds sit at the observed min/max), so any in-axes
        # legend placement sits on top of either the line or the points.
        # set_xlabel is guaranteed clear of both and is accounted for by
        # tight_layout automatically.
        ax.set_xlabel(THRESHOLD_LEGEND[feat], fontsize=6.5, color=MUTED_GREY, labelpad=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NOFAIL_COLOR, markersize=6, label="No Failure"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAIL_COLOR, markersize=6, label="Failure"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Engineered features vs. machine failure", fontsize=11, y=1.06)
    fig.tight_layout()
    save(fig, "fig51")


# ============================================================================
# fig52 -- simulated recall, OR-logic vs majority voting, vs. detector
# correlation rho. Source: outputs/experiments/simulation/sim_summary.csv.
# ============================================================================
def fig52_sim_or_vs_voting():
    df = pd.read_csv(OUTPUTS_DIR / "experiments" / "simulation" / "sim_summary.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    rule_colors = {"OR-logic": MODEL_COLORS["OR-Logic"], "Majority (>=3)": MODEL_COLORS["Traditional Voting"]}
    for rule, color in rule_colors.items():
        sub = df[df["rule"] == rule].sort_values("rho")
        ax.plot(sub["rho"], sub["recall_mean"], color=color, lw=1.8, marker="o", markersize=3.5, label=rule)
        ax.fill_between(sub["rho"], sub["recall_mean"] - sub["recall_std"],
                         sub["recall_mean"] + sub["recall_std"], color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Pairwise error correlation (rho) among the 5 base detectors")
    ax.set_ylabel("Ensemble recall (simulated)")
    ax.set_title("OR-logic vs. majority-voting recall vs. detector correlation")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "fig52")


if __name__ == "__main__":
    fig23_normal_dist()
    fig30_corr_original()
    fig31_corr_modified()
    for m in BASE_MODEL_ORDER:
        fig_confusion_single(m, CM_FIG_BY_MODEL[m])
        fig_roc_single(m, ROC_FIG_BY_MODEL[m])
    fig_confusion_ensemble(voting_pred, "Traditional Voting Ensemble", "Traditional Voting", "fig42")
    fig_confusion_ensemble(or_pred, "Proposed OR-Logic Ensemble", "OR-Logic", "fig43")
    fig51_scatter_engineered()
    fig52_sim_or_vs_voting()
    print("done")
