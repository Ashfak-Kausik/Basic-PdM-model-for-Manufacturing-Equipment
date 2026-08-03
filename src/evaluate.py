"""Metrics, ensemble derivation, and figure generation for the clean pipeline."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, auc

from . import config


def positive_class_metrics(y_true, y_pred, model_or_ensemble: str, feature_set: str) -> dict:
    """Confusion matrix + accuracy/precision/recall/F1/specificity/FNR/FPR/MCC
    for the POSITIVE (failure=1) class specifically."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = tn + fp + fn + tp

    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR / sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # TNR
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "feature_set": feature_set,
        "model_or_ensemble": model_or_ensemble,
        "n_test": int(n),
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fnr": fnr,
        "fpr": fpr,
        "mcc": mcc,
    }


def ensemble_from_predictions(pred_df: pd.DataFrame, threshold: int) -> np.ndarray:
    """pred_df: DataFrame of shape (n_test, n_models), each column a model's
    0/1 test-set predictions. Returns 0/1 array: 1 if the row-sum >= threshold."""
    return (pred_df.sum(axis=1) >= threshold).astype(int).to_numpy()


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_roc(proba_by_model: dict, y_test, out_path):
    """proba_by_model: {model_name: array of P(failure=1) on the test set}.
    All five models on one axes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves -- five base models, modified (feature-engineered) data")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_grid(pred_by_name: dict, y_test, out_path, ncols=4):
    """pred_by_name: {name: 0/1 predictions array}. One grid figure with a
    confusion-matrix heatmap per entry."""
    names = list(pred_by_name.keys())
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(names):
        ax = axes[i]
        cm = confusion_matrix(y_test, pred_by_name[name], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            ax=ax,
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion matrices -- modified (feature-engineered) data", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation(df_with_target: pd.DataFrame, out_path, title: str):
    corr = df_with_target.corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(corr.columns) + 2, 1.1 * len(corr.columns) + 1))
    cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9)
    sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
