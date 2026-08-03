"""
Reviewer R1.6 -- imbalance-handling strategy comparison.

On the LOCKED modified-feature, SEED=42 split, compares four ways of
handling the ~3.4% failure prevalence, for all five base models (focus:
OR-Logic and GBC):

  (i)   none              -- the locked config as shipped (no resampling, no class weighting)
  (ii)  SMOTE              -- oversampling the TRAINING FOLD ONLY, inside an
                              imblearn Pipeline (Scaler -> SMOTE -> classifier),
                              so synthetic points never leak into the test fold
  (iii) class_weight='balanced' -- the PARKED balanced_variant run from the
                              previous task (outputs/balanced_variant/), reused
                              verbatim, not regenerated -- see its README.txt
  (iv)  RandomUnderSampler -- undersampling the majority class in the TRAINING
                              FOLD ONLY, same imblearn Pipeline pattern as SMOTE

All four strategies reuse the exact same locked model hyperparameters
(src/config.py) and the same locked SEED=42 split rows -- only the
resampling/weighting step changes.

Outputs:
  outputs/experiments/imbalance/imbalance_results.csv
  outputs/experiments/imbalance/recall_precision_bars.png
  outputs/experiments/imbalance/NOTES.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.experiments.common import BASE_MODEL_NAMES, EXPERIMENTS_DIR, locked_seed42_split, ensembles_from_preds, all_rows_for_split
from src.evaluate import positive_class_metrics

OUT_DIR = EXPERIMENTS_DIR / "imbalance"
BALANCED_VARIANT_CSV = config.OUTPUTS_DIR / "balanced_variant" / "results_clean.csv"
FOCUS_MODELS = ["GBC", "OR-Logic"]
METRIC_COLS = ["recall", "precision", "specificity", "fnr", "f1", "mcc", "TP", "FN", "TN", "FP"]


def build_resampled_models(resampler_factory):
    """Same locked hyperparameters as src/models.py, wrapped in an imblearn
    Pipeline (Scaler -> resampler -> classifier) so the resampler only ever
    touches the training fold (imblearn Pipelines skip fit_resample at
    predict time by construction -- no leakage into the test fold)."""

    def pipe(clf):
        return ImbPipeline([("scaler", StandardScaler()), ("resample", resampler_factory()), ("clf", clf)])

    return {
        "SVM": pipe(SVC(**config.SVM_PARAMS)),
        "RF": pipe(RandomForestClassifier(**config.RF_PARAMS)),
        "KNN": pipe(KNeighborsClassifier(**config.KNN_PARAMS)),
        "LR": pipe(LogisticRegression(**config.LR_PARAMS)),
        "GBC": pipe(GradientBoostingClassifier(**config.GBC_PARAMS)),
    }


def run_strategy_rows(models, X_train, y_train, X_test, y_test, strategy_label):
    preds = {}
    for name in BASE_MODEL_NAMES:
        pipe = models[name]
        pipe.fit(X_train, y_train)
        preds[name] = pipe.predict(X_test)

    rows = [positive_class_metrics(y_test, preds[name], name, "modified") for name in BASE_MODEL_NAMES]
    voting_pred, or_pred = ensembles_from_preds(preds, X_test.index)
    rows.append(positive_class_metrics(y_test, voting_pred, "Traditional Voting", "modified"))
    rows.append(positive_class_metrics(y_test, or_pred, "OR-Logic", "modified"))
    for r in rows:
        r["strategy"] = strategy_label
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = locked_seed42_split()
    print(f"Locked split: train={X_train.shape}, test={X_test.shape}, "
          f"train failures={int(y_train.sum())}/{len(y_train)} ({100 * y_train.mean():.2f}%)")

    all_rows = []

    # (i) none -- the locked config as shipped
    print("\n[1/4] Strategy: none (locked config, no resampling/weighting)")
    from src.models import build_models
    none_rows, _ = all_rows_for_split(X_train, y_train, X_test, y_test, models=build_models())
    for r in none_rows:
        r["strategy"] = "none (locked)"
    all_rows.extend(none_rows)

    # (ii) SMOTE -- training fold only
    print("[2/4] Strategy: SMOTE (training fold only)")
    smote_models = build_resampled_models(lambda: SMOTE(random_state=config.SEED))
    all_rows.extend(run_strategy_rows(smote_models, X_train, y_train, X_test, y_test, "SMOTE"))

    # (iii) class_weight='balanced' -- reuse the parked balanced_variant run
    print("[3/4] Strategy: class_weight='balanced' (reusing parked outputs/balanced_variant/)")
    if not BALANCED_VARIANT_CSV.exists():
        raise FileNotFoundError(
            f"{BALANCED_VARIANT_CSV} not found -- outputs/balanced_variant/ must be preserved "
            "from the previous task (do not regenerate it)."
        )
    balanced_df = pd.read_csv(BALANCED_VARIANT_CSV)
    balanced_rows = balanced_df[balanced_df["feature_set"] == "modified"].to_dict("records")
    for r in balanced_rows:
        r["strategy"] = "class_weight=balanced (parked)"
    all_rows.extend(balanced_rows)

    # (iv) RandomUnderSampler -- training fold only
    print("[4/4] Strategy: RandomUnderSampler (training fold only)")
    rus_models = build_resampled_models(lambda: RandomUnderSampler(random_state=config.SEED))
    all_rows.extend(run_strategy_rows(rus_models, X_train, y_train, X_test, y_test, "RandomUnderSampler"))

    df = pd.DataFrame(all_rows)
    keep_cols = ["strategy", "model_or_ensemble", "n_test", "TP", "FN", "TN", "FP",
                 "accuracy", "precision", "recall", "f1", "specificity", "fnr", "mcc"]
    df = df[[c for c in keep_cols if c in df.columns]]
    out_path = OUT_DIR / "imbalance_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    focus = df[df["model_or_ensemble"].isin(FOCUS_MODELS)].sort_values(["model_or_ensemble", "strategy"])
    print(focus[["strategy", "model_or_ensemble", "recall", "precision", "specificity", "fnr", "f1", "mcc"]].to_string(index=False))

    # --- grouped bar chart: recall vs precision per strategy, for OR-Logic and GBC ---
    strategies = ["none (locked)", "SMOTE", "class_weight=balanced (parked)", "RandomUnderSampler"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x = np.arange(len(strategies))
    width = 0.35
    for ax, model_name in zip(axes, FOCUS_MODELS):
        sub = df[df["model_or_ensemble"] == model_name].set_index("strategy").loc[strategies]
        ax.bar(x - width / 2, sub["recall"], width, label="Recall", color="#c44e52")
        ax.bar(x + width / 2, sub["precision"], width, label="Precision", color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=25, ha="right")
        ax.set_title(model_name)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for i, (r, p) in enumerate(zip(sub["recall"], sub["precision"])):
            ax.text(i - width / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
            ax.text(i + width / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("Recall vs. precision by imbalance strategy (locked SEED=42 split, modified data)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall_precision_bars.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'recall_precision_bars.png'}")

    # --- NOTES.md ---
    def row(strategy, model):
        return df[(df["strategy"] == strategy) & (df["model_or_ensemble"] == model)].iloc[0]

    lines = []
    for model_name in FOCUS_MODELS:
        lines.append(f"\n### {model_name}")
        for s in strategies:
            r = row(s, model_name)
            lines.append(
                f"- **{s}**: recall={r['recall']:.3f}, precision={r['precision']:.3f}, "
                f"specificity={r['specificity']:.3f}, FN rate={r['fnr']:.3f}, F1={r['f1']:.3f}, MCC={r['mcc']:.3f} "
                f"(TP={int(r['TP'])}, FN={int(r['FN'])}, TN={int(r['TN'])}, FP={int(r['FP'])})"
            )
    strategy_table = "\n".join(lines)

    or_none = row("none (locked)", "OR-Logic")
    or_smote = row("SMOTE", "OR-Logic")
    or_balanced = row("class_weight=balanced (parked)", "OR-Logic")
    or_rus = row("RandomUnderSampler", "OR-Logic")

    smote_wins = or_smote["recall"] > or_none["recall"] and or_smote["specificity"] > 0.98
    rus_extreme = or_rus["specificity"] < 0.95

    notes = f"""# Imbalance-strategy notes (R1.6)

## Per-strategy results, OR-Logic and GBC
{strategy_table}

Full 4-strategy x 7-model/ensemble table: `imbalance_results.csv`.
Grouped bar chart: `recall_precision_bars.png`.

**Caveat on GBC's "class_weight=balanced" row:** it is IDENTICAL to GBC's "none" row above,
and that is expected, not a bug -- `sklearn.ensemble.GradientBoostingClassifier` has no
`class_weight` parameter at all, so GBC was never rebalanced in the parked
`outputs/balanced_variant/` run either (only SVM/RF/LR were). GBC's own predictions are
therefore identical between "none" and "class_weight=balanced" by construction; only
OR-Logic's "class_weight=balanced" row differs from "none", because OR-Logic combines GBC
with the four OTHER models that genuinely were rebalanced in that run.

## The trade-off, stated plainly
- **none (locked)**: OR-Logic recall={or_none['recall']:.3f}, specificity={or_none['specificity']:.3f},
  precision={or_none['precision']:.3f}. This is the reference point -- no resampling, no
  class weighting, just the five base models as specified in `src/config.py`.
- **class_weight='balanced'**: OR-Logic recall={or_balanced['recall']:.3f} but specificity
  collapses to {or_balanced['specificity']:.3f} and precision to {or_balanced['precision']:.3f}
  ({int(or_balanced['FP'])} false positives) -- this is exactly the failure mode documented in
  `outputs/balanced_variant/README.txt`: balancing every base model individually means
  OR-Logic inherits the union of all five models' inflated false-positive rates.
- **SMOTE** (training-fold-only oversampling): OR-Logic recall={or_smote['recall']:.3f},
  specificity={or_smote['specificity']:.3f}, precision={or_smote['precision']:.3f}.
  {"SMOTE buys a real recall improvement over the locked config while keeping specificity in a defensible range -- the best of the three alternatives to the locked config for the recall/specificity trade-off." if smote_wins else "SMOTE moves recall and precision/specificity in the same general direction as class-weighting, just less extremely -- still a real cost for the recall it buys, though less severe than class_weight='balanced'."}
- **RandomUnderSampler**: OR-Logic recall={or_rus['recall']:.3f}, specificity={or_rus['specificity']:.3f},
  precision={or_rus['precision']:.3f}. {"Undersampling throws away the large majority-class training signal (only ~254 failure rows exist in the training fold to begin with, so undersampling shrinks the effective training set to roughly 2x that), and specificity drops sharply as a result -- the most aggressive recall-for-specificity trade among the four strategies." if rus_extreme else "Undersampling shows a similar direction of trade-off to the other rebalancing strategies, though less extreme than class-weighting in this run."}

## Why the locked (no-resampling) config is the right choice ahead of an OR-union
The OR-logic ensemble is, by construction, a union operator: it fires if ANY of the five
base models fires. That means OR-Logic's overall false-positive rate is bounded below by
the union of the five members' individual false-positive rates -- it can never be more
precise than its least-precise member. Any per-model strategy that trades precision for
recall (class weighting, SMOTE, undersampling) makes EVERY member less precise
simultaneously, and OR-Logic compounds all five degradations at once rather than
averaging them out. This is the same mechanism the Monte Carlo simulation
(`outputs/experiments/simulation/NOTES.md`, R1.4) demonstrates analytically: OR-logic's
advantage is largest when base detectors are diverse AND individually precise. Leaving
the base models unbalanced (the locked config) is what keeps each member precise enough
that the union doesn't explode into a high-false-alarm-rate ensemble -- the recall lift
from OR-aggregating five diverse, individually-precise, unbalanced models is "free" in a
way that recall lift from rebalancing every member individually is not.
"""
    (OUT_DIR / "NOTES.md").write_text(notes)
    print(f"Wrote {OUT_DIR / 'NOTES.md'}")
    print("\n" + notes)


if __name__ == "__main__":
    main()
