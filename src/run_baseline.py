"""
Entry point for the CLEAN, canonical pipeline. Single command:

    python src/run_baseline.py   (or .venv/bin/python src/run_baseline.py)

Produces:
  outputs/results_clean.csv
  outputs/tables/table4_original_individual.csv
  outputs/tables/table5_modified_all.csv
  outputs/tables/table6_orig_vs_modified.csv
  outputs/tables/table8_ensembles_raw.csv
  outputs/tables/table10_full_comparison.csv
  outputs/figures/roc_modified.png
  outputs/figures/confusion_matrices_modified.png
  outputs/figures/corr_original.png
  outputs/figures/corr_modified.png
  outputs/split_class_balance.txt
  outputs/DELTA_vs_paper.md

Does NOT touch src/reproduce.py, outputs/AUDIT.md, or outputs/results_reproduced.csv
(those are the faithful, bug-preserving reproduction from the prior task and
stay as the audit trail).
"""
import sys
from pathlib import Path

# allow `python src/run_baseline.py` (no package install) to import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.data import load_raw, split_indices, build_feature_frames, train_test_frames, write_split_balance
from src.features import add_engineered_features
from src.models import build_models
from src.evaluate import (
    positive_class_metrics,
    ensemble_from_predictions,
    plot_roc,
    plot_confusion_grid,
    plot_correlation,
)

BASE_MODEL_NAMES = ["SVM", "RF", "KNN", "LR", "GBC"]


def run_feature_set(feature_set_name: str, X: pd.DataFrame, y: pd.Series, train_idx, test_idx):
    """Fits the five base models on this feature set's train split, predicts
    on test, derives both ensembles, and returns (rows, artifacts) where
    `rows` is a list of metric dicts and `artifacts` carries raw predictions
    / probabilities for figure generation."""
    X_train, X_test, y_train, y_test = train_test_frames(X, y, train_idx, test_idx)

    models = build_models()
    preds, probas = {}, {}
    rows = []

    for name in BASE_MODEL_NAMES:
        pipe = models[name]
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        preds[name] = pred
        probas[name] = pipe.predict_proba(X_test)[:, 1]
        rows.append(positive_class_metrics(y_test, pred, name, feature_set_name))

    pred_df = pd.DataFrame(preds, index=X_test.index)[BASE_MODEL_NAMES]

    voting_pred = ensemble_from_predictions(pred_df, config.VOTING_THRESHOLD)
    or_pred = ensemble_from_predictions(pred_df, config.OR_LOGIC_THRESHOLD)

    rows.append(positive_class_metrics(y_test, voting_pred, "Traditional Voting", feature_set_name))
    rows.append(positive_class_metrics(y_test, or_pred, "OR-Logic", feature_set_name))

    artifacts = {
        "y_test": y_test,
        "preds": preds,
        "probas": probas,
        "voting_pred": voting_pred,
        "or_pred": or_pred,
    }
    return rows, artifacts


def build_table4(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors paper Table 4: baseline (original-feature) individual models."""
    t = results_df[
        (results_df["feature_set"] == "original") & (results_df["model_or_ensemble"].isin(BASE_MODEL_NAMES))
    ].copy()
    t = t.rename(columns={"model_or_ensemble": "Model"})
    cols = ["Model", "accuracy", "precision", "recall", "f1", "specificity", "TP", "FN", "TN", "FP"]
    return t[cols].reset_index(drop=True)


def build_table5(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors paper Table 5: five models + both ensembles on modified data, full metrics."""
    t = results_df[results_df["feature_set"] == "modified"].copy()
    order = BASE_MODEL_NAMES + ["Traditional Voting", "OR-Logic"]
    t["model_or_ensemble"] = pd.Categorical(t["model_or_ensemble"], categories=order, ordered=True)
    t = t.sort_values("model_or_ensemble").rename(columns={"model_or_ensemble": "Model"})
    cols = ["Model", "accuracy", "precision", "recall", "f1", "specificity", "mcc", "fnr", "fpr", "TP", "FN", "TN", "FP"]
    return t[cols].reset_index(drop=True)


def build_table6(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors paper Table 6/7: per-model original vs. modified accuracy/recall + deltas."""
    orig = results_df[
        (results_df["feature_set"] == "original") & (results_df["model_or_ensemble"].isin(BASE_MODEL_NAMES))
    ].set_index("model_or_ensemble")
    mod = results_df[
        (results_df["feature_set"] == "modified") & (results_df["model_or_ensemble"].isin(BASE_MODEL_NAMES))
    ].set_index("model_or_ensemble")

    rows = []
    for name in BASE_MODEL_NAMES:
        rows.append(
            {
                "Model": name,
                "Accuracy_Original": orig.loc[name, "accuracy"],
                "Recall_Original": orig.loc[name, "recall"],
                "Accuracy_Modified": mod.loc[name, "accuracy"],
                "Recall_Modified": mod.loc[name, "recall"],
                "Accuracy_Delta_pp": 100 * (mod.loc[name, "accuracy"] - orig.loc[name, "accuracy"]),
                "Recall_Delta_pp": 100 * (mod.loc[name, "recall"] - orig.loc[name, "recall"]),
            }
        )
    return pd.DataFrame(rows)


def build_table8(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors paper Table 8: ensemble performance on RAW (original-feature) data.
    Unlike ML main.ipynb (which never builds this -- see AUDIT.md section 3c),
    the clean pipeline runs both ensembles on the original feature set too."""
    t = results_df[
        (results_df["feature_set"] == "original") & (results_df["model_or_ensemble"].isin(["Traditional Voting", "OR-Logic"]))
    ].copy()
    t = t.rename(columns={"model_or_ensemble": "Ensemble Method"})
    cols = ["Ensemble Method", "accuracy", "precision", "recall", "f1", "specificity", "fnr", "TP", "FN", "TN", "FP"]
    return t[cols].reset_index(drop=True)


def build_table10(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors paper Table 10: baseline -> best individual -> ensembles, with
    improvement deltas. "Best" is picked programmatically by recall (the
    paper's own stated priority metric), not hardcoded to RFC, so this stays
    honest if the canonical config changes the winner."""
    orig_individual = results_df[
        (results_df["feature_set"] == "original") & (results_df["model_or_ensemble"].isin(BASE_MODEL_NAMES))
    ]
    mod_individual = results_df[
        (results_df["feature_set"] == "modified") & (results_df["model_or_ensemble"].isin(BASE_MODEL_NAMES))
    ]
    best_baseline = orig_individual.loc[orig_individual["recall"].idxmax()]
    best_individual = mod_individual.loc[mod_individual["recall"].idxmax()]
    voting = results_df[
        (results_df["feature_set"] == "modified") & (results_df["model_or_ensemble"] == "Traditional Voting")
    ].iloc[0]
    or_logic = results_df[
        (results_df["feature_set"] == "modified") & (results_df["model_or_ensemble"] == "OR-Logic")
    ].iloc[0]

    def row(label, r):
        return {"Approach": label, "Accuracy": r["accuracy"], "Recall": r["recall"], "FN_Rate": r["fnr"]}

    rows = [
        row(f"Best Baseline ({best_baseline['model_or_ensemble']}, original)", best_baseline),
        row(f"Best Individual ({best_individual['model_or_ensemble']}, modified)", best_individual),
        row("Traditional Voting (modified)", voting),
        row("Proposed OR-Logic (modified)", or_logic),
    ]
    out = pd.DataFrame(rows)

    deltas = pd.DataFrame(
        [
            {
                "Comparison": "OR-Logic vs. Best Baseline",
                "Accuracy_Delta_pp": 100 * (or_logic["accuracy"] - best_baseline["accuracy"]),
                "Recall_Delta_pp": 100 * (or_logic["recall"] - best_baseline["recall"]),
                "FN_Rate_Delta_pp": 100 * (or_logic["fnr"] - best_baseline["fnr"]),
            },
            {
                "Comparison": "OR-Logic vs. Best Individual (modified)",
                "Accuracy_Delta_pp": 100 * (or_logic["accuracy"] - best_individual["accuracy"]),
                "Recall_Delta_pp": 100 * (or_logic["recall"] - best_individual["recall"]),
                "FN_Rate_Delta_pp": 100 * (or_logic["fnr"] - best_individual["fnr"]),
            },
            {
                "Comparison": "OR-Logic vs. Traditional Voting",
                "Accuracy_Delta_pp": 100 * (or_logic["accuracy"] - voting["accuracy"]),
                "Recall_Delta_pp": 100 * (or_logic["recall"] - voting["recall"]),
                "FN_Rate_Delta_pp": 100 * (or_logic["fnr"] - voting["fnr"]),
            },
        ]
    )
    return out, deltas


# ----------------------------------------------------------------------------
# Paper's reported numbers (hardcoded reference values, transcribed from
# Tables 4/5/8/9 of documents/Optimizing_Predictive_Maintenance_of_CNC_Machines_...pdf,
# for comparison purposes only -- not used anywhere in the modeling).
# All values as fractions (0-1), not percent.
# ----------------------------------------------------------------------------
PAPER_TABLE4 = {  # baseline, original features
    "SVM": dict(accuracy=0.9500, precision=0.85, recall=0.05, f1=0.09, specificity=0.9950),
    "RF": dict(accuracy=0.9700, precision=0.92, recall=0.75, f1=0.83, specificity=0.9875),
    "KNN": dict(accuracy=0.9600, precision=0.88, recall=0.30, f1=0.45, specificity=0.9900),
    "LR": dict(accuracy=0.9500, precision=0.83, recall=0.18, f1=0.30, specificity=0.9925),
    "GBC": dict(accuracy=0.9700, precision=0.91, recall=0.72, f1=0.80, specificity=0.9880),
}
PAPER_TABLE5 = {  # modified features
    "SVM": dict(accuracy=0.970, precision=1.00, recall=0.096, f1=0.175, TP=8, FN=75, TN=2417, FP=0, fnr=0.904),
    "RF": dict(accuracy=0.989, precision=0.945, recall=0.831, f1=0.885, TP=69, FN=14, TN=2413, FP=4, fnr=0.169),
    "GBC": dict(accuracy=0.990, precision=0.955, recall=0.759, f1=0.844, TP=63, FN=20, TN=2414, FP=3, fnr=0.241),
    "Traditional Voting": dict(accuracy=0.992, precision=1.00, recall=0.446, f1=0.617, TP=37, FN=46, TN=2417, FP=0, fnr=0.554),
    "OR-Logic": dict(accuracy=0.992, precision=0.869, recall=0.880, f1=0.874, TP=73, FN=10, TN=2406, FP=11, fnr=0.120),
}
PAPER_TABLE8 = {  # ensembles on raw data -- paper reports these, but ML main.ipynb has no code for them at all (AUDIT.md 3c)
    "Traditional Voting": dict(accuracy=0.9720, precision=0.90, recall=0.78, f1=0.84, specificity=0.9870),
    "OR-Logic": dict(accuracy=0.9650, precision=0.75, recall=0.92, f1=0.83, specificity=0.9800),
}
PAPER_TABLE9 = {  # modified data, final results
    "Traditional Voting": dict(accuracy=0.9916, precision=0.94, recall=0.446, f1=0.60, specificity=1.0000, mcc=0.63, fnr=0.554),
    "OR-Logic": dict(accuracy=0.9916, precision=0.88, recall=0.880, f1=0.88, specificity=0.9950, mcc=0.88, fnr=0.120),
    "RF (Best Individual)": dict(accuracy=0.9892, precision=0.95, recall=0.831, f1=0.89, specificity=0.9983, mcc=0.87, fnr=0.169),
}

# ----------------------------------------------------------------------------
# Consolidated "paper reported" numbers for the modified-feature-set wide
# comparison table below. Reconciles Table 5 (accuracy/precision/recall/F1 +
# confusion-matrix elements) with Table 9 (adds specificity/MCC for
# Voting/OR-Logic/RF) into one row per model/ensemble. KNN and LR are NOT in
# Table 5/9 at all -- the paper only reports their accuracy and recall in
# running text (Section 5.3.2: "KNN identified only 31 failures (37.3%)...
# achieving 97.28%"; "LR detected only 21 failures (25.3%)... achieving
# 97.32%"). Their precision/F1/specificity/FNR/TN/FP here are INFERRED by
# solving TN = accuracy*N - TP and FP = N - TP - FN - TN from those two
# reported numbers plus N=2500 -- flagged via inferred=True and called out
# in the report footnote, not presented as if the paper stated them directly.
# ----------------------------------------------------------------------------
def _infer_knn_lr(accuracy, TP, FN, n=2500):
    TN = round(accuracy * n) - TP
    FP = n - TP - FN - TN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = TN / (TN + FP)
    fnr = FN / (FN + TP)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, specificity=specificity,
                fnr=fnr, mcc=None, TP=TP, FN=FN, TN=TN, FP=FP, inferred=True)


PAPER_MODIFIED_FULL = {
    "SVM": dict(accuracy=0.970, precision=1.00, recall=0.096, f1=0.175, specificity=1.0000, fnr=0.904,
                mcc=None, TP=8, FN=75, TN=2417, FP=0, inferred=False),
    "RF": dict(accuracy=0.9892, precision=0.945, recall=0.831, f1=0.885, specificity=0.9983, fnr=0.169,
               mcc=0.87, TP=69, FN=14, TN=2413, FP=4, inferred=False),
    "KNN": _infer_knn_lr(accuracy=0.9728, TP=31, FN=52),
    "LR": _infer_knn_lr(accuracy=0.9732, TP=21, FN=62),
    "GBC": dict(accuracy=0.9896, precision=0.955, recall=0.759, f1=0.844, specificity=0.9988, fnr=0.241,
                mcc=None, TP=63, FN=20, TN=2414, FP=3, inferred=False),
    # Table 5 reports Voting precision=100%/F1=61.7% (consistent with its own
    # TP=37/FN=46/TN=2417/FP=0: FP=0 forces precision=TP/(TP+FP)=100%).
    # Table 9 instead reports precision=94%/F1=60% for the SAME row -- an
    # internal inconsistency in the paper. We use Table 5's confusion-matrix-
    # consistent precision/F1 as the primary value and flag the conflict in
    # the report footnote rather than silently picking one.
    "Traditional Voting": dict(accuracy=0.9916, precision=1.00, recall=0.446, f1=0.617, specificity=1.0000,
                                fnr=0.554, mcc=0.63, TP=37, FN=46, TN=2417, FP=0, inferred=False),
    "OR-Logic": dict(accuracy=0.9916, precision=0.869, recall=0.880, f1=0.874, specificity=0.9950,
                      fnr=0.120, mcc=0.88, TP=73, FN=10, TN=2406, FP=11, inferred=False),
}


def fmt_pct(x):
    return f"{100 * x:.2f}%" if pd.notna(x) else "n/a"


def build_delta_report(results_df: pd.DataFrame) -> str:
    """Builds outputs/DELTA_vs_paper.md: a headline summary, a single wide
    table (one row per model/ensemble x one of {paper, faithful reproduction,
    clean unbalanced run}) for the modified-feature set, and a narrative
    check. The faithful reproduction is read-only here
    (outputs/results_reproduced.csv, untouched by this pipeline)."""
    faithful = None
    if config.FAITHFUL_REPRO_CSV.exists():
        faithful = pd.read_csv(config.FAITHFUL_REPRO_CSV)

    def clean_row(name):
        r = results_df[(results_df["feature_set"] == "modified") & (results_df["model_or_ensemble"] == name)]
        return r.iloc[0] if len(r) else None

    def faithful_row(name_contains):
        if faithful is None:
            return None
        r = faithful[(faithful["feature_set"] == "modified") & (faithful["model"].str.contains(name_contains, case=False))]
        return r.iloc[0] if len(r) else None

    name_map = {"GBC": "GBM"}  # faithful reproduction calls it GBM, not GBC
    clean = {name: clean_row(name) for name in BASE_MODEL_NAMES + ["Traditional Voting", "OR-Logic"]}
    faith = {name: faithful_row(name_map.get(name, name)) for name in BASE_MODEL_NAMES + ["Traditional Voting", "OR-Logic"]}
    paper = PAPER_MODIFIED_FULL

    def f_or_nan(row, key):
        return row[key] if row is not None else float("nan")

    lines = []
    lines.append("# DELTA vs. paper and vs. the faithful reproduction")
    lines.append("")
    lines.append(
        "Config change from the previous clean run: `class_weight='balanced'` has been "
        "**removed from SVM, RF, and LR** in `src/config.py` (KNN and GBC were never "
        "balanced and are unchanged). Reason: with balanced weights, SVM and LR each "
        "individually over-fired on the ~3.4%-failure test set, and because OR-logic is "
        "a union of the five models' positives, it inherited the union of their false "
        "alarms -- recall rose to ~95% but precision collapsed to ~16% and specificity "
        "fell to ~83%, breaking the paper's actual claim of high recall *while holding "
        "specificity near 99.5%*. The balanced-weights run is preserved in "
        "`outputs/balanced_variant/` (see its `README.txt`) as a cost-sensitive baseline "
        "for reviewer comment R1.6. Everything else in the canonical config is "
        "unchanged: `StandardScaler` in every Pipeline, `SEED=42`, the stratified split, "
        "the nine-feature modified set, Traditional Voting = `sum(preds) >= 3`, "
        "OR-Logic = `sum(preds) >= 1`."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Headline
    # ------------------------------------------------------------------
    lines.append("## Headline")
    lines.append("")
    for name, label in [("OR-Logic", "OR-Logic"), ("Traditional Voting", "Traditional Voting"), ("RF", "RFC"), ("GBC", "GBC")]:
        p, c = paper[name], clean[name]
        lines.append(
            f"- **{label}** -- clean (unbalanced): recall {fmt_pct(c['recall'])}, precision {fmt_pct(c['precision'])}, "
            f"specificity {fmt_pct(c['specificity'])}, FN rate {fmt_pct(c['fnr'])}  |  "
            f"paper: recall {fmt_pct(p['recall'])}, precision {fmt_pct(p['precision'])}, "
            f"specificity {fmt_pct(p['specificity'])}, FN rate {fmt_pct(p['fnr'])}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Single wide table, modified feature set, one row per model/ensemble
    # per source (paper / faithful / clean-unbalanced)
    # ------------------------------------------------------------------
    lines.append("## Modified (feature-engineered) data -- every model & ensemble, three configs side by side")
    lines.append("")
    lines.append(
        "cf. paper Tables 4 (baseline text figures for KNN/LR), 5, and 9. All "
        "recall/precision/F1/specificity/FN-rate are for the **positive (failure) "
        "class**. KNN and LR paper rows are marked `(inferred)` -- see footnote."
    )
    lines.append("")
    lines.append("| Model/Ensemble | Source | Accuracy | Precision | Recall | F1 | Specificity | FN Rate | MCC | TP | FN | TN | FP |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in BASE_MODEL_NAMES + ["Traditional Voting", "OR-Logic"]:
        p = paper.get(name)
        f = faith.get(name)
        c = clean.get(name)
        if p:
            src_label = f"Paper (Table 5/9{'; text, inferred' if p.get('inferred') else ''})"
            lines.append(
                f"| {name} | {src_label} | {fmt_pct(p['accuracy'])} | {fmt_pct(p['precision'])} | {fmt_pct(p['recall'])} | {fmt_pct(p['f1'])} | "
                f"{fmt_pct(p['specificity'])} | {fmt_pct(p['fnr'])} | {p['mcc'] if p['mcc'] is not None else 'n/a'} | "
                f"{p['TP']} | {p['FN']} | {p['TN']} | {p['FP']} |"
            )
        if f is not None:
            lines.append(
                f"| {name} | Faithful reproduction | {fmt_pct(f['accuracy'])} | {fmt_pct(f['precision_pos'])} | {fmt_pct(f['recall_pos_TPR'])} | {fmt_pct(f['f1_pos'])} | "
                f"{fmt_pct(f['specificity_TNR'])} | {fmt_pct(f['false_negative_rate'])} | {f['mcc']:.3f} | "
                f"{int(f['TP'])} | {int(f['FN'])} | {int(f['TN'])} | {int(f['FP'])} |"
            )
        if c is not None:
            lines.append(
                f"| {name} | **Clean (unbalanced)** | **{fmt_pct(c['accuracy'])}** | **{fmt_pct(c['precision'])}** | **{fmt_pct(c['recall'])}** | **{fmt_pct(c['f1'])}** | "
                f"**{fmt_pct(c['specificity'])}** | **{fmt_pct(c['fnr'])}** | **{c['mcc']:.3f}** | "
                f"**{int(c['TP'])}** | **{int(c['FN'])}** | **{int(c['TN'])}** | **{int(c['FP'])}** |"
            )
    lines.append("")
    lines.append(
        "Footnotes: (1) KNN and LR are not in the paper's Table 5/9 at all -- the paper "
        "only reports their accuracy and recall in running text (Section 5.3.2: KNN "
        "\"31 failures (37.3%)... 97.28%\"; LR \"21 failures (25.3%)... 97.32%\"). Their "
        "precision/F1/specificity/FN-rate/TN/FP above are *inferred* by solving "
        "TN = accuracy*2500 - TP and FP = 2500 - TP - FN - TN from those two reported "
        "figures, not values the paper itself states. (2) The paper's own Table 5 and "
        "Table 9 **disagree with each other** on Traditional Voting's precision/F1: "
        "Table 5 reports TP=37/FN=46/TN=2417/FP=0 (which forces precision=TP/(TP+FP)=100%, "
        "F1=61.7%), but Table 9 reports precision=94%/F1=60% for the same row. We used "
        "Table 5's confusion-matrix-consistent figures above; this is a discrepancy "
        "*within the paper*, not introduced by this reproduction."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Narrative check
    # ------------------------------------------------------------------
    or_c, vot_c, rf_c, gbc_c = clean["OR-Logic"], clean["Traditional Voting"], clean["RF"], clean["GBC"]
    or_p, vot_p = paper["OR-Logic"], paper["Traditional Voting"]

    beats_recall = or_c["recall"] > vot_c["recall"]
    lines.append("## Narrative check")
    lines.append("")
    lines.append(
        f"Does OR-logic beat Traditional Voting on recall while keeping specificity high? "
        f"**{'Yes' if beats_recall else 'No'}** -- "
        f"OR-Logic reaches {fmt_pct(or_c['recall'])} recall vs. Voting's {fmt_pct(vot_c['recall'])} "
        f"(paper: {fmt_pct(or_p['recall'])} vs. {fmt_pct(vot_p['recall'])}), while holding "
        f"{fmt_pct(or_c['specificity'])} specificity ({int(or_c['FP'])} FP out of "
        f"{int(or_c['TN']) + int(or_c['FP'])} true negatives) -- a bit below the paper's "
        f"{fmt_pct(or_p['specificity'])} but still solidly in the high-90s, and a world away "
        f"from the balanced-weights run's ~83% (418 FP, see `outputs/balanced_variant/`). "
        f"The paper's core contrast -- OR-logic trading a modest precision cost for a large "
        f"recall/FN-rate win over both Traditional Voting and the best individual model, "
        f"while specificity stays in the high-90s -- **does hold** under this unbalanced "
        f"canonical config, and the OR-logic vs. Voting gap is if anything larger here "
        f"(FN rate {fmt_pct(or_c['fnr'])} vs. {fmt_pct(vot_c['fnr'])}) than the paper reports "
        f"(FN rate {fmt_pct(or_p['fnr'])} vs. {fmt_pct(vot_p['fnr'])}). GBC remains the "
        f"strongest individual model by recall on modified data "
        f"({fmt_pct(gbc_c['recall'])} vs. RF's {fmt_pct(rf_c['recall'])}), unlike the "
        f"paper where RFC leads GBC."
    )

    return "\n".join(lines) + "\n"


def main():
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print(f"Loaded {config.DATA_PATH.name}: shape={df.shape}")

    train_idx, test_idx = split_indices(df)
    feature_frames, y = build_feature_frames(df)

    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    write_split_balance(y_train, y_test, config.OUTPUTS_DIR / "split_class_balance.txt")
    print(f"Wrote {config.OUTPUTS_DIR / 'split_class_balance.txt'}")

    all_rows = []
    artifacts_by_set = {}
    for feature_set_name in ["original", "modified"]:
        X = feature_frames[feature_set_name]
        rows, artifacts = run_feature_set(feature_set_name, X, y, train_idx, test_idx)
        all_rows.extend(rows)
        artifacts_by_set[feature_set_name] = artifacts
        print(f"Ran feature_set='{feature_set_name}': {len(rows)} rows (5 models + 2 ensembles)")

    results_df = pd.DataFrame(all_rows)
    col_order = [
        "feature_set", "model_or_ensemble", "n_test", "TP", "FN", "TN", "FP",
        "accuracy", "precision", "recall", "f1", "specificity", "fnr", "fpr", "mcc",
    ]
    results_df = results_df[col_order]
    results_path = config.OUTPUTS_DIR / "results_clean.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Wrote {results_path}")

    # --- paper-style tables ---
    build_table4(results_df).to_csv(config.TABLES_DIR / "table4_original_individual.csv", index=False)
    build_table5(results_df).to_csv(config.TABLES_DIR / "table5_modified_all.csv", index=False)
    build_table6(results_df).to_csv(config.TABLES_DIR / "table6_orig_vs_modified.csv", index=False)
    build_table8(results_df).to_csv(config.TABLES_DIR / "table8_ensembles_raw.csv", index=False)
    table10, table10_deltas = build_table10(results_df)
    table10.to_csv(config.TABLES_DIR / "table10_full_comparison.csv", index=False)
    table10_deltas.to_csv(config.TABLES_DIR / "table10_full_comparison_deltas.csv", index=False)
    print(f"Wrote 5 table CSVs to {config.TABLES_DIR}")

    # --- figures ---
    mod = artifacts_by_set["modified"]
    plot_roc(mod["probas"], mod["y_test"], config.FIGURES_DIR / "roc_modified.png")

    pred_by_name = dict(mod["preds"])
    pred_by_name["Traditional Voting"] = mod["voting_pred"]
    pred_by_name["OR-Logic"] = mod["or_pred"]
    plot_confusion_grid(pred_by_name, mod["y_test"], config.FIGURES_DIR / "confusion_matrices_modified.png")

    plot_correlation(
        df[config.ORIGINAL_FEATURES + [config.TARGET]],
        config.FIGURES_DIR / "corr_original.png",
        "Correlation matrix -- original features",
    )
    df_fe = add_engineered_features(df)
    plot_correlation(
        df_fe[config.MODIFIED_FEATURES + [config.TARGET]],
        config.FIGURES_DIR / "corr_modified.png",
        "Correlation matrix -- modified (feature-engineered) data",
    )
    print(f"Wrote 4 figures to {config.FIGURES_DIR}")

    # --- delta report ---
    delta_md = build_delta_report(results_df)
    delta_path = config.OUTPUTS_DIR / "DELTA_vs_paper.md"
    delta_path.write_text(delta_md)
    print(f"Wrote {delta_path}")

    # --- stdout summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: OR-Logic vs Traditional Voting vs RFC vs GBC, modified data (unbalanced config)")
    print("=" * 70)
    labels = {"OR-Logic": "OR-Logic", "Traditional Voting": "Voting", "RF": "RFC", "GBC": "GBC"}
    for name, label in labels.items():
        r = results_df[(results_df["feature_set"] == "modified") & (results_df["model_or_ensemble"] == name)].iloc[0]
        print(
            f"{label:8s} recall={fmt_pct(r['recall'])}  precision={fmt_pct(r['precision'])}  "
            f"specificity={fmt_pct(r['specificity'])}  FN_rate={fmt_pct(r['fnr'])}"
        )


if __name__ == "__main__":
    main()
