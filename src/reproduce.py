"""
Faithful reproduction of the pipeline in ML main.ipynb.

This script is a line-for-line port of the notebook's code cells into a
runnable script. It intentionally preserves every methodological choice
found in the notebook, INCLUDING what look like bugs (see the "BUG"
comments below) -- the point of this script is to show exactly what the
current code does, not what it should do. Do not "fix" anything here
without checking with the project owner first; see outputs/AUDIT.md for
the list of issues this reproduction surfaced.

Cell references (e.g. "cell 16") refer to the code-cell index (0-based)
in ML main.ipynb, counting only actual cells in file order, so the audit
trail can be cross-checked against the notebook directly.

Usage:
    .venv/bin/python src/reproduce.py

Requires data/ai4i2020.csv (AI4I 2020 Predictive Maintenance dataset).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "ai4i2020.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def positive_class_metrics(y_true, y_pred, model_name, feature_set, notes=""):
    """Full confusion matrix + accuracy/precision/recall/F1/specificity/FNR
    for the POSITIVE (failure=1) class specifically -- not weighted/macro
    averages. This is what the task requires for results_reproduced.csv,
    and what the paper's Tables 4/5/8/9 actually report (e.g. "recall 88.0%"
    is TP/(TP+FN) for class 1, not a weighted average)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # = TPR = sensitivity
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # TNR
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")  # false negative rate = 1 - recall
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "feature_set": feature_set,
        "model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy,
        "precision_pos": precision,
        "recall_pos_TPR": recall,
        "f1_pos": f1,
        "specificity_TNR": specificity,
        "false_negative_rate": fnr,
        "mcc": mcc,
        "n_test": int(tn + fp + fn + tp),
        "notes": notes,
    }


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        print("Place the AI4I 2020 Predictive Maintenance dataset (ai4i2020.csv) there and re-run.")
        sys.exit(1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # ------------------------------------------------------------------
    # Cell 1: raw_data = pd.read_csv('ai4i2020.csv')
    # ------------------------------------------------------------------
    raw_data = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.name}: shape={raw_data.shape}")

    # ------------------------------------------------------------------
    # Cell 13: LabelEncoder on 'Type' -> 'Type_encoded'
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder()
    raw_data["Type_encoded"] = label_encoder.fit_transform(raw_data["Type"])

    # ==================================================================
    # PASS 1 (cells 14-21): baseline models on the ORIGINAL features,
    # before feature engineering. Corresponds to paper Table 4.
    # NOTE: the notebook never builds a results_df, VotingClassifier, or
    # OR-logic ensemble for this pass -- only individual model accuracy is
    # printed (cells 17-21). There is therefore no code path in the
    # notebook producing paper Table 8 (ensemble performance on raw data).
    # ==================================================================
    X_raw = raw_data.drop(
        columns=["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )
    y_raw = raw_data["Machine failure"]

    # Cell 16: train_test_split(test_size=0.25, random_state=25) -- NO stratify, NO scaling
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=25
    )

    # Cell 17: SVM
    m1_raw = svm.SVC(C=10, degree=2)
    m1_raw.fit(X_train_raw, y_train_raw)
    results.append(
        positive_class_metrics(y_test_raw, m1_raw.predict(X_test_raw), "SVM", "original")
    )

    # Cell 18: Random Forest
    m2_raw = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    m2_raw.fit(X_train_raw, y_train_raw)
    results.append(
        positive_class_metrics(y_test_raw, m2_raw.predict(X_test_raw), "RF", "original")
    )

    # Cell 19: KNN
    m3_raw = KNeighborsClassifier(n_neighbors=10)
    m3_raw.fit(X_train_raw, y_train_raw)
    results.append(
        positive_class_metrics(y_test_raw, m3_raw.predict(X_test_raw), "KNN", "original")
    )

    # Cell 20: Logistic Regression
    m4_raw = LogisticRegression()
    m4_raw.fit(X_train_raw, y_train_raw)
    results.append(
        positive_class_metrics(y_test_raw, m4_raw.predict(X_test_raw), "LR", "original")
    )

    # Cell 21: Gradient Boosting
    m5_raw = GradientBoostingClassifier()
    m5_raw.fit(X_train_raw, y_train_raw)
    results.append(
        positive_class_metrics(y_test_raw, m5_raw.predict(X_test_raw), "GBM", "original")
    )

    print("Pass 1 (original features, cells 14-21) complete.")

    # ==================================================================
    # PASS 2 (cells 27-61): feature engineering + modified-data models
    # + both ensembles. Corresponds to paper Tables 5/6/7/9.
    # ==================================================================

    # Cell 27: f_data = raw_data  (NOTE: this is a reference, not a copy --
    # the notebook mutates raw_data in place from here on. We mirror that
    # exactly by continuing to operate on `raw_data`/`f_data` as the same
    # object, since it has no effect on results -- raw_data is not reused.)
    f_data = raw_data

    # Cell 28: engineered features
    f_data["RelationTemperature"] = f_data["Air temperature [K]"] - f_data["Process temperature [K]"]
    f_data["Power (W)"] = (f_data["Torque [Nm]"] * f_data["Rotational speed [rpm]"]) / 9.5488
    f_data["WearRPM"] = f_data["Tool wear [min]"] / f_data["Rotational speed [rpm]"]
    f_data["ToolWearTorque"] = f_data["Tool wear [min]"] / f_data["Torque [Nm]"]

    # Cell 32: X/y for the modified pipeline.
    # NOTE: this drop list does NOT drop 'Type_encoded', so X ends up with
    # 10 feature columns (5 original + Type_encoded + 4 engineered), not the
    # "nine features" (5 original + 4 engineered) the paper's Section 5.3.2
    # describes. See AUDIT.md.
    X = f_data.drop(columns=["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
    y = f_data["Machine failure"]
    print(f"Modified feature matrix columns ({len(X.columns)}): {list(X.columns)}")

    # Cell 34: train_test_split(test_size=0.25, random_state=25) -- same
    # params as cell 16. Because train_test_split's row shuffling depends
    # only on n_samples and random_state (not on column content), this
    # produces the SAME train/test row split as Pass 1.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    results_df = pd.DataFrame({"Real": y_test})

    # Cell 35: SVM
    model1 = svm.SVC(C=10, degree=2)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    results_df["SVM"] = y_pred
    results.append(positive_class_metrics(y_test, y_pred, "SVM", "modified"))

    # Cell 39: Random Forest
    model2 = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    results_df["RF"] = y_pred
    results.append(positive_class_metrics(y_test, y_pred, "RF", "modified"))

    # Cell 43: KNN
    model3 = KNeighborsClassifier(n_neighbors=10)
    model3.fit(X_train, y_train)
    y_pred = model3.predict(X_test)
    results_df["KNN"] = y_pred
    results.append(positive_class_metrics(y_test, y_pred, "KNN", "modified"))

    # Cell 47: Logistic Regression
    model4 = LogisticRegression()
    model4.fit(X_train, y_train)
    y_pred = model4.predict(X_test)
    results_df["LR"] = y_pred
    results.append(positive_class_metrics(y_test, y_pred, "LR", "modified"))

    # Cell 51: Gradient Boosting
    model5 = GradientBoostingClassifier()
    model5.fit(X_train, y_train)
    y_pred = model5.predict(X_test)
    results_df["GBM"] = y_pred
    results.append(positive_class_metrics(y_test, y_pred, "GBM", "modified"))

    print("Pass 2 individual models (cells 35-54) complete.")

    # Cell 55: hard VotingClassifier ("Traditional Voting" ensemble).
    # sklearn clones + refits the 5 estimators internally with the same
    # hyperparameters; majority (>=3/5) vote decides the class.
    ensemble_clf = VotingClassifier(
        estimators=[
            ("svm", model1),
            ("random_forest", model2),
            ("knn", model3),
            ("logistic_regression", model4),
            ("gradient_boosting", model5),
        ],
        voting="hard",
    )
    ensemble_clf.fit(X_train, y_train)
    y_pred = ensemble_clf.predict(X_test)  # this is the notebook's cell-55 `y_pred`
    results.append(positive_class_metrics(y_test, y_pred, "Traditional Voting (VotingClassifier)", "modified"))

    # Cell 58: OR-logic ensemble = sum of the 5 model prediction columns > 0
    sum_excluding_real = results_df.drop("Real", axis=1).sum(axis=1)
    results_df["sum_excluding_real"] = (sum_excluding_real > 0).astype(int)
    results.append(
        positive_class_metrics(
            results_df["Real"], results_df["sum_excluding_real"], "OR-Logic (sum of 5 models > 0)", "modified"
        )
    )

    # Cell 61: y_test = results_df['Real']; y_pred = results_df['sum_excluding_real']
    # This reassignment of `y_pred` in the notebook is important: it is the
    # LAST assignment to `y_pred` before cell 78 (`results_df['E_col'] =
    # y_pred`). Nothing between here and cell 78 reassigns y_pred again
    # (cell 77's train_test_split unpacks into X_train, X_pred, y_train,
    # y_test -- note the 2nd position is named X_pred, not X_test, so
    # y_pred itself is untouched). We mirror that by holding onto this
    # value under the same name.
    y_test = results_df["Real"]
    y_pred = results_df["sum_excluding_real"]

    print("Pass 2 ensembles (cells 55-61) complete.")

    # ==================================================================
    # "2nd Phase" (cells 62-82): risk factor + ColumnFailure0 threshold
    # logic, and the second train/test split. See AUDIT.md for the
    # leakage and dead-code findings this section produces.
    # ==================================================================

    # Cells 62-63: hardcoded, manually-normalized feature weights (not
    # derived programmatically anywhere in the notebook -- these are
    # magic numbers typed directly into the cell).
    air_temp = 0.083
    process_temp = 0.068
    speed = 0.025
    torque = 0.19
    wear = 0.11
    r_temp = 0.062
    power = 0.19
    wear_rpm = 0.13
    tw_torque = 0.082
    total_sum = air_temp + process_temp + speed + torque + wear + r_temp + power + wear_rpm + tw_torque
    air_temp_n = air_temp / total_sum
    process_temp_n = process_temp / total_sum
    speed_n = speed / total_sum
    torque_n = torque / total_sum
    wear_n = wear / total_sum
    r_temp_n = r_temp / total_sum
    power_n = power / total_sum
    wear_rpm_n = wear_rpm / total_sum
    tw_torque_n = tw_torque / total_sum

    # Cell 64: risk factor, computed for EVERY row of f_data (train+test).
    f_data["risk factor"] = (
        f_data["Air temperature [K]"] * air_temp_n
        + f_data["Process temperature [K]"] * process_temp_n
        - f_data["Rotational speed [rpm]"] * speed_n
        + f_data["Torque [Nm]"] * torque_n
        + f_data["Tool wear [min]"] * wear_n
        + f_data["RelationTemperature"] * r_temp_n
        + f_data["Power (W)"] * power_n
        + f_data["WearRPM"] * wear_rpm_n
        + f_data["ToolWearTorque"] * tw_torque_n
    )

    # Cell 74: min/max envelope of 7 columns, computed on
    # f_data[f_data['Machine failure'] == 0] -- i.e. over the FULL dataset
    # (train rows AND test rows together), not the training split alone.
    # LEAKAGE: this envelope is later used (cell 76) to derive a label
    # (ColumnFailure0) that depends on knowing, for every test-set row,
    # whether that row's values exceed bounds computed using other
    # test-set rows' true failure status. See AUDIT.md item (c).
    failure_0_data = f_data[f_data["Machine failure"] == 0]  # NOT restricted to X_train/y_train rows
    envelope_cols = [
        "ToolWearTorque",
        "WearRPM",
        "Power (W)",
        "Torque [Nm]",
        "Tool wear [min]",
        "Rotational speed [rpm]",
        "risk factor",
    ]
    max_values_failure_0 = {c: failure_0_data[c].max() for c in envelope_cols}
    min_values_failure_0 = {c: failure_0_data[c].min() for c in envelope_cols}

    f_data["ColumnFailure0"] = 0
    for column, max_value in max_values_failure_0.items():
        f_data.loc[f_data[column] > max_value, "ColumnFailure0"] = 1
    for column, min_value in min_values_failure_0.items():
        f_data.loc[f_data[column] < min_value, "ColumnFailure0"] = 1

    n_flagged = int(f_data["ColumnFailure0"].sum())
    print(f"ColumnFailure0 flagged {n_flagged}/{len(f_data)} rows (full-dataset envelope, cell 74).")

    # Cell 76: X2 built from the SAME drop list as cell 32 -- this list does
    # NOT drop 'risk factor' or 'ColumnFailure0', so X2 contains
    # 'ColumnFailure0' as a feature column *while y2 is also
    # 'ColumnFailure0'* (direct target leakage into X2), plus 'risk factor'
    # (itself leakage-derived). This X2/y2 pair is never actually used to
    # train anything (see cell 77 note below), so this leakage is currently
    # inert, but it is a landmine if anyone "completes" this stage.
    X2 = f_data.drop(columns=["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
    y2 = f_data["ColumnFailure0"]

    # Cell 77: X_train, X_pred, y_train, y_test = train_test_split(X2, y2, ...)
    # BUG: the notebook never calls .fit()/.predict() on any model using
    # this split. X_train/X_pred/y_train are dead values, and this line's
    # y_test overwrites the y_test name -- but nothing downstream reads
    # y_test again either, so this whole split is orphaned/unused.
    X2_train, X2_pred, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=25)
    # (Intentionally not fit/predicted on -- preserving the notebook's actual behavior.)

    # Cell 78: results_df['E_col'] = y_pred
    # BUG: `y_pred` here is NOT a new model's prediction on ColumnFailure0.
    # It is still the stale value from cell 61: results_df['sum_excluding_real']
    # (the stage-1 OR-logic ensemble prediction on the *original*
    # Machine-failure target). So E_col is a verbatim copy of
    # sum_excluding_real, not an independent "stage 2" model output.
    results_df["E_col"] = y_pred

    # Cell 80: sum_excluding_real2 = sum(SVM, RF, KNN, LR, GBM, E_col) > 0
    sum_excluding_real_2 = results_df.drop(columns=["Real", "sum_excluding_real"], axis=1).sum(axis=1)
    results_df["sum_excluding_real2"] = (sum_excluding_real_2 > 0).astype(int)

    # Cell 82: accuracy_score(results_df['Real'], results_df['sum_excluding_real2'])
    results.append(
        positive_class_metrics(
            results_df["Real"],
            results_df["sum_excluding_real2"],
            "Stage-2 'ColumnFailure0' ensemble (sum_excluding_real2)",
            "modified",
            notes=(
                "BUG: E_col is a stale copy of the stage-1 OR-logic column, not a model trained on "
                "ColumnFailure0. Because E_col == sum_excluding_real for every row, "
                "sum_excluding_real2>0 is mathematically identical to sum_excluding_real>0 -- "
                "these predictions are IDENTICAL to the OR-Logic row above, so this row's numbers "
                "are not evidence of any distinct 'stage two' model."
            ),
        )
    )

    identical_to_stage1 = (results_df["sum_excluding_real2"] == results_df["sum_excluding_real"]).all()
    print(f"sum_excluding_real2 identical to sum_excluding_real for every row: {identical_to_stage1}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    results_df_out = pd.DataFrame(results)
    col_order = [
        "feature_set",
        "model",
        "n_test",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision_pos",
        "recall_pos_TPR",
        "f1_pos",
        "specificity_TNR",
        "false_negative_rate",
        "mcc",
        "notes",
    ]
    results_df_out = results_df_out[col_order]
    out_path = OUTPUTS_DIR / "results_reproduced.csv"
    results_df_out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(results_df_out.to_string(index=False))


if __name__ == "__main__":
    main()
