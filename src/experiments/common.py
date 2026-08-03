"""
Shared plumbing for src/experiments/*.

This module REUSES the locked canonical pipeline (src/config.py, src/data.py,
src/features.py, src/models.py, src/evaluate.py) as-is. It does not redefine
any model, hyperparameter, feature list, or the locked SEED=42 split used by
src/run_baseline.py. Where an experiment needs a *different* split (e.g. the
stability analysis resamples over 30 seeds), that variation happens locally
in the experiment script using sklearn's train_test_split directly with the
same test_size/stratify contract -- src/config.py itself is never touched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_raw, split_indices, build_feature_frames, train_test_frames
from src.models import build_models
from src.evaluate import positive_class_metrics, ensemble_from_predictions

BASE_MODEL_NAMES = ["SVM", "RF", "KNN", "LR", "GBC"]
EXPERIMENTS_DIR = config.OUTPUTS_DIR / "experiments"


def load_modified_xy():
    """Returns (X_modified, y, raw_df) at full length, unsplit."""
    df = load_raw()
    feature_frames, y = build_feature_frames(df)
    return feature_frames["modified"], y, df


def locked_seed42_split():
    """The exact same modified-data, SEED=42, stratified split that
    src/run_baseline.py uses -- reused verbatim, not recomputed differently."""
    df = load_raw()
    feature_frames, y = build_feature_frames(df)
    train_idx, test_idx = split_indices(df)
    return train_test_frames(feature_frames["modified"], y, train_idx, test_idx)


def resample_split(X, y, seed, test_size=None):
    """A stratified split with test_size fixed at the locked config's value
    but random_state=seed -- used ONLY by the stability/CI experiment (task 1)
    to resample across 30 seeds. Does not modify or read config.SEED."""
    test_size = config.TEST_SIZE if test_size is None else test_size
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def fit_predict_all(X_train, y_train, X_test, models=None):
    """Fits the five locked-config model Pipelines (via build_models(), which
    reads all hyperparameters from src/config.py unchanged) and returns
    (preds, probas, models)."""
    models = models if models is not None else build_models()
    preds, probas = {}, {}
    for name in BASE_MODEL_NAMES:
        pipe = models[name]
        pipe.fit(X_train, y_train)
        preds[name] = pipe.predict(X_test)
        probas[name] = pipe.predict_proba(X_test)[:, 1]
    return preds, probas, models


def ensembles_from_preds(preds, index):
    """Applies the locked ensemble thresholds (config.VOTING_THRESHOLD=3,
    config.OR_LOGIC_THRESHOLD=1) to a dict of per-model 0/1 predictions."""
    pred_df = pd.DataFrame(preds, index=index)[BASE_MODEL_NAMES]
    voting = ensemble_from_predictions(pred_df, config.VOTING_THRESHOLD)
    or_logic = ensemble_from_predictions(pred_df, config.OR_LOGIC_THRESHOLD)
    return voting, or_logic


def all_rows_for_split(X_train, y_train, X_test, y_test, models=None):
    """Fits all 5 locked models + derives both locked ensembles on one split;
    returns a list of positive_class_metrics() dicts (7 rows) plus the raw
    preds/probas for downstream use (SHAP, latency, etc.)."""
    preds, probas, fitted_models = fit_predict_all(X_train, y_train, X_test, models=models)
    rows = [positive_class_metrics(y_test, preds[name], name, "modified") for name in BASE_MODEL_NAMES]

    voting_pred, or_pred = ensembles_from_preds(preds, X_test.index)
    rows.append(positive_class_metrics(y_test, voting_pred, "Traditional Voting", "modified"))
    rows.append(positive_class_metrics(y_test, or_pred, "OR-Logic", "modified"))

    return rows, {"preds": preds, "probas": probas, "models": fitted_models, "voting_pred": voting_pred, "or_pred": or_pred}
