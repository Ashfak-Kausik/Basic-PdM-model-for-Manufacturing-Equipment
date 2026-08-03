"""Data loading, feature-set construction, and the (single, shared) stratified split.

The train/test row split is computed ONCE, from the target column, and then
applied identically to both the original and modified feature matrices --
this guarantees the two feature sets share the exact same train/test rows,
which the five-model ensembles below rely on (predictions are stacked
column-wise per row).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .features import add_engineered_features


def load_raw() -> pd.DataFrame:
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_PATH}. "
            "Place the AI4I 2020 Predictive Maintenance dataset (ai4i2020.csv) there and re-run."
        )
    return pd.read_csv(config.DATA_PATH)


def split_indices(df: pd.DataFrame):
    """The one and only train_test_split call: stratified on the target,
    seeded by config.SEED. Returns (train_idx, test_idx) as pandas Index
    objects aligned to df.index."""
    y = df[config.TARGET]
    train_idx, test_idx = train_test_split(
        df.index, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    return train_idx, test_idx


def build_feature_frames(df: pd.DataFrame):
    """Returns {'original': X_original, 'modified': X_modified}, y -- all
    still at full length (not yet split); splitting happens via split_indices()."""
    y = df[config.TARGET]

    X_original = df[config.ORIGINAL_FEATURES]

    df_fe = add_engineered_features(df)
    X_modified = df_fe[config.MODIFIED_FEATURES]

    return {"original": X_original, "modified": X_modified}, y


def train_test_frames(X: pd.DataFrame, y: pd.Series, train_idx, test_idx):
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def write_split_balance(y_train: pd.Series, y_test: pd.Series, out_path):
    n_train, n_test = len(y_train), len(y_test)
    train_fail, test_fail = int(y_train.sum()), int(y_test.sum())
    lines = [
        "Stratified train/test split -- class balance",
        "=" * 45,
        f"Seed: {config.SEED}  |  test_size: {config.TEST_SIZE}  |  stratify: y ({config.TARGET})",
        "",
        f"Train rows: {n_train}",
        f"  Machine failure = 1 (failure):     {train_fail:5d}  ({100 * train_fail / n_train:.3f}%)",
        f"  Machine failure = 0 (non-failure): {n_train - train_fail:5d}  ({100 * (n_train - train_fail) / n_train:.3f}%)",
        "",
        f"Test rows: {n_test}",
        f"  Machine failure = 1 (failure):     {test_fail:5d}  ({100 * test_fail / n_test:.3f}%)",
        f"  Machine failure = 0 (non-failure): {n_test - test_fail:5d}  ({100 * (n_test - test_fail) / n_test:.3f}%)",
        "",
        f"Total rows: {n_train + n_test}  |  Total failures: {train_fail + test_fail} "
        f"({100 * (train_fail + test_fail) / (n_train + n_test):.3f}%)",
    ]
    out_path.write_text("\n".join(lines) + "\n")
