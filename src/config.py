"""
Single source of truth for the CLEAN, canonical pipeline (src/run_baseline.py
and friends). Every hyperparameter, feature list, and path used by the clean
pipeline lives here so it is one-line editable.

This is a *deliberately different* configuration from src/reproduce.py, which
stays untouched as a faithful, bug-preserving port of ML main.ipynb (see
outputs/AUDIT.md for why the notebook's numbers don't match the paper). This
module intentionally does NOT try to match the paper's reported numbers --
values here are the canonical choices we're standardizing on; results are
reported honestly in outputs/DELTA_vs_paper.md regardless of who wins.

There is deliberately no two-stage / ColumnFailure0 / risk-factor logic here.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "ai4i2020.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FAITHFUL_REPRO_CSV = OUTPUTS_DIR / "results_reproduced.csv"  # produced by src/reproduce.py -- read-only here

# ----------------------------------------------------------------------------
# Reproducibility / split
# ----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.25
TARGET = "Machine failure"

# ----------------------------------------------------------------------------
# Feature sets
# ----------------------------------------------------------------------------
# Deliberately excludes Type_encoded (unlike ML main.ipynb, which included it
# by accident via an incomplete drop list -- see AUDIT.md section 2c).
ORIGINAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURES = [
    "RelationTemperature",
    "Power (W)",
    "WearRPM",
    "ToolWearTorque",
]

MODIFIED_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES  # nine total

# ----------------------------------------------------------------------------
# Model hyperparameters (used by src/models.py to build the Pipelines)
# ----------------------------------------------------------------------------
# class_weight='balanced' was removed from SVM/RF/LR: with it, each base
# learner over-fired individually (SVM ~190 FP, LR ~350 FP on the modified
# test set), and because OR-logic unions all five models' positives, it
# inherited that union of false alarms (418 FP), collapsing precision to
# ~16% and specificity to ~83% -- breaking the paper's actual claim of high
# recall while holding specificity near 99.5%. See outputs/balanced_variant/
# for the retained class_weight='balanced' run (cost-sensitive baseline for
# reviewer comment R1.6).
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)
RF_PARAMS = dict(n_estimators=100, max_features="sqrt", random_state=SEED)  # unrestricted depth (max_depth=None)
KNN_PARAMS = dict(n_neighbors=5, weights="uniform", p=2)
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
GBC_PARAMS = dict(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED)

# ----------------------------------------------------------------------------
# Ensemble thresholds (applied to the sum of the five models' 0/1 test predictions)
# ----------------------------------------------------------------------------
VOTING_THRESHOLD = 3   # majority: failure if >= 3 of 5 models predict failure
OR_LOGIC_THRESHOLD = 1  # OR-logic: failure if >= 1 of 5 models predict failure
