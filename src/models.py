"""Factory for the five base-model Pipelines (scaler + classifier).

Every model is wrapped in Pipeline([('scaler', StandardScaler()), ('clf', ...)])
so that scaling is always fit on the training fold only -- no leakage. All
hyperparameters come from src/config.py; nothing here is tuned to match the
paper.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from . import config


def _pipeline(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_models() -> dict:
    """Returns an ordered dict of name -> unfit Pipeline. Order matches the
    paper's base-classifier ordering (SVM, RF, KNN, LR, GBC) so ensemble
    "sum of predictions" is reproducible/inspectable in that order."""
    return {
        "SVM": _pipeline(SVC(**config.SVM_PARAMS)),
        "RF": _pipeline(RandomForestClassifier(**config.RF_PARAMS)),
        "KNN": _pipeline(KNeighborsClassifier(**config.KNN_PARAMS)),
        "LR": _pipeline(LogisticRegression(**config.LR_PARAMS)),
        "GBC": _pipeline(GradientBoostingClassifier(**config.GBC_PARAMS)),
    }
