"""Engineered feature formulas for the clean pipeline (§ config.ENGINEERED_FEATURES).

Formulas match ML main.ipynb cell 28 exactly; the difference from that
notebook is that this function returns a NEW dataframe (via .copy()) instead
of mutating its input in place.
"""
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with RelationTemperature, Power (W), WearRPM, and
    ToolWearTorque added."""
    out = df.copy()
    out["RelationTemperature"] = out["Air temperature [K]"] - out["Process temperature [K]"]
    out["Power (W)"] = (out["Torque [Nm]"] * out["Rotational speed [rpm]"]) / 9.5488
    out["WearRPM"] = out["Tool wear [min]"] / out["Rotational speed [rpm]"]
    out["ToolWearTorque"] = out["Tool wear [min]"] / out["Torque [Nm]"]
    return out
