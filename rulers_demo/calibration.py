from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import ASAP2_TRAITS
from .utils import clamp_int


FEATURE_COLUMNS_BASE = [
    "raw_score_cont",
    "confidence",
    "invalid_evidence",
    "ev_miss",
    "raw_quote_total",
    "valid_quote_total",
    "tie",
    "trait_range",
]


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for trait in ASAP2_TRAITS:
        c = f"used_{trait}"
        if c in df.columns:
            cols.append(c)
    for c in FEATURE_COLUMNS_BASE:
        if c in df.columns:
            cols.append(c)
    return cols


def rows_to_X(df: pd.DataFrame) -> np.ndarray:
    cols = feature_columns(df)
    if not cols:
        raise ValueError("No calibration feature columns found.")
    return df[cols].fillna(0.0).astype(float).values


def fit_wgr_calibration(
    calib_df: pd.DataFrame,
    score_min: int = 1,
    score_max: int = 6,
    alpha: float = 2.5,
) -> Dict[str, object]:
    if "human_score" not in calib_df.columns:
        raise KeyError("calib_df must include human_score.")
    fit_df = calib_df.dropna(subset=["human_score"]).copy()
    if len(fit_df) < 5:
        raise ValueError("Need at least 5 labeled calibration rows for WGR calibration.")

    X = rows_to_X(fit_df)
    y = fit_df["human_score"].astype(float).values
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(X, y)
    z = model.predict(X)

    order_z = np.argsort(z)
    z_sorted = z[order_z]
    y_sorted = np.sort(y)
    # Duplicate x values can break interpolation; add tiny monotone jitter only if needed.
    z_unique = z_sorted.astype(float).copy()
    for i in range(1, len(z_unique)):
        if z_unique[i] <= z_unique[i - 1]:
            z_unique[i] = z_unique[i - 1] + 1e-8

    mapper = interpolate.interp1d(
        z_unique,
        y_sorted,
        kind="linear",
        fill_value=(float(y_sorted[0]), float(y_sorted[-1])),
        bounds_error=False,
        assume_sorted=True,
    )
    return {
        "model": model,
        "mapper": mapper,
        "feature_columns": feature_columns(fit_df),
        "score_min": score_min,
        "score_max": score_max,
        "alpha": alpha,
    }


def predict_wgr_cont(df: pd.DataFrame, calib: Dict[str, object]) -> np.ndarray:
    cols = calib["feature_columns"]
    X = df[list(cols)].fillna(0.0).astype(float).values
    z = calib["model"].predict(X)
    pred = calib["mapper"](z)
    return np.asarray(pred, dtype=float)


def predict_wgr_int(df: pd.DataFrame, calib: Dict[str, object]) -> np.ndarray:
    cont = predict_wgr_cont(df, calib)
    lo = int(calib["score_min"])
    hi = int(calib["score_max"])
    return np.array([clamp_int(v, lo, hi) for v in cont], dtype=int)


def attach_final_scores(
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    enable_calibration: bool = True,
    score_min: int = 1,
    score_max: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object] | None]:
    calib_df = calib_df.copy()
    test_df = test_df.copy()
    if enable_calibration:
        calib = fit_wgr_calibration(calib_df, score_min=score_min, score_max=score_max)
        calib_df["final_score_cont"] = predict_wgr_cont(calib_df, calib)
        calib_df["final_score"] = predict_wgr_int(calib_df, calib)
        test_df["final_score_cont"] = predict_wgr_cont(test_df, calib)
        test_df["final_score"] = predict_wgr_int(test_df, calib)
        return calib_df, test_df, calib

    calib_df["final_score_cont"] = calib_df["raw_score_cont"].astype(float)
    calib_df["final_score"] = calib_df["raw_score"].astype(int)
    test_df["final_score_cont"] = test_df["raw_score_cont"].astype(float)
    test_df["final_score"] = test_df["raw_score"].astype(int)
    return calib_df, test_df, None
