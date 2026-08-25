from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    return {
        "QWK": qwk,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": float(s_true.corr(s_pred, method="pearson")),
        "Spearman": float(s_true.corr(s_pred, method="spearman")),
    }


def summarize_cost(rows: list[dict]) -> Dict[str, float]:
    if not rows:
        return {
            "avg_calls": 0.0,
            "avg_tokens": 0.0,
            "avg_latency_sec": 0.0,
            "total_calls": 0.0,
            "total_tokens": 0.0,
        }
    calls = np.array([float(r.get("total_calls", 0.0)) for r in rows], dtype=float)
    toks = np.array([float(r.get("total_tokens", 0.0)) for r in rows], dtype=float)
    lat = np.array([float(r.get("latency_sec", 0.0)) for r in rows], dtype=float)
    return {
        "avg_calls": float(calls.mean()),
        "avg_tokens": float(toks.mean()),
        "avg_latency_sec": float(lat.mean()),
        "total_calls": float(calls.sum()),
        "total_tokens": float(toks.sum()),
    }
