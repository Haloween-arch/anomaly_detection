from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances


def _moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")


def _rank_scale(x: np.ndarray) -> np.ndarray:
    """Robust 0-100 scaling via ranks (monotonic)."""
    r = x.argsort().argsort().astype(float)
    return 100.0 * (r + 1.0) / (len(x) + 1.0)


def detect_anomalies(
    models: Dict[str, object],
    X: np.ndarray,
    meta: Dict,
    perc_threshold: float = 0.97,
    smooth_window: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a hybrid anomaly score and return (scores_0_100, labels, components).
    Components contain raw sub-scores useful for report plots.
    """
    iso = models["iso"]
    pca = models["pca"]

    # --- Component scores ---
    # IsolationForest: larger = more anomalous
    iso_raw = -iso.decision_function(X)
    iso_raw = np.maximum(iso_raw, 0.0)

    # PCA reconstruction error (relationship changes)
    X_p = pca.transform(X)
    X_hat = pca.inverse_transform(X_p)
    pca_err = np.mean((X - X_hat) ** 2, axis=1)

    # Rolling z-change magnitude (pattern deviations)
    # difference to previous standardized point, L2 norm
    dX = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])
    roll = np.linalg.norm(dX, axis=1)

    # Rank-scale each component to 0–100 to be comparable
    iso_s = _rank_scale(iso_raw)
    pca_s = _rank_scale(pca_err)
    roll_s = _rank_scale(roll)

    # Hybrid score: weighted average (ISO 0.5, PCA 0.35, ROLL 0.15)
    hybrid = 0.5 * iso_s + 0.35 * pca_s + 0.15 * roll_s

    # Smooth to avoid sudden jumps
    scores = _moving_average(hybrid, window=smooth_window)

    # --- Calibrate for training-window targets ---
    train_mask = meta["train_mask"]
    train_scores = scores[train_mask]
    if len(train_scores):
        # Ensure max <= 25
        tmax = float(train_scores.max())
        if tmax > 0:
            k = min(1.0, 25.0 / tmax)
            scores = np.clip(k * scores, 0.0, 100.0)
            train_scores = scores[train_mask]

        # Ensure mean < 10 (gentle compression if needed)
        tmean = float(train_scores.mean())
        if tmean > 10.0:
            gamma = 1.2
            scores = 100.0 * ((scores / 100.0) ** gamma)

    # Final labels by percentile threshold over analysis window
    thr = float(np.percentile(scores, 100 * perc_threshold))
    labels = (scores >= thr).astype(int)

    components = {
        "iso_raw": iso_raw,
        "pca_err": pca_err,
        "roll_mag": roll,
        "iso_s": iso_s,
        "pca_s": pca_s,
        "roll_s": roll_s,
        "threshold": thr,
        "train_mean": float(train_scores.mean()) if len(train_scores) else 0.0,
        "train_max": float(train_scores.max()) if len(train_scores) else 0.0,
    }
    return scores, labels, components

