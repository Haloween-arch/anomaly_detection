from __future__ import annotations

from typing import Dict, List

import numpy as np


def explain_anomalies(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: Dict,
    max_k: int = 7,
    min_percent: float = 1.0,
) -> List[List[str]]:
    """
    Feature attribution using per-feature absolute standardized deviation.
    (X is already standardized by training-fitted scaler.)

    Returns: list of lists of feature names for each row (top_k; ≥ min_percent).
    Ties broken alphabetically.
    """
    feats = meta["feature_cols"]
    attributions: List[List[str]] = []

    for i in range(X.shape[0]):
        dev = np.abs(X[i])
        total = float(dev.sum()) + 1e-12
        pct = (dev / total) * 100.0
        idx = np.where(pct >= min_percent)[0]
        order = sorted(idx, key=lambda j: (-pct[j], feats[j]))
        top_idx = order[:max_k]
        top_feats = [feats[j] for j in top_idx]
        attributions.append(top_feats)

    return attributions
