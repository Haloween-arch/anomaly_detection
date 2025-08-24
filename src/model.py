from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def train_models(X_train: np.ndarray) -> Dict[str, object]:
    """
    Train lightweight hybrid models:
      - IsolationForest: global/point anomalies
      - PCA: reconstruction error → correlation/relationship shifts
    """
    iso = IsolationForest(
        n_estimators=200, max_samples="auto", contamination="auto", random_state=42, n_jobs=-1
    )
    iso.fit(X_train)

    # Keep enough PCs to explain 95% variance
    pca = PCA(n_components=min(X_train.shape) - 1, svd_solver="full", random_state=42)
    pca.fit(X_train)
    return {"iso": iso, "pca": pca}
