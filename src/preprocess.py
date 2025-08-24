from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PrepError(RuntimeError):
    """Raised for data-quality issues with actionable messages."""


def _infer_regular_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """If irregular, infer median step and return regular grid index."""
    if index.inferred_freq is not None:
        return index
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        raise PrepError("Cannot infer time frequency (need ≥2 timestamps).")
    step = diffs.median()
    return pd.date_range(index.min(), index.max(), freq=step)


def _pick_timestamp_column(df: pd.DataFrame, provided: str) -> str:
    if provided != "auto":
        if provided not in df.columns:
            raise PrepError(f"Configured timestamp column '{provided}' not found.")
        return provided
    for cand in ("timestamp", "Timestamp", "time", "Time", "datetime", "DateTime"):
        if cand in df.columns:
            return cand
    raise PrepError("Missing timestamp column (tried: timestamp/Time/datetime/etc.).")


def load_and_prepare(
    csv_path: str,
    timestamp_col: str,
    normal_start: str,
    normal_end: str,
    analysis_start: str,
    analysis_end: str,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Load CSV, validate timestamps, clean data, and scale features.

    Returns:
      df (DataFrame): original columns restricted to analysis window
      X_full (ndarray): scaled numeric features for analysis window
      meta (dict): feature names, scaler, timestamps, masks, X_train
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise PrepError(f"Failed to read CSV: {e}")

    ts_col = _pick_timestamp_column(df, timestamp_col)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        raise PrepError("Invalid timestamps detected; please correct the source file.")

    df = df.sort_values(ts_col).reset_index(drop=True)

    # Restrict to analysis window
    mask_analysis = (df[ts_col] >= pd.Timestamp(analysis_start)) & (
        df[ts_col] <= pd.Timestamp(analysis_end)
    )
    df = df.loc[mask_analysis].copy()
    if df.empty:
        raise PrepError("No rows found in the requested analysis window.")

    # Enforce regular interval grid (reindex + forward fill)
    idx = pd.DatetimeIndex(df[ts_col])
    new_idx = _infer_regular_grid(idx)
    if not idx.equals(new_idx):
        logging.warning(
            "Irregular timestamps detected → reindexed to regular grid with forward-fill."
        )
        df = (
            df.set_index(ts_col)
            .reindex(new_idx)
            .ffill()
            .reset_index()
            .rename(columns={"index": ts_col})
        )

    # Coerce numerics; keep numeric columns only
    feature_cols: List[str] = []
    for c in df.columns:
        if c == ts_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        df[c] = s
        feature_cols.append(c)

    # Handle missing values (linear interpolation → ffill/bfill)
    df[feature_cols] = df[feature_cols].interpolate(limit_direction="both").ffill().bfill()

    # Drop constant features
    nunique = df[feature_cols].nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    dropped = sorted(set(feature_cols) - set(keep))
    if dropped:
        logging.warning(f"Dropping constant features: {dropped}")
    feature_cols = keep
    if not feature_cols:
        raise PrepError("All numeric features became constant after cleaning.")

    # Training mask
    mask_train = (df[ts_col] >= pd.Timestamp(normal_start)) & (
        df[ts_col] <= pd.Timestamp(normal_end)
    )
    if mask_train.sum() < 72:  # spec: minimum ~72 hours
        raise PrepError("Insufficient training data (< 72 rows).")

    # Scale with training-only fit to avoid leakage
    X_raw = df[feature_cols].values
    scaler = StandardScaler()
    scaler.fit(X_raw[mask_train.values])
    X_full = scaler.transform(X_raw)
    X_train = X_full[mask_train.values]

    meta = {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "train_mask": mask_train.values,
        "timestamps": df[ts_col].values,
        "timestamp_col": ts_col,
        "X_train": X_train,
        "analysis_len": len(df),
    }
    return df, X_full, meta
