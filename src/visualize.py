from __future__ import annotations
from typing import Iterable, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scores(
    timestamps: Iterable,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None
) -> None:
    """Line plot of anomaly scores with anomalies highlighted."""
    plt.figure(figsize=(14, 6))

    # Plot anomaly scores
    plt.plot(
        timestamps, scores,
        label="Anomaly Score",
        color="blue", alpha=0.7
    )

    # Highlight anomalies
    idx = np.where(labels == 1)[0]
    if len(idx):
        plt.scatter(
            np.array(timestamps)[idx],
            scores[idx],
            label="Detected Anomaly",
            color="red", marker="x", s=40
        )

    # Add threshold line if provided
    if threshold is not None:
        plt.axhline(
            threshold, color="green", linestyle="--", linewidth=1.5,
            label=f"Decision Threshold ({threshold:.2f})"
        )

    # Labels and legend
    plt.title("Anomaly Scores Over Time", fontsize=14)
    plt.ylabel("Score (0–100)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_features(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    top_n: int = 5
) -> None:
    """Bar chart of most frequently attributed features among anomalies."""
    anom = df[labels == 1]
    cols = [f"top_feature_{i}" for i in range(1, 8) if f"top_feature_{i}" in anom]
    if not cols:
        return

    # Count frequency of features
    counts: dict[str, int] = {}
    for c in cols:
        for name in anom[c]:
            if isinstance(name, str) and name.strip():
                counts[name] = counts.get(name, 0) + 1

    if not counts:
        return

    # Sort by frequency and take top_n
    names, vals = zip(*sorted(counts.items(), key=lambda kv: -kv[1])[:top_n])

    # Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, vals, color="orange", alpha=0.8)

    # Add count labels above bars
    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(val),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.title(f"Top {top_n} Contributing Features in Anomalies", fontsize=14)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
