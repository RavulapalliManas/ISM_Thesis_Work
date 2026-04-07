from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_matrix(similarity: np.ndarray, env_ids: Sequence[str]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(similarity, cmap="viridis", vmin=np.nanmin(similarity), vmax=np.nanmax(similarity))
    ax.set_xticks(range(len(env_ids)))
    ax.set_xticklabels(env_ids, rotation=90)
    ax.set_yticks(range(len(env_ids)))
    ax.set_yticklabels(env_ids)
    ax.set_title("Structural Similarity Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_transfer_vs_similarity(
    similarity_values: np.ndarray,
    transfer_values: np.ndarray,
    *,
    title: str = "Transfer Efficiency vs Structural Similarity",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(similarity_values, transfer_values, color="black", alpha=0.8)
    if len(similarity_values) >= 2:
        slope, intercept = np.polyfit(similarity_values, transfer_values, deg=1)
        xs = np.linspace(similarity_values.min(), similarity_values.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="tab:red", linewidth=2)
    ax.set_xlabel("Structural Similarity")
    ax.set_ylabel("Transfer Efficiency")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_learning_curve(history: Sequence[dict]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 4))
    steps = [entry["trained_steps"] for entry in history]
    srsa = [entry["sRSA"] for entry in history]
    ax.plot(steps, srsa, color="tab:blue", linewidth=2)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("sRSA")
    ax.set_title("Learning Curve")
    fig.tight_layout()
    return fig, ax


__all__ = [
    "plot_learning_curve",
    "plot_similarity_matrix",
    "plot_transfer_vs_similarity",
]
