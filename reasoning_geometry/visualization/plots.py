from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path_base: str | Path) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1_mds(embeddings: Sequence[np.ndarray], hop_colors: Sequence[np.ndarray], cluster_colors: Sequence[np.ndarray]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for emb, hops in zip(embeddings, hop_colors):
        axes[0].scatter(emb[:, 0], emb[:, 1], c=hops, cmap="viridis", s=35)
        axes[0].plot(emb[:, 0], emb[:, 1], alpha=0.5)
    for emb, cluster in zip(embeddings, cluster_colors):
        axes[1].scatter(emb[:, 0], emb[:, 1], c=cluster, cmap="tab10", s=35)
        axes[1].plot(emb[:, 0], emb[:, 1], alpha=0.5)
    axes[0].set_title("Logical Hop Coloring")
    axes[1].set_title("Surface Cluster Coloring")
    return fig


def figure2_srsa(summary: Mapping[str, float], scatter_points: Sequence[tuple[float, float]]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(summary.keys())
    values = [summary[key] for key in labels]
    axes[0].bar(labels, values, color=["#175676", "#4BA3C3", "#CCE6F4", "#D62839"])
    axes[0].set_ylabel("Mean sRSA")
    if scatter_points:
        x, y = zip(*scatter_points)
        axes[1].scatter(x, y, alpha=0.7)
    axes[1].set_xlabel("sRSA logical")
    axes[1].set_ylabel("Best surface sRSA")
    return fig


def figure3_intrinsic_dimension(series: Mapping[str, tuple[float, float, float]]):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(series.keys())
    means = [series[key][0] for key in labels]
    lows = [series[key][0] - series[key][1] for key in labels]
    highs = [series[key][2] - series[key][0] for key in labels]
    ax.bar(labels, means, color=["#175676", "#4BA3C3", "#D62839"])
    ax.errorbar(labels, means, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4)
    ax.set_ylabel("Intrinsic Dimensionality")
    return fig


def figure4_gpu(angles: np.ndarray, null_95: np.ndarray | float, phi_factual: np.ndarray, phi_hall: np.ndarray, detector_curves: Mapping[str, Dict[str, Sequence[float]]]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(np.arange(1, len(angles) + 1), angles, color="#175676")
    if np.isscalar(null_95):
        axes[0].axhline(float(null_95), color="#D62839", linestyle="--")
    axes[0].set_title("Canonical Angle Spectrum")
    axes[1].plot(phi_factual, label="Factual", color="#175676")
    axes[1].plot(phi_hall, label="Hallucinated", color="#D62839")
    axes[1].legend()
    axes[1].set_title("Projection Ratio Time Series")
    for name, curves in detector_curves.items():
        axes[2].plot(curves["fpr"], curves["tpr"], label=name)
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].legend()
    axes[2].set_title("Detector ROC")
    return fig


def figure5_gpu(delta_phi_t_minus_2: Sequence[float], labels: Sequence[int], latencies: Mapping[str, Sequence[float] | float]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(delta_phi_t_minus_2, labels, alpha=0.7)
    axes[0].set_xlabel("Delta phi at t-2")
    axes[0].set_ylabel("Hallucination label")
    names = list(latencies.keys())
    values = [
        value if np.isscalar(value) else np.mean(np.asarray(value, dtype=np.float64))
        for value in latencies.values()
    ]
    axes[1].boxplot([[v] for v in values], labels=names)
    axes[1].set_ylabel("Detection latency")
    return fig

