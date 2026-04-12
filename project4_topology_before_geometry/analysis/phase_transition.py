"""Figure helpers for convergence and cross-environment summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_convergence_curves(log_df, run_name: str, topology_label: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    x_col = "trial"
    x_label = "Update" if "trajectories_seen" in log_df.columns else "Trial"
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(log_df[x_col], log_df["srsa_geodesic"], color="tab:blue", label="sRSA geodesic")
    if "srsa_euclidean" in log_df:
        ax1.plot(log_df[x_col], log_df["srsa_euclidean"], color="tab:cyan", alpha=0.5, label="sRSA euclidean")
    if "betti_correct" in log_df:
        ax2.step(log_df[x_col], log_df["betti_correct"].fillna(0), color="tab:orange", where="post", label="betti correct")
    if "rdm_frobenius_delta" in log_df:
        ax2.plot(log_df[x_col], log_df["rdm_frobenius_delta"], color="0.5", alpha=0.7, label="RDM drift")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("sRSA")
    ax2.set_ylabel("Topology / Drift")
    ax1.set_title(f"{run_name} | B0={topology_label['betti_0']} B1={topology_label['betti_1']}")
    path = output_dir / f"{run_name}_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_sleep_wake(hidden_wake: np.ndarray, hidden_sleep: np.ndarray, positions: np.ndarray, output_dir: Path, name: str) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    keep = min(len(hidden_wake), 4000)
    pca = PCA(n_components=2)
    wake_2d = pca.fit_transform(hidden_wake[:keep])
    sleep_2d = pca.transform(hidden_sleep[: min(len(hidden_sleep), keep)])
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    scatter = ax.scatter(wake_2d[:, 0], wake_2d[:, 1], c=positions[:keep, 0], s=4, cmap="viridis", alpha=0.7)
    ax.scatter(sleep_2d[:, 0], sleep_2d[:, 1], color="red", s=6, alpha=0.35)
    fig.colorbar(scatter, ax=ax, label="x position")
    ax.set_title(f"Sleep/Wake manifold: {name}")
    path = output_dir / f"{name}_sleep_wake.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_gap_vs_complexity(all_results: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    values = list(all_results.values())
    x = [entry.get("complexity_index", np.nan) for entry in values]
    y = [entry.get("gap", np.nan) for entry in values]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, c="tab:green")
    ax.set_xlabel("Complexity index")
    ax.set_ylabel("Gap")
    ax.set_title("Gap vs complexity")
    path = output_dir / "gap_vs_complexity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_gap_vs_aliasing(all_results: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    values = list(all_results.values())
    x = [entry.get("aliasing_score", np.nan) for entry in values]
    y = [entry.get("gap", np.nan) for entry in values]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, c="tab:purple")
    ax.set_xlabel("Aliasing score")
    ax.set_ylabel("Gap")
    ax.set_title("Gap vs aliasing")
    path = output_dir / "gap_vs_aliasing.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_manifold_snapshots(*_, **__):
    return None


def plot_b1_persistence_vs_radius(summary: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    radii = summary.get("radii", [])
    lifetimes = summary.get("lifetimes", [])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(radii, lifetimes, marker="o", color="coral")
    ax.set_xlabel("Inner radius")
    ax.set_ylabel("Dominant H1 lifetime")
    ax.set_title("B1 persistence vs radius")
    path = output_dir / "b1_persistence_vs_radius.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cross_env_summary(all_results: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    values = list(all_results.values())
    x = [entry.get("T_topology", np.nan) for entry in values]
    y = [entry.get("T_geometry", np.nan) for entry in values]
    colors = [entry.get("betti_1_gt", 0) for entry in values]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    scatter = ax.scatter(x, y, c=colors, cmap="coolwarm", s=40)
    finite_xy = np.asarray([(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)], dtype=float)
    limit = float(np.max(finite_xy)) if finite_xy.size else 1.0
    ax.plot([0, limit], [0, limit], linestyle="--", color="0.4")
    ax.set_xlabel("T_topology")
    ax.set_ylabel("T_geometry")
    ax.set_title("Cross-environment convergence summary")
    fig.colorbar(scatter, ax=ax, label="Betti-1 ground truth")
    path = output_dir / "cross_env_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
