#!/usr/bin/env python3
"""Diagnostics and validation checks for aliasing-controlled MiniGrid environments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.sensory.aliasing_control import compute_aliasing_score


def detect_symmetry_axes(mask: np.ndarray) -> list[str]:
    axes: list[str] = []
    if np.array_equal(mask, np.flip(mask, axis=1)):
        axes.append("vertical")
    if np.array_equal(mask, np.flip(mask, axis=0)):
        axes.append("horizontal")
    if np.array_equal(mask, np.flip(np.flip(mask, axis=0), axis=1)):
        axes.append("point")
    return axes


def duplicate_patch_fraction(observations: np.ndarray) -> float:
    if len(observations) == 0:
        return 0.0
    rounded = np.round(np.asarray(observations, dtype=np.float32), decimals=5)
    keys = [tuple(obs.tolist()) for obs in rounded]
    counts: dict[tuple[float, ...], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    duplicated = sum(1 for key in keys if counts[key] > 1)
    return duplicated / max(len(keys), 1)


def highlight_bottlenecks(graph: nx.Graph) -> set[tuple[int, int]]:
    return set(nx.articulation_points(graph))


def visualize_connectivity(env, output_dir: Path) -> Path:
    graph = env.build_geodesic_graph()
    mask = env.traversable_mask.astype(bool)
    articulation = highlight_bottlenecks(graph)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap="Greys", origin="lower")
    if graph.number_of_nodes() > 0:
        nodes = np.asarray(list(graph.nodes()), dtype=np.float32)
        ax.scatter(nodes[:, 1], nodes[:, 0], s=2, c="steelblue", alpha=0.18)
    if articulation:
        art = np.asarray(sorted(articulation), dtype=np.float32)
        ax.scatter(art[:, 1], art[:, 0], s=18, c="orange", label="bottlenecks")
        ax.legend(loc="upper right")
    ax.set_title(f"{env.env_name} connectivity")
    ax.set_axis_off()
    save_path = output_dir / f"{env.env_name}_connectivity.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def run_checks(env_name: str, cfg: dict | None = None) -> dict[str, object]:
    env = get_env(env_name, cfg)
    mask = env.traversable_mask.astype(bool)
    graph = env.build_geodesic_graph()
    rollout = env.sample_rollout(512, seed=env.seed + 101)
    aliasing_score = compute_aliasing_score(env)
    duplicated_fraction = duplicate_patch_fraction(rollout.observations)
    symmetry_axes = detect_symmetry_axes(mask)
    bottlenecks = highlight_bottlenecks(graph)
    disconnected = nx.number_connected_components(graph) != 1

    if duplicated_fraction > 0.20:
        warnings.warn(
            f"`{env_name}` has identical patches across {duplicated_fraction:.1%} of sampled positions.",
            stacklevel=2,
        )
    if symmetry_axes and getattr(env, "landmark_density", 0.0) <= 0.0 and getattr(env, "gradient_weight", 0.0) <= 0.0:
        warnings.warn(
            f"`{env_name}` is symmetric across {symmetry_axes} without disambiguating cues.",
            stacklevel=2,
        )
    if disconnected:
        warnings.warn(f"`{env_name}` has disconnected regions and is invalid.", stacklevel=2)

    return {
        "env_name": env.env_name,
        "canonical_name": getattr(env, "canonical_name", env.env_name),
        "aliasing_score": float(aliasing_score),
        "duplicate_patch_fraction": float(duplicated_fraction),
        "symmetry_axes": symmetry_axes,
        "n_bottlenecks": len(bottlenecks),
        "disconnected": disconnected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="square")
    parser.add_argument("--output-dir", type=str, default="project4_topology_before_geometry/figures/aliasing_diagnostics")
    args = parser.parse_args()

    cfg = {"seed": 42}
    env = get_env(args.env, cfg)
    result = run_checks(args.env, cfg)
    save_path = visualize_connectivity(env, Path(args.output_dir))
    print(result)
    print(f"connectivity_plot={save_path}")


if __name__ == "__main__":
    main()

