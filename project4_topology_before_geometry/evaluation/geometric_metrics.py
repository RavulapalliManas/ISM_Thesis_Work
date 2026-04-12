"""Geometric evaluation metrics for predictive hippocampal representations."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_distances

from project3_generalization.evaluation.metrics import compute_tuning_curves, participation_ratio


def _subsample_aligned(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    max_samples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(hidden_states), len(positions))
    hidden_states = np.asarray(hidden_states[:n], dtype=np.float32)
    positions = np.asarray(positions[:n], dtype=np.float32)
    if n <= max_samples:
        return hidden_states, positions
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n, size=max_samples, replace=False))
    return hidden_states[keep], positions[keep]


def _pairwise_geodesic_distances(environment, positions: np.ndarray) -> np.ndarray:
    graph = environment.build_geodesic_graph()
    nodes = [tuple(node) for node in environment.discretize_positions(positions)]
    unique_nodes = sorted(set(nodes))
    all_lengths: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for node in unique_nodes:
        all_lengths[node] = nx.single_source_dijkstra_path_length(graph, node, weight="weight")

    dists: list[float] = []
    for idx in range(len(nodes)):
        for jdx in range(idx + 1, len(nodes)):
            dists.append(float(all_lengths.get(nodes[idx], {}).get(nodes[jdx], math.inf)))
    dists = np.asarray(dists, dtype=np.float32)
    if np.any(~np.isfinite(dists)):
        finite = dists[np.isfinite(dists)]
        fill = float(np.max(finite)) if finite.size else 0.0
        dists[~np.isfinite(dists)] = fill
    return dists


def compute_srsa(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    environment,
    distance_type: str = "geodesic",
    max_samples: int = 750,
    seed: int = 42,
) -> float:
    """Compute representational similarity against Euclidean or geodesic spatial distances."""
    hidden_states, positions = _subsample_aligned(hidden_states, positions, max_samples=max_samples, seed=seed)
    neural_rdm = pdist(np.asarray(hidden_states, dtype=np.float32), metric="cosine")
    if distance_type == "euclidean":
        spatial_rdm = pdist(np.asarray(positions, dtype=np.float32), metric="euclidean")
    elif distance_type == "geodesic":
        spatial_rdm = _pairwise_geodesic_distances(environment, positions)
    else:
        raise ValueError(f"Unsupported distance_type `{distance_type}`.")
    rho, _ = spearmanr(neural_rdm, spatial_rdm)
    return float(rho)


def compute_sw_dist(hidden_wake: np.ndarray, hidden_sleep: np.ndarray, max_samples: int = 2000) -> float:
    """Sleep-wake distance from the paper: nearest wake-state cosine distance averaged over sleep."""
    hidden_wake = np.asarray(hidden_wake[:max_samples], dtype=np.float32)
    hidden_sleep = np.asarray(hidden_sleep[:max_samples], dtype=np.float32)
    distances = cosine_distances(hidden_sleep, hidden_wake)
    return float(np.mean(np.min(distances, axis=1)))


def compute_spatial_information(hidden_states: np.ndarray, positions: np.ndarray, grid_size: int = 20) -> dict[str, np.ndarray | float]:
    """Compute unitwise spatial information and its population average."""
    tuning = compute_tuning_curves(np.asarray(hidden_states), np.asarray(positions), grid_size=grid_size)
    spatial_info = np.asarray(tuning["spatial_information"], dtype=np.float32)
    return {"spatial_information": spatial_info, "mean_spatial_information": float(np.mean(spatial_info))}


def compute_decoding_error(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    device: torch.device | str | None = None,
    n_updates: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 3e-1,
    batch_size: int = 5000,
) -> float:
    """Train a linear decoder on wake activity and report cityblock decoding error."""
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_states), len(positions))
    if n == 0:
        return float("nan")

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(hidden_states[:n], dtype=torch.float32, device=device)
    y = torch.as_tensor(positions[:n], dtype=torch.float32, device=device)
    decoder = torch.nn.Linear(x.shape[-1], y.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.default_rng(42)
    for _ in range(int(n_updates)):
        keep = rng.choice(n, size=min(batch_size, n), replace=False)
        pred = decoder(x[keep])
        loss = torch.nn.functional.mse_loss(pred, y[keep])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        decoded = decoder(x).cpu().numpy()
    return float(np.mean(np.abs(decoded - positions[:n]).sum(axis=1)))


def compute_explained_variance_spatial(hidden_states: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray | float]:
    """Estimate spatial explained variance for each unit with a linear readout from position."""
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_states), len(positions))
    if n == 0:
        return {"evs": np.zeros((0,), dtype=np.float32), "fraction_tuned": 0.0}
    positions = positions[:n]
    hidden_states = hidden_states[:n]
    evs = np.zeros((hidden_states.shape[1],), dtype=np.float32)
    reg = LinearRegression()
    for unit in range(hidden_states.shape[1]):
        reg.fit(positions, hidden_states[:, unit])
        pred = reg.predict(positions)
        evs[unit] = float(max(0.0, r2_score(hidden_states[:, unit], pred)))
    return {"evs": evs, "fraction_tuned": float(np.mean(evs > 0.5))}


def compute_participation_ratio(hidden_states: np.ndarray) -> float:
    """Proxy for manifold dimensionality / attractor coherence."""
    return float(participation_ratio(np.asarray(hidden_states, dtype=np.float32)))


def compute_place_field_coverage_map(hidden_states: np.ndarray, positions: np.ndarray, env, grid_size: int = 24) -> np.ndarray:
    """Map where the strongest mean population activity is concentrated."""
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_states), len(positions))
    if n == 0:
        return np.zeros_like(env.traversable_mask, dtype=np.float32)

    positions = positions[:n]
    hidden_states = hidden_states[:n]
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-6)
    xs = np.clip(((positions[:, 0] - mins[0]) / ranges[0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    ys = np.clip(((positions[:, 1] - mins[1]) / ranges[1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    coverage = np.zeros((grid_size, grid_size), dtype=np.float32)
    activity = np.mean(hidden_states, axis=1)
    np.add.at(coverage, (ys, xs), activity)
    return coverage

