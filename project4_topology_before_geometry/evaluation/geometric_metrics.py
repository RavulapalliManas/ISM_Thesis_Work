"""Geometric evaluation metrics for predictive hippocampal representations."""

from __future__ import annotations

import time
import warnings
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances

from project3_generalization.evaluation.metrics import participation_ratio


_GEODESIC_CACHE: dict[str, dict[str, Any]] = {}


def _env_cache_key(env) -> str:
    return str(getattr(env, "name", getattr(env, "env_name", env.__class__.__name__)))


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


def precompute_geodesic_matrix(env, resolution: float = 0.05) -> np.ndarray:
    """Precompute and cache all-pairs geodesic distances for one environment."""
    key = _env_cache_key(env)
    cached = _GEODESIC_CACHE.get(key)
    if cached is not None and float(cached["resolution"]) == float(resolution):
        return cached["matrix"]

    start = time.perf_counter()
    graph = env.build_geodesic_graph()
    nodes = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    index_map = np.full(env.traversable_mask.shape, -1, dtype=np.int32)
    for node, idx in node_to_index.items():
        index_map[node[0], node[1]] = idx

    matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=np.float32)
    for idx, node in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight="weight")
        matrix[idx, idx] = 0.0
        for target, distance in lengths.items():
            matrix[idx, node_to_index[target]] = float(distance)

    _GEODESIC_CACHE[key] = {
        "matrix": matrix,
        "node_to_index": node_to_index,
        "index_map": index_map,
        "nodes": np.asarray(nodes, dtype=np.int32),
        "resolution": float(resolution),
        "elapsed_s": float(time.perf_counter() - start),
    }
    return matrix


def _positions_to_cached_indices(positions: np.ndarray, env) -> np.ndarray:
    key = _env_cache_key(env)
    cache = _GEODESIC_CACHE[key]
    discrete = env.discretize_positions(np.asarray(positions, dtype=np.float32))
    index_map = cache["index_map"]
    nodes = cache["nodes"]

    indices = np.full((len(discrete),), -1, dtype=np.int32)
    valid_rows = (0 <= discrete[:, 0]) & (discrete[:, 0] < index_map.shape[0])
    valid_cols = (0 <= discrete[:, 1]) & (discrete[:, 1] < index_map.shape[1])
    valid = valid_rows & valid_cols
    indices[valid] = index_map[discrete[valid, 0], discrete[valid, 1]]

    missing = np.where(indices < 0)[0]
    if missing.size:
        for idx in missing.tolist():
            deltas = nodes - discrete[idx][None, :]
            nearest = int(np.argmin(np.sum(deltas * deltas, axis=1)))
            indices[idx] = nearest
    return indices


def lookup_geodesic_distance(pos_a, pos_b, env) -> float:
    """Look up the cached geodesic distance between two continuous positions."""
    key = _env_cache_key(env)
    if key not in _GEODESIC_CACHE:
        precompute_geodesic_matrix(env)
    cache = _GEODESIC_CACHE[key]
    indices = _positions_to_cached_indices(np.asarray([pos_a, pos_b], dtype=np.float32), env)
    return float(cache["matrix"][indices[0], indices[1]])


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
            dists.append(float(all_lengths.get(nodes[idx], {}).get(nodes[jdx], np.inf)))
    dists = np.asarray(dists, dtype=np.float32)
    if np.any(~np.isfinite(dists)):
        finite = dists[np.isfinite(dists)]
        fill = float(np.max(finite)) if finite.size else 0.0
        dists[~np.isfinite(dists)] = fill
    return dists


def _cached_pairwise_geodesic_distances(environment, positions: np.ndarray) -> np.ndarray:
    key = _env_cache_key(environment)
    cache = _GEODESIC_CACHE[key]
    node_indices = _positions_to_cached_indices(positions, environment)
    pairwise = cache["matrix"][np.ix_(node_indices, node_indices)]
    triu = np.triu_indices(len(node_indices), k=1)
    condensed = np.asarray(pairwise[triu], dtype=np.float32)
    if np.any(~np.isfinite(condensed)):
        finite = condensed[np.isfinite(condensed)]
        fill = float(np.max(finite)) if finite.size else 0.0
        condensed[~np.isfinite(condensed)] = fill
    return condensed


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
        key = _env_cache_key(environment)
        if key in _GEODESIC_CACHE:
            spatial_rdm = _cached_pairwise_geodesic_distances(environment, positions)
        else:
            warnings.warn(
                f"Geodesic cache missing for `{key}`. Falling back to uncached Dijkstra; call precompute_geodesic_matrix() at init.",
                stacklevel=2,
            )
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


def compute_spatial_information_vectorized(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    n_bins: int = 20,
    tuned_threshold: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Compute spatial information for all units simultaneously using vectorized binning.

    This preserves the legacy project metric exactly while removing the Python
    loop over units. The legacy implementation weights by occupancy probability
    and mean bin activity, not by the normalized rate ratio alone.
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_states), len(positions))
    if n == 0:
        return np.zeros((0,), dtype=np.float32), 0.0

    hidden_states = hidden_states[:n]
    positions = positions[:n]
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-6)
    xs = np.clip(((positions[:, 0] - mins[0]) / ranges[0] * n_bins).astype(int), 0, n_bins - 1)
    ys = np.clip(((positions[:, 1] - mins[1]) / ranges[1] * n_bins).astype(int), 0, n_bins - 1)
    flat_bins = ys * n_bins + xs
    n_total_bins = n_bins * n_bins

    occupancy = np.bincount(flat_bins, minlength=n_total_bins).astype(np.float32)
    occupancy_prob = occupancy / max(float(occupancy.sum()), 1.0)

    rate_sums = np.zeros((n_total_bins, hidden_states.shape[1]), dtype=np.float32)
    np.add.at(rate_sums, flat_bins, hidden_states)
    mean_rates = np.divide(rate_sums, np.maximum(occupancy[:, None], 1.0), dtype=np.float32)
    global_mean = np.maximum(np.mean(hidden_states, axis=0, keepdims=True), 1e-9)
    ratio = np.maximum(mean_rates / global_mean, 1e-9)

    si_per_unit = np.sum(
        occupancy_prob[:, None] * mean_rates * np.log2(ratio),
        axis=0,
    ).astype(np.float32)
    return si_per_unit, float(np.mean(si_per_unit > tuned_threshold))


def compute_spatial_information(hidden_states: np.ndarray, positions: np.ndarray, grid_size: int = 20) -> dict[str, np.ndarray | float]:
    """Compute unitwise spatial information and its population average."""
    spatial_info, fraction_tuned = compute_spatial_information_vectorized(
        hidden_states,
        positions,
        n_bins=grid_size,
    )
    return {
        "spatial_information": spatial_info,
        "mean_spatial_information": float(np.mean(spatial_info)) if spatial_info.size else 0.0,
        "fraction_tuned": fraction_tuned,
    }


def compute_decoding_error(hidden_states: np.ndarray, positions: np.ndarray) -> float:
    """Fit a ridge decoder from hidden states to 2-D position and report mean Euclidean error."""
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_states), len(positions))
    if n == 0:
        return float("nan")

    decoder = Ridge(alpha=1.0)
    decoder.fit(hidden_states[:n], positions[:n])
    pred = decoder.predict(hidden_states[:n])
    return float(np.mean(np.sqrt(np.sum((pred - positions[:n]) ** 2, axis=1))))


def compute_explained_variance_spatial(hidden_states: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray | float]:
    """Estimate spatial explained variance for each unit with one vectorized linear solve."""
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_states), len(positions))
    if n == 0:
        return {"evs": np.zeros((0,), dtype=np.float32), "fraction_tuned": 0.0}

    x = np.concatenate([positions[:n], np.ones((n, 1), dtype=np.float32)], axis=1)
    y = hidden_states[:n]
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeffs
    residual = np.sum((y - pred) ** 2, axis=0)
    total = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2, axis=0)
    evs = 1.0 - np.divide(residual, np.maximum(total, 1e-9), dtype=np.float32)
    evs = np.clip(evs.astype(np.float32), a_min=0.0, a_max=None)
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
