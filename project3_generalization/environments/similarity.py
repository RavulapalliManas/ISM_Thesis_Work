from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from project3_generalization.environments.suite_2d import (
    DEFAULT_DT,
    EnvironmentSpec2D,
    build_suite_2d,
    simulate_random_walk_2d,
)


@dataclass(frozen=True)
class SimilarityConfig:
    num_steps: int = 100_000
    grid_size: int = 50
    gamma: float = 0.9
    temporal_horizon: int = 10
    dt: float = DEFAULT_DT


@dataclass
class TransitionEstimate:
    env_id: str
    transition_matrix: sparse.csr_matrix
    successor_representation: np.ndarray
    occupancy: np.ndarray
    extent: np.ndarray


def _positions_to_grid_indices(
    positions: np.ndarray,
    extent: Sequence[float],
    grid_size: int,
) -> np.ndarray:
    left, right, bottom, top = extent
    width = max(right - left, 1e-9)
    height = max(top - bottom, 1e-9)
    xs = np.clip(((positions[:, 0] - left) / width * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip(((positions[:, 1] - bottom) / height * grid_size).astype(int), 0, grid_size - 1)
    return ys * grid_size + xs


def estimate_transition_matrix(
    spec: EnvironmentSpec2D,
    *,
    num_steps: int = 100_000,
    grid_size: int = 50,
    gamma: float = 0.9,
    temporal_horizon: int = 10,
    seed: int | None = None,
    dt: float = DEFAULT_DT,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    env, _, positions, _, _ = simulate_random_walk_2d(
        spec,
        n_steps=num_steps,
        seed=seed,
        dt=dt,
    )
    n_states = grid_size * grid_size
    flat_indices = _positions_to_grid_indices(positions, env.extent, grid_size)
    occupancy = np.bincount(flat_indices, minlength=n_states).astype(float)

    counts = sparse.lil_matrix((n_states, n_states), dtype=np.float64)
    max_lag = max(1, temporal_horizon)
    for lag in range(1, max_lag + 1):
        weight = gamma ** (lag - 1)
        sources = flat_indices[:-lag]
        targets = flat_indices[lag:]
        lag_counts = sparse.coo_matrix(
            (
                np.full_like(sources, weight, dtype=np.float64),
                (sources, targets),
            ),
            shape=(n_states, n_states),
        )
        counts += lag_counts

    counts = counts.tocsr()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    normalizer = sparse.diags(1.0 / row_sums)
    transition = normalizer @ counts
    return transition.tocsr(), occupancy.reshape(grid_size, grid_size), np.asarray(env.extent, dtype=float)


def compute_successor_representation(
    transition_matrix: sparse.csr_matrix,
    gamma: float = 0.9,
) -> np.ndarray:
    n_states = transition_matrix.shape[0]
    system = sparse.eye(n_states, format="csc") - (gamma * transition_matrix).tocsc()
    identity = np.eye(n_states, dtype=np.float64)
    lu = sparse_linalg.splu(system)
    return lu.solve(identity)


def estimate_environment_structure(
    spec: EnvironmentSpec2D,
    config: SimilarityConfig,
    *,
    seed: int | None = None,
) -> TransitionEstimate:
    transition, occupancy, extent = estimate_transition_matrix(
        spec,
        num_steps=config.num_steps,
        grid_size=config.grid_size,
        gamma=config.gamma,
        temporal_horizon=config.temporal_horizon,
        seed=seed,
        dt=config.dt,
    )
    sr = compute_successor_representation(transition, gamma=config.gamma)
    return TransitionEstimate(
        env_id=spec.env_id,
        transition_matrix=transition,
        successor_representation=sr,
        occupancy=occupancy,
        extent=extent,
    )


def compute_structural_similarity(sr_a: np.ndarray, sr_b: np.ndarray) -> float:
    numerator = np.linalg.norm(sr_a - sr_b, ord="fro")
    denominator = np.linalg.norm(sr_a, ord="fro") + np.linalg.norm(sr_b, ord="fro") + 1e-12
    return float(1.0 - numerator / denominator)


def compute_similarity_matrix(
    specs: Sequence[EnvironmentSpec2D] | None = None,
    *,
    config: SimilarityConfig | None = None,
    seed: int = 0,
    output_path: str | Path | None = None,
    return_estimates: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, TransitionEstimate]] | tuple[np.ndarray, list[str]]:
    if specs is None:
        specs = list(build_suite_2d().values())
    if config is None:
        config = SimilarityConfig()

    estimates: dict[str, TransitionEstimate] = {}
    env_ids = [spec.env_id for spec in specs]
    for idx, spec in enumerate(specs):
        estimates[spec.env_id] = estimate_environment_structure(spec, config, seed=seed + idx)

    similarity = np.eye(len(specs), dtype=np.float64)
    for row, spec_a in enumerate(specs):
        for col in range(row + 1, len(specs)):
            spec_b = specs[col]
            sim_value = compute_structural_similarity(
                estimates[spec_a.env_id].successor_representation,
                estimates[spec_b.env_id].successor_representation,
            )
            similarity[row, col] = sim_value
            similarity[col, row] = sim_value

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            similarity=similarity,
            env_ids=np.array(env_ids, dtype=object),
        )

    if return_estimates:
        return similarity, env_ids, estimates
    return similarity, env_ids


__all__ = [
    "SimilarityConfig",
    "TransitionEstimate",
    "compute_similarity_matrix",
    "compute_structural_similarity",
    "compute_successor_representation",
    "estimate_environment_structure",
    "estimate_transition_matrix",
]
