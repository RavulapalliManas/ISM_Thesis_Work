from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from project3_generalization.environments.suite_2d import (
    DEFAULT_DT,
    EnvironmentSpec2D,
    build_suite_2d,
    simulate_random_walk_2d,
)


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(6, cpu_count - 2))


@dataclass(frozen=True)
class SimilarityConfig:
    num_steps: int = 25_000
    grid_size: int = 30
    gamma: float = 0.9
    temporal_horizon: int = 10
    dt: float = DEFAULT_DT
    cg_tolerance: float = 1e-5
    cg_max_iter: int = 1_500
    num_workers: int = _default_num_workers()
    use_memmap: bool = False
    memmap_dir: str | Path | None = None


@dataclass
class TransitionEstimate:
    env_id: str
    transition_matrix: sparse.csr_matrix
    successor_representation: np.ndarray | None
    occupancy: np.ndarray
    extent: np.ndarray
    successor_representation_path: str | None = None

    def load_successor_representation(self) -> np.ndarray:
        if self.successor_representation is not None:
            return np.asarray(self.successor_representation)
        if self.successor_representation_path is None:
            raise ValueError("No successor representation is attached to this estimate.")
        return np.load(self.successor_representation_path, mmap_mode="r")


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
    num_steps: int = 25_000,
    grid_size: int = 30,
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
    occupancy = np.bincount(flat_indices, minlength=n_states).astype(np.float32)

    counts = sparse.lil_matrix((n_states, n_states), dtype=np.float32)
    max_lag = max(1, temporal_horizon)
    for lag in range(1, max_lag + 1):
        weight = float(gamma ** (lag - 1))
        sources = flat_indices[:-lag]
        targets = flat_indices[lag:]
        lag_counts = sparse.coo_matrix(
            (
                np.full(sources.shape, weight, dtype=np.float32),
                (sources, targets),
            ),
            shape=(n_states, n_states),
        )
        counts += lag_counts

    counts = counts.tocsr()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    transition = sparse.diags(1.0 / row_sums.astype(np.float32)) @ counts
    return transition.tocsr(), occupancy.reshape(grid_size, grid_size), np.asarray(env.extent, dtype=np.float32)


def _cg_solve(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    try:
        solution, info = sparse_linalg.cg(matrix, rhs, rtol=tolerance, atol=0.0, maxiter=max_iter)
    except TypeError:
        solution, info = sparse_linalg.cg(matrix, rhs, tol=tolerance, maxiter=max_iter)
    if info != 0:
        try:
            solution, _ = sparse_linalg.cg(matrix, rhs, rtol=max(tolerance * 10.0, 1e-4), atol=0.0, maxiter=max_iter * 2)
        except TypeError:
            solution, _ = sparse_linalg.cg(matrix, rhs, tol=max(tolerance * 10.0, 1e-4), maxiter=max_iter * 2)
    return np.asarray(solution, dtype=np.float32)


def compute_successor_representation(
    transition_matrix: sparse.csr_matrix,
    gamma: float = 0.9,
    *,
    tolerance: float = 1e-5,
    max_iter: int = 1_500,
    output_path: str | Path | None = None,
) -> np.ndarray:
    n_states = transition_matrix.shape[0]
    system = sparse.eye(n_states, format="csr", dtype=np.float32) - (gamma * transition_matrix).tocsr().astype(np.float32)
    normal_matrix = (system.T @ system).tocsr() + sparse.eye(n_states, format="csr", dtype=np.float32) * 1e-5

    if output_path is None:
        sr = np.zeros((n_states, n_states), dtype=np.float32)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(n_states, n_states))

    for state_idx in range(n_states):
        rhs = np.asarray(system.getrow(state_idx).toarray()).ravel()
        sr[:, state_idx] = _cg_solve(normal_matrix, rhs, tolerance=tolerance, max_iter=max_iter)

    if isinstance(sr, np.memmap):
        sr.flush()
    return sr


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
    sr_path = None
    if config.use_memmap and config.memmap_dir is not None:
        sr_path = str(Path(config.memmap_dir) / f"{spec.env_id}_sr.npy")
    sr = compute_successor_representation(
        transition,
        gamma=config.gamma,
        tolerance=config.cg_tolerance,
        max_iter=config.cg_max_iter,
        output_path=sr_path,
    )
    return TransitionEstimate(
        env_id=spec.env_id,
        transition_matrix=transition,
        successor_representation=None if sr_path is not None else sr,
        occupancy=occupancy,
        extent=extent,
        successor_representation_path=sr_path,
    )


def _estimate_environment_structure_worker(
    payload: tuple[EnvironmentSpec2D, dict, int],
) -> TransitionEstimate:
    spec, config_dict, seed = payload
    return estimate_environment_structure(spec, SimilarityConfig(**config_dict), seed=seed)


def compute_structural_similarity(sr_a: np.ndarray | Path | str, sr_b: np.ndarray | Path | str) -> float:
    matrix_a = np.load(sr_a, mmap_mode="r") if isinstance(sr_a, (str, Path)) else np.asarray(sr_a)
    matrix_b = np.load(sr_b, mmap_mode="r") if isinstance(sr_b, (str, Path)) else np.asarray(sr_b)
    numerator = np.linalg.norm(matrix_a - matrix_b, ord="fro")
    denominator = np.linalg.norm(matrix_a, ord="fro") + np.linalg.norm(matrix_b, ord="fro") + 1e-12
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

    env_ids = [spec.env_id for spec in specs]
    memmap_dir = None
    if config.use_memmap:
        if config.memmap_dir is not None:
            memmap_dir = Path(config.memmap_dir)
        elif output_path is not None:
            memmap_dir = Path(output_path).parent / "sr_memmap"
        else:
            memmap_dir = Path("project3_generalization/results/hardware_constrained/sr_memmap")
        memmap_dir.mkdir(parents=True, exist_ok=True)
        config = SimilarityConfig(**{**asdict(config), "memmap_dir": str(memmap_dir)})

    estimates: dict[str, TransitionEstimate] = {}
    if config.num_workers > 1 and len(specs) > 1:
        ctx = mp.get_context("spawn")
        payloads = [(spec, asdict(config), seed + idx) for idx, spec in enumerate(specs)]
        with ctx.Pool(processes=min(config.num_workers, len(specs))) as pool:
            for estimate in pool.map(_estimate_environment_structure_worker, payloads):
                estimates[estimate.env_id] = estimate
    else:
        for idx, spec in enumerate(specs):
            estimates[spec.env_id] = estimate_environment_structure(spec, config, seed=seed + idx)

    similarity = np.eye(len(specs), dtype=np.float32)
    for row, spec_a in enumerate(specs):
        sr_a = estimates[spec_a.env_id].successor_representation_path or estimates[spec_a.env_id].load_successor_representation()
        for col in range(row + 1, len(specs)):
            spec_b = specs[col]
            sr_b = estimates[spec_b.env_id].successor_representation_path or estimates[spec_b.env_id].load_successor_representation()
            sim_value = compute_structural_similarity(sr_a, sr_b)
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
