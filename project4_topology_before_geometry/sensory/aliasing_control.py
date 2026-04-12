"""Aliasing controls and environment-level aliasing descriptors."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


DEFAULT_TILE_COLORS = ("red", "blue", "yellow", "green", "purple")
ALIAS_PRESETS: dict[str, dict[str, float | int | str]] = {
    "zero_alias": {"tile_period": 1, "landmark_density": 0.28, "wall_entropy": 0.35, "gradient_weight": 1.0, "aliasing_type": "gradient"},
    "low_alias": {"tile_period": 2, "landmark_density": 0.18, "wall_entropy": 0.25, "gradient_weight": 0.65, "aliasing_type": "sparse_random"},
    "medium_alias": {"tile_period": 4, "landmark_density": 0.10, "wall_entropy": 0.12, "gradient_weight": 0.30, "aliasing_type": "periodic"},
    "high_alias": {"tile_period": 8, "landmark_density": 0.03, "wall_entropy": 0.03, "gradient_weight": 0.05, "aliasing_type": "periodic"},
    "maximum_alias": {"tile_period": 10_000, "landmark_density": 0.0, "wall_entropy": 0.0, "gradient_weight": 0.0, "aliasing_type": "uniform"},
}


def _grid_spacing(aliasing_level: str) -> int:
    aliasing_level = str(aliasing_level).lower()
    if aliasing_level == "low":
        return 2
    if aliasing_level == "medium":
        return 4
    if aliasing_level == "high":
        return 8
    if aliasing_level == "none":
        return 10_000
    raise ValueError(f"Unsupported aliasing level `{aliasing_level}`.")


def alias_level_from_params(tile_period: int | None, landmark_density: float | None, gradient_weight: float | None) -> str:
    """Assign a coarse aliasing bucket from explicit alias parameters."""
    tile_period = int(tile_period or 10_000)
    landmark_density = float(landmark_density or 0.0)
    gradient_weight = float(gradient_weight or 0.0)
    if landmark_density <= 0.0 and gradient_weight <= 0.0 and tile_period >= 10_000:
        return "none"
    if tile_period <= 2 or landmark_density >= 0.16 or gradient_weight >= 0.5:
        return "low"
    if tile_period <= 4 or landmark_density >= 0.08 or gradient_weight >= 0.2:
        return "medium"
    return "high"


def preset_to_params(alias_preset: str) -> dict[str, float | int | str]:
    """Resolve a named alias preset into explicit control values."""
    preset_key = str(alias_preset).lower()
    if preset_key not in ALIAS_PRESETS:
        raise KeyError(f"Unknown alias preset `{alias_preset}`. Available: {sorted(ALIAS_PRESETS)}")
    return dict(ALIAS_PRESETS[preset_key])


def _traversable_coords(env_shape: tuple[int, int], mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    height, width = map(int, env_shape)
    if mask is None:
        return [(row, col) for row in range(height) for col in range(width)]
    traversable = np.asarray(mask, dtype=bool)
    return [(int(row), int(col)) for row, col in np.argwhere(traversable)]


def _boundary_coords(coords: list[tuple[int, int]], env_shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, env_shape)
    return [
        (row, col)
        for row, col in coords
        if row in (0, height - 1) or col in (0, width - 1)
    ]


def _center_sorted_coords(coords: list[tuple[int, int]], env_shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, env_shape)
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0], dtype=np.float32)
    return sorted(coords, key=lambda rc: float(np.sum((np.asarray(rc, dtype=np.float32) - center) ** 2)))


def _base_coords(
    env_shape: tuple[int, int],
    aliasing_type: str,
    spacing: int,
    *,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    traversable = _traversable_coords(env_shape, mask)
    traversable_set = set(traversable)
    height, width = map(int, env_shape)

    if aliasing_type == "uniform":
        return []
    if aliasing_type == "periodic":
        return [(row, col) for row in range(0, height, spacing) for col in range(0, width, spacing) if (row, col) in traversable_set]
    if aliasing_type == "clustered":
        return [
            (row, col)
            for row in range(0, max(height // 2, 1), max(spacing // 2, 1))
            for col in range(0, max(width // 2, 1), max(spacing // 2, 1))
            if (row, col) in traversable_set
        ]
    if aliasing_type == "sparse_random":
        n_tiles = max(1, math.ceil(len(traversable) / max(spacing ** 2, 1)))
        perm = rng.permutation(len(traversable))
        return [traversable[int(idx)] for idx in perm[:n_tiles].tolist()]
    if aliasing_type in {"stripes", "repeating_stripes"}:
        return [(row, col) for row, col in traversable if row % max(spacing, 1) == 0]
    if aliasing_type == "checkerboard":
        return [(row, col) for row, col in traversable if ((row // max(spacing, 1)) + (col // max(spacing, 1))) % 2 == 0]
    if aliasing_type == "symmetry":
        half = [(row, col) for row, col in traversable if col <= width // 2 and row % max(spacing, 1) == 0]
        mirrored = {(row, width - 1 - col) for row, col in half if (row, width - 1 - col) in traversable_set}
        return list(dict.fromkeys(half + list(mirrored)))
    if aliasing_type == "boundary_only":
        return _boundary_coords(traversable, env_shape)[:: max(spacing // 2, 1)]
    if aliasing_type == "center_only":
        sorted_coords = _center_sorted_coords(traversable, env_shape)
        return sorted_coords[:1]
    if aliasing_type == "gradient":
        return traversable[:: max(spacing // 2, 1)]
    raise ValueError(f"Unsupported aliasing type `{aliasing_type}`.")


def generate_tile_pattern(
    env_shape: tuple[int, int],
    aliasing_level: str,
    aliasing_type: str,
    seed: int = 42,
    colors: Iterable[str] = DEFAULT_TILE_COLORS,
    *,
    tile_period: int | None = None,
    landmark_density: float | None = None,
    wall_entropy: float = 0.0,
    gradient_weight: float = 0.0,
    landmark_mode: str = "mixed",
    mask: np.ndarray | None = None,
) -> list[tuple[int, int, str]]:
    """Generate floor-tile anchors for discrete environments.

    Backward-compatible with the earlier `(aliasing_level, aliasing_type)` interface,
    while also supporting explicit `(tau, lambda, H, omega)` controls.
    """
    aliasing_type = str(aliasing_type).lower()
    colors = tuple(colors)
    if not colors:
        raise ValueError("At least one tile color must be provided.")

    spacing = int(tile_period) if tile_period is not None else _grid_spacing(aliasing_level)
    density = float(landmark_density if landmark_density is not None else 1.0 / max(spacing ** 2, 1))
    rng = np.random.default_rng(seed)
    traversable = _traversable_coords(env_shape, mask)
    if not traversable:
        return []

    coords = _base_coords(env_shape, aliasing_type, spacing, rng=rng, mask=mask)
    target_count = max(0, int(math.ceil(len(traversable) * max(density, 0.0))))

    if target_count and len(coords) > target_count:
        keep = rng.permutation(len(coords))[:target_count]
        coords = [coords[int(idx)] for idx in keep.tolist()]
    elif target_count and len(coords) < target_count:
        available = [coord for coord in traversable if coord not in set(coords)]
        if available:
            extra_count = min(target_count - len(coords), len(available))
            extra_idx = rng.permutation(len(available))[:extra_count]
            coords.extend([available[int(idx)] for idx in extra_idx.tolist()])

    if landmark_mode == "boundary_only":
        coords = _boundary_coords(coords or traversable, env_shape)
    elif landmark_mode == "center_only":
        coords = _center_sorted_coords(coords or traversable, env_shape)[:1]
    elif landmark_mode == "sparse_random":
        target_count = max(1, int(math.ceil(len(traversable) * max(density, 0.02))))
        perm = rng.permutation(len(traversable))[:target_count]
        coords = [traversable[int(idx)] for idx in perm.tolist()]

    if wall_entropy > 0.0:
        boundary = _boundary_coords(traversable, env_shape)
        n_boundary = min(len(boundary), int(math.ceil(len(traversable) * float(wall_entropy) * 0.2)))
        if n_boundary > 0:
            perm = rng.permutation(len(boundary))[:n_boundary]
            coords.extend([boundary[int(idx)] for idx in perm.tolist()])

    coords = list(dict.fromkeys(coords))
    tiles: list[tuple[int, int, str]] = []
    for idx, (row, col) in enumerate(coords):
        if gradient_weight > 0.0 and len(colors) > 1:
            score = (row / max(env_shape[0] - 1, 1) + col / max(env_shape[1] - 1, 1)) / 2.0
            weighted = float((1.0 - gradient_weight) * (idx / max(len(coords) - 1, 1)) + gradient_weight * score)
            color_idx = min(int(round(weighted * (len(colors) - 1))), len(colors) - 1)
        else:
            color_idx = idx % len(colors)
        tiles.append((int(col), int(row), colors[color_idx]))
    return tiles


def compute_aliasing_score(env, n_samples: int = 500) -> float:
    """Estimate observation aliasing by comparing randomly separated samples."""
    rollout = env.sample_rollout(max(n_samples * 2, 32), seed=env.seed + 17)
    observations = np.asarray(rollout.observations, dtype=np.float32)
    positions = np.asarray(rollout.positions, dtype=np.float32)
    if len(observations) < 2:
        return 0.0

    norms = np.linalg.norm(observations, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    observations = observations / norms

    rng = np.random.default_rng(env.seed + 123)
    similarities: list[float] = []
    for _ in range(n_samples):
        idx_a, idx_b = rng.integers(0, len(observations), size=2)
        if np.linalg.norm(positions[idx_a] - positions[idx_b]) <= 3.0:
            continue
        similarities.append(float(np.dot(observations[idx_a], observations[idx_b])))
    if not similarities:
        return 0.0
    return float(np.mean(similarities))


def compute_geo_euclidean_discrepancy(env, n_pairs: int = 2000) -> dict[str, float]:
    """Compare graph geodesic distances to Euclidean distances across random pairs."""
    rollout = env.sample_rollout(max(n_pairs * 2, 64), seed=env.seed + 29)
    positions = np.asarray(rollout.positions, dtype=np.float32)
    if len(positions) < 2:
        return {"mean": 1.0, "std": 0.0, "p95": 1.0}

    rng = np.random.default_rng(env.seed + 999)
    ratios: list[float] = []
    for _ in range(n_pairs):
        idx_a, idx_b = rng.integers(0, len(positions), size=2)
        if idx_a == idx_b:
            continue
        euclidean = float(np.linalg.norm(positions[idx_a] - positions[idx_b]))
        if euclidean <= 1e-6:
            continue
        geodesic = env.geodesic_distance(positions[idx_a], positions[idx_b])
        if not np.isfinite(geodesic):
            continue
        ratios.append(geodesic / euclidean)
    if not ratios:
        return {"mean": 1.0, "std": 0.0, "p95": 1.0}

    ratios_np = np.asarray(ratios, dtype=np.float32)
    return {
        "mean": float(np.mean(ratios_np)),
        "std": float(np.std(ratios_np)),
        "p95": float(np.percentile(ratios_np, 95)),
    }

