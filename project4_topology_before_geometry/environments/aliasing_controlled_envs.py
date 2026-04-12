"""MiniGrid-only aliasing-controlled environment library for geometry x aliasing studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import warnings

import networkx as nx
import numpy as np

from project4_topology_before_geometry.environments.base_env import (
    BaseTopologyEnv,
    RolloutBatch,
    compute_complexity_index_from_mask,
)
from project4_topology_before_geometry.environments.topology_labels import get_topology_label
from project4_topology_before_geometry.sensory.aliasing_control import (
    alias_level_from_params,
    generate_tile_pattern,
    preset_to_params,
)

try:  # pragma: no cover
    from minigrid.core.mission import MissionSpace
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Floor, Wall

    MINIGRID_AVAILABLE = True
except ImportError:  # pragma: no cover
    try:
        from gym_minigrid.minigrid import Floor, Grid, MiniGridEnv, Wall

        MissionSpace = None
        MINIGRID_AVAILABLE = True
    except ImportError:  # pragma: no cover
        MissionSpace = None
        MiniGridEnv = object  # type: ignore[assignment]
        Grid = object  # type: ignore[assignment]
        Floor = object  # type: ignore[assignment]
        Wall = object  # type: ignore[assignment]
        MINIGRID_AVAILABLE = False


_COLOR_TO_SCALE = np.array([10.0, 5.0, 3.0], dtype=np.float32)
_CANONICAL_RE = re.compile(
    r"^(?P<geometry>.+?)(?:_(?:τ|tau)=(?P<tau>[^_]+)_(?:λ|lambda)=(?P<lambda>[^_]+)_H=(?P<H>[^_]+)_(?:ω|omega)=(?P<omega>[^_]+))$"
)
_ALIAS_SUFFIXES = ("zero_alias", "low_alias", "medium_alias", "high_alias", "maximum_alias")
_BASE_PRESET_NAMES = set(_ALIAS_SUFFIXES)
_GEOMETRY_NAMES = {
    "square",
    "rectangle",
    "circle_approx",
    "l_shape",
    "u_shape",
    "corridor",
    "t_maze",
    "plus_maze",
    "two_room",
    "three_room",
    "four_room",
    "bottleneck_room",
    "maze_simple",
    "maze_medium",
    "loop_corridor",
    "spiral_maze",
    "dead_end_maze",
    "branching_tree",
    "figure_8",
    "double_loop",
    "nested_rooms",
    "room_with_island",
}
_SPECIAL_ENVS: dict[str, dict[str, object]] = {
    "uniform_box": {"geometry": "square", "alias_preset": "maximum_alias", "aliasing_type": "uniform"},
    "repeating_stripes": {"geometry": "square", "alias_preset": "high_alias", "aliasing_type": "repeating_stripes"},
    "checkerboard_large_period": {"geometry": "square", "alias_preset": "medium_alias", "aliasing_type": "checkerboard", "tile_period": 6},
    "symmetry_trap": {"geometry": "plus_maze", "alias_preset": "maximum_alias", "aliasing_type": "symmetry"},
    "long_corridor_alias": {"geometry": "corridor", "alias_preset": "maximum_alias", "aliasing_type": "uniform"},
    "ambiguous_junctions": {"geometry": "plus_maze", "alias_preset": "high_alias", "aliasing_type": "symmetry"},
    "perceptual_alias_maze": {"geometry": "maze_simple", "alias_preset": "high_alias", "aliasing_type": "periodic"},
    "no_landmarks": {"geometry": "square", "alias_preset": "maximum_alias", "aliasing_type": "uniform"},
    "boundary_only_landmarks": {"geometry": "square", "alias_preset": "low_alias", "aliasing_type": "boundary_only", "landmark_mode": "boundary_only"},
    "center_only_landmark": {"geometry": "square", "alias_preset": "low_alias", "aliasing_type": "center_only", "landmark_mode": "center_only"},
    "sparse_random_landmarks": {"geometry": "square", "alias_preset": "low_alias", "aliasing_type": "sparse_random", "landmark_mode": "sparse_random"},
    "square_low_alias": {"geometry": "square", "alias_preset": "low_alias"},
    "square_high_alias": {"geometry": "square", "alias_preset": "high_alias"},
    "two_room_low_alias": {"geometry": "two_room", "alias_preset": "low_alias"},
    "two_room_high_alias": {"geometry": "two_room", "alias_preset": "high_alias"},
    "l_shape_low_alias": {"geometry": "l_shape", "alias_preset": "low_alias"},
    "l_shape_high_alias": {"geometry": "l_shape", "alias_preset": "high_alias"},
    "maze_low_alias": {"geometry": "maze_simple", "alias_preset": "low_alias"},
    "maze_high_alias": {"geometry": "maze_simple", "alias_preset": "high_alias"},
}


def _normalize_obs_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return (image / _COLOR_TO_SCALE[None, None, :]).reshape(-1)


@dataclass(frozen=True)
class AliasingEnvSpec:
    requested_name: str
    geometry_name: str
    mask: np.ndarray
    width: int
    height: int
    topology_label: dict[str, int]
    aliasing_level: str
    aliasing_type: str
    tile_period: int
    landmark_density: float
    wall_entropy: float
    gradient_weight: float
    landmark_mode: str
    floor_tiles: tuple[tuple[int, int, str], ...]
    canonical_name: str
    canonical_display_name: str
    layout_seed: int = 42
    arena_scale: float = 1.0
    complexity_hint: float | None = None


def _blank(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def _fill(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    mask[top : top + height, left : left + width] = True


def _carve(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    mask[top : top + height, left : left + width] = False


def _full(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _l_shape_mask(size: int = 18) -> np.ndarray:
    mask = _full(size, size)
    cut = size // 2
    mask[:cut, cut:] = False
    return mask


def _u_shape_mask(size: int = 18) -> np.ndarray:
    mask = _blank(size, size)
    _fill(mask, 1, 1, size - 2, 3)
    _fill(mask, 1, size - 4, size - 2, 3)
    _fill(mask, 1, 1, 3, size - 2)
    return mask


def _corridor_mask(height: int = 7, width: int = 30) -> np.ndarray:
    mask = _blank(height, width)
    _fill(mask, 2, 1, 3, width - 2)
    return mask


def _t_maze_mask(height: int = 21, width: int = 21) -> np.ndarray:
    mask = _blank(height, width)
    _fill(mask, 1, width // 2 - 1, height - 2, 3)
    _fill(mask, height - 5, 2, 3, width - 4)
    return mask


def _plus_maze_mask(size: int = 21) -> np.ndarray:
    mask = _blank(size, size)
    _fill(mask, 1, size // 2 - 2, size - 2, 5)
    _fill(mask, size // 2 - 2, 1, 5, size - 2)
    return mask


def _circle_mask(size: int = 18) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    radius = size / 2.15
    return (((xx - center) ** 2 + (yy - center) ** 2) <= radius ** 2).astype(bool)


def _rectangle_mask(height: int = 14, width: int = 28) -> np.ndarray:
    return _full(height, width)


def _two_room_mask(height: int = 18, width: int = 18, door_width: int = 3) -> np.ndarray:
    mask = _full(height, width)
    wall_col = width // 2
    mask[:, wall_col] = False
    start = height // 2 - door_width // 2
    mask[start : start + door_width, wall_col] = True
    return mask


def _three_room_mask(height: int = 18, width: int = 24) -> np.ndarray:
    mask = _full(height, width)
    cols = (width // 3, 2 * width // 3)
    for idx, wall_col in enumerate(cols):
        mask[:, wall_col] = False
        gap_row = 4 if idx == 0 else height - 7
        mask[gap_row : gap_row + 4, wall_col] = True
    return mask


def _four_room_mask(size: int = 19) -> np.ndarray:
    mask = _full(size, size)
    mid = size // 2
    mask[:, mid] = False
    mask[mid, :] = False
    for gap in (3, size - 4):
        mask[gap : gap + 2, mid] = True
        mask[mid, gap : gap + 2] = True
    return mask


def _bottleneck_room_mask(height: int = 18, width: int = 18) -> np.ndarray:
    mask = _full(height, width)
    wall_col = width // 2
    mask[:, wall_col] = False
    mask[height // 2, wall_col] = True
    return mask


def _maze_simple_mask(size: int = 18) -> np.ndarray:
    mask = _full(size, size)
    walls = [
        (slice(2, size - 2), 4),
        (slice(1, size - 5), 8),
        (slice(5, size - 1), 12),
    ]
    for rows, col in walls:
        mask[rows, col] = False
    mask[4:7, 4] = True
    mask[10:13, 8] = True
    mask[6:9, 12] = True
    return mask


def _maze_medium_mask(size: int = 24) -> np.ndarray:
    mask = _full(size, size)
    for col in (4, 8, 12, 16, 20):
        mask[2 : size - 2, col] = False
    for row in (4, 8, 12, 16, 20):
        mask[row, 2 : size - 2] = False
    gaps = [
        (4, 5), (8, 10), (12, 15), (16, 6), (20, 18),
        (6, 4), (10, 8), (15, 12), (18, 16), (5, 20),
    ]
    for row, col in gaps:
        mask[row : row + 2, col : col + 2] = True
    return mask


def _loop_corridor_mask(size: int = 20) -> np.ndarray:
    mask = _full(size, size)
    _carve(mask, 5, 5, size - 10, size - 10)
    return mask


def _spiral_maze_mask(size: int = 21) -> np.ndarray:
    mask = _blank(size, size)
    top, left, bottom, right = 1, 1, size - 2, size - 2
    while top <= bottom and left <= right:
        _fill(mask, top, left, 2, right - left + 1)
        _fill(mask, top, max(right - 1, left), bottom - top + 1, 2)
        if bottom - top > 3:
            _fill(mask, max(bottom - 1, top), left + 2, 2, right - left - 1)
        if right - left > 3:
            _fill(mask, top + 2, left, bottom - top - 1, 2)
        top += 4
        left += 4
        bottom -= 4
        right -= 4
    return mask


def _dead_end_maze_mask(height: int = 18, width: int = 22) -> np.ndarray:
    mask = _blank(height, width)
    _fill(mask, height // 2 - 1, 1, 3, width - 2)
    for col in (4, 8, 12, 16, 19):
        _fill(mask, height // 2 - 1, col, height // 2 - 2, 3)
    return mask


def _branching_tree_mask(height: int = 21, width: int = 23) -> np.ndarray:
    mask = _blank(height, width)
    _fill(mask, 1, width // 2 - 1, height - 2, 3)
    branches = [
        (5, width // 2 - 7, 3, 7),
        (10, width // 2, 3, 7),
        (15, width // 2 - 9, 3, 9),
    ]
    for row, left, branch_h, branch_w in branches:
        _fill(mask, row, left, branch_h, branch_w)
    return mask


def _room_with_island_mask(size: int = 20) -> np.ndarray:
    mask = _full(size, size)
    _carve(mask, size // 2 - 3, size // 2 - 3, 6, 6)
    return mask


def _figure8_mask(height: int = 22, width: int = 26) -> np.ndarray:
    mask = _full(height, width)
    _carve(mask, 5, 4, 6, 6)
    _carve(mask, 5, width - 10, 6, 6)
    _carve(mask, 12, width // 2 - 3, 5, 6)
    return mask


def _double_loop_mask(height: int = 20, width: int = 28) -> np.ndarray:
    mask = _full(height, width)
    _carve(mask, 5, 4, 6, 6)
    _carve(mask, 5, width - 10, 6, 6)
    return mask


def _nested_rooms_mask(size: int = 22) -> np.ndarray:
    mask = _full(size, size)
    _carve(mask, 4, 4, 5, 5)
    _carve(mask, 13, 13, 5, 5)
    return mask


def _legacy_shape_mask(name: str) -> np.ndarray:
    if name == "square":
        return _full(18, 18)
    if name == "rectangle":
        return _rectangle_mask()
    if name == "circle_approx":
        return _circle_mask()
    if name == "l_shape":
        return _l_shape_mask()
    if name == "u_shape":
        return _u_shape_mask()
    if name == "corridor":
        return _corridor_mask()
    if name == "t_maze":
        return _t_maze_mask()
    if name == "plus_maze":
        return _plus_maze_mask()
    if name == "two_room":
        return _two_room_mask()
    if name == "three_room":
        return _three_room_mask()
    if name == "four_room":
        return _four_room_mask()
    if name == "bottleneck_room":
        return _bottleneck_room_mask()
    if name == "maze_simple":
        return _maze_simple_mask()
    if name == "maze_medium":
        return _maze_medium_mask()
    if name == "loop_corridor":
        return _loop_corridor_mask()
    if name == "spiral_maze":
        return _spiral_maze_mask()
    if name == "dead_end_maze":
        return _dead_end_maze_mask()
    if name == "branching_tree":
        return _branching_tree_mask()
    if name == "figure_8":
        return _figure8_mask()
    if name == "double_loop":
        return _double_loop_mask()
    if name == "nested_rooms":
        return _nested_rooms_mask()
    if name == "room_with_island":
        return _room_with_island_mask()
    raise KeyError(f"Unknown geometry `{name}`.")


def _validate_mask(mask: np.ndarray, env_name: str) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError(f"`{env_name}` generated an empty or invalid traversable mask.")

    graph = nx.Graph()
    rows, cols = np.where(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        graph.add_node((row, col))
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < mask.shape[0] and 0 <= n_col < mask.shape[1] and mask[n_row, n_col]:
                graph.add_edge((row, col), (n_row, n_col))
    if nx.number_connected_components(graph) != 1:
        raise ValueError(f"`{env_name}` produced disconnected traversable regions, which are invalid for this suite.")


def _detect_symmetry_axes(mask: np.ndarray) -> list[str]:
    axes: list[str] = []
    if np.array_equal(mask, np.flip(mask, axis=1)):
        axes.append("vertical")
    if np.array_equal(mask, np.flip(mask, axis=0)):
        axes.append("horizontal")
    if np.array_equal(mask, np.flip(np.flip(mask, axis=0), axis=1)):
        axes.append("point")
    return axes


def _canonical_name(geometry: str, tile_period: int, landmark_density: float, wall_entropy: float, gradient_weight: float) -> str:
    return (
        f"{geometry}_tau={int(tile_period)}"
        f"_lambda={landmark_density:.3g}"
        f"_H={wall_entropy:.3g}"
        f"_omega={gradient_weight:.3g}"
    )


def _canonical_display_name(geometry: str, tile_period: int, landmark_density: float, wall_entropy: float, gradient_weight: float) -> str:
    return (
        f"{geometry}_τ={int(tile_period)}"
        f"_λ={landmark_density:.3g}"
        f"_H={wall_entropy:.3g}"
        f"_ω={gradient_weight:.3g}"
    )


def _resolve_request(env_type, **kwargs) -> dict[str, object]:
    if isinstance(env_type, dict):
        request = dict(env_type)
    else:
        request = {"env_type": str(env_type)}
    request.update(kwargs)

    env_name = str(request.get("env_type") or request.get("name") or request.get("geometry") or "square")
    if env_name in _SPECIAL_ENVS:
        merged = dict(_SPECIAL_ENVS[env_name])
        merged.update({key: value for key, value in request.items() if key != "env_type"})
        merged["requested_name"] = env_name
        return merged

    canonical_match = _CANONICAL_RE.match(env_name)
    if canonical_match:
        geometry_name = canonical_match.group("geometry")
        resolved = {
            "requested_name": env_name,
            "geometry": geometry_name,
            "tile_period": int(float(canonical_match.group("tau"))),
            "landmark_density": float(canonical_match.group("lambda")),
            "wall_entropy": float(canonical_match.group("H")),
            "gradient_weight": float(canonical_match.group("omega")),
            "aliasing_type": str(request.get("aliasing_type", "periodic")),
            "landmark_mode": str(request.get("landmark_mode", "mixed")),
        }
        return resolved

    if env_name in _BASE_PRESET_NAMES:
        resolved = {"requested_name": env_name, "geometry": "square", "alias_preset": env_name}
        resolved.update({key: value for key, value in request.items() if key != "env_type"})
        return resolved

    for alias_suffix in _ALIAS_SUFFIXES:
        token = f"_{alias_suffix}"
        if env_name.endswith(token):
            geometry_name = env_name[: -len(token)]
            if geometry_name in _GEOMETRY_NAMES:
                resolved = {"requested_name": env_name, "geometry": geometry_name, "alias_preset": alias_suffix}
                resolved.update({key: value for key, value in request.items() if key != "env_type"})
                return resolved

    if env_name in _GEOMETRY_NAMES:
        resolved = {"requested_name": env_name, "geometry": env_name, "alias_preset": "low_alias"}
        resolved.update({key: value for key, value in request.items() if key != "env_type"})
        return resolved

    if "geometry" in request:
        resolved = {"requested_name": env_name, **request}
        return resolved

    raise KeyError(f"Unknown aliasing-controlled environment request `{env_name}`.")


def build_layout(layout_type: str) -> tuple[np.ndarray, np.ndarray, nx.Graph]:
    """Construct a MiniGrid-compatible wall map, free-space mask, and connectivity graph."""
    free_space = _legacy_shape_mask(layout_type)
    _validate_mask(free_space, layout_type)
    wall_map = ~free_space
    graph = nx.Graph()
    rows, cols = np.where(free_space)
    for row, col in zip(rows.tolist(), cols.tolist()):
        graph.add_node((row, col))
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < free_space.shape[0] and 0 <= n_col < free_space.shape[1] and free_space[n_row, n_col]:
                graph.add_edge((row, col), (n_row, n_col))
    return wall_map.astype(bool), free_space.astype(bool), graph


def _build_spec(env_type, seed: int = 42, **kwargs) -> AliasingEnvSpec:
    resolved = _resolve_request(env_type, **kwargs)
    geometry_name = str(resolved.get("geometry", "square"))
    wall_map, free_space, _ = build_layout(geometry_name)
    alias_preset = resolved.get("alias_preset")
    params = preset_to_params(str(alias_preset)) if alias_preset is not None else {}
    params.update({key: value for key, value in resolved.items() if key in {"tile_period", "landmark_density", "wall_entropy", "gradient_weight", "aliasing_type"}})

    tile_period = int(params.get("tile_period", 4))
    landmark_density = float(params.get("landmark_density", 0.1))
    wall_entropy = float(params.get("wall_entropy", 0.1))
    gradient_weight = float(params.get("gradient_weight", 0.0))
    aliasing_type = str(params.get("aliasing_type", "periodic"))
    landmark_mode = str(resolved.get("landmark_mode", "mixed"))
    aliasing_level = alias_level_from_params(tile_period, landmark_density, gradient_weight)
    floor_tiles = tuple(
        generate_tile_pattern(
            env_shape=free_space.shape,
            aliasing_level=aliasing_level,
            aliasing_type=aliasing_type,
            seed=seed,
            tile_period=tile_period,
            landmark_density=landmark_density,
            wall_entropy=wall_entropy,
            gradient_weight=gradient_weight,
            landmark_mode=landmark_mode,
            mask=free_space,
        )
    )
    requested_name = str(resolved.get("requested_name", geometry_name))
    symmetry_axes = _detect_symmetry_axes(free_space)
    if symmetry_axes and landmark_density <= 0.0 and gradient_weight <= 0.0:
        warnings.warn(
            f"`{requested_name}` is symmetric across {symmetry_axes} without disambiguating cues; expect high aliasing.",
            stacklevel=2,
        )

    return AliasingEnvSpec(
        requested_name=requested_name,
        geometry_name=geometry_name,
        mask=free_space,
        width=int(free_space.shape[1]),
        height=int(free_space.shape[0]),
        topology_label=get_topology_label(geometry_name),
        aliasing_level=aliasing_level,
        aliasing_type=aliasing_type,
        tile_period=tile_period,
        landmark_density=landmark_density,
        wall_entropy=wall_entropy,
        gradient_weight=gradient_weight,
        landmark_mode=landmark_mode,
        floor_tiles=floor_tiles,
        canonical_name=_canonical_name(geometry_name, tile_period, landmark_density, wall_entropy, gradient_weight),
        canonical_display_name=_canonical_display_name(geometry_name, tile_period, landmark_density, wall_entropy, gradient_weight),
        layout_seed=seed,
        complexity_hint=compute_complexity_index_from_mask(free_space),
    )


class AliasingControlledMiniGridCore(MiniGridEnv):
    """MiniGrid core that keeps geometry fixed while re-sampling trajectories."""

    def __init__(self, spec: AliasingEnvSpec, agent_seed: int = 42):
        if not MINIGRID_AVAILABLE:  # pragma: no cover
            raise ImportError("MiniGrid is not installed.")
        self.spec = spec
        self._agent_rng = np.random.default_rng(agent_seed)
        if MissionSpace is not None:
            super().__init__(
                mission_space=MissionSpace(mission_func=lambda: "aliasing_controlled_navigation"),
                width=spec.width + 2,
                height=spec.height + 2,
                see_through_walls=False,
                max_steps=10_000,
                agent_view_size=7,
            )
        else:  # pragma: no cover - older gym-minigrid fallback
            super().__init__(width=spec.width + 2, height=spec.height + 2, max_steps=10_000, agent_view_size=7)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        mask = self.spec.mask
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row, col]:
                    self.put_obj(Floor("grey"), col + 1, row + 1)
                else:
                    self.put_obj(Wall(), col + 1, row + 1)

        for col, row, color in self.spec.floor_tiles:
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col]:
                self.put_obj(Floor(color), int(col) + 1, int(row) + 1)

        traversable = np.argwhere(mask)
        start_idx = int(self._agent_rng.integers(0, len(traversable)))
        row, col = traversable[start_idx]
        self.agent_pos = np.array([col + 1, row + 1], dtype=int)
        self.agent_dir = int(self._agent_rng.integers(0, 4))
        self.mission = self.spec.canonical_name


class AliasingControlledEnv(BaseTopologyEnv):
    """MiniGrid-backed environment with explicit geometry x aliasing controls."""

    backend = "minigrid"

    def __init__(self, spec: AliasingEnvSpec, seed: int = 42):
        super().__init__(
            spec.requested_name,
            obs_dim=147,
            act_dim=5,
            topology_label=spec.topology_label,
            aliasing_level=spec.aliasing_level,
            aliasing_type=spec.aliasing_type,
            arena_scale=spec.arena_scale,
            seed=seed,
        )
        self.spec = spec
        self.geometry_name = spec.geometry_name
        self.canonical_name = spec.canonical_name
        self.canonical_display_name = spec.canonical_display_name
        self.name = spec.canonical_name
        self.tile_period = spec.tile_period
        self.landmark_density = spec.landmark_density
        self.wall_entropy = spec.wall_entropy
        self.gradient_weight = spec.gradient_weight
        self.landmark_mode = spec.landmark_mode
        self.complexity_index = float(spec.complexity_hint or compute_complexity_index_from_mask(spec.mask))

    @property
    def traversable_mask(self) -> np.ndarray:
        return self.spec.mask

    @property
    def coordinate_extent(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, float(self.spec.width - 1), float(self.spec.height - 1))

    def build_layout(self, layout_type: str):
        return build_layout(layout_type)

    def make_env(self, seed: int | None = None) -> AliasingControlledMiniGridCore:
        return AliasingControlledMiniGridCore(self.spec, agent_seed=self.seed if seed is None else int(seed))

    def sample_rollout(self, n_steps: int, seed: int | None = None) -> RolloutBatch:
        env = self.make_env(seed=seed)
        reset_out = env.reset(seed=self.seed if seed is None else int(seed))
        observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        observations = [_normalize_obs_image(observation["image"])]
        positions = [np.asarray(env.agent_pos, dtype=np.float32) - 1.0]
        headings = [float(env.agent_dir)]
        raw_actions: list[int] = []
        action_values = [
            int(env.actions.left),
            int(env.actions.right),
            int(env.actions.forward),
            int(env.actions.done),
        ]
        action_probs = np.array([0.15, 0.15, 0.60, 0.10], dtype=np.float64)
        rng = np.random.default_rng(self.seed if seed is None else int(seed))

        for _ in range(int(n_steps)):
            action = int(rng.choice(action_values, p=action_probs))
            step_out = env.step(action)
            if len(step_out) == 5:
                observation, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:  # pragma: no cover
                observation, _, done, _ = step_out
            if done:
                reset_out = env.reset()
                observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            raw_actions.append(action)
            observations.append(_normalize_obs_image(observation["image"]))
            positions.append(np.asarray(env.agent_pos, dtype=np.float32) - 1.0)
            headings.append(float(env.agent_dir))

        return RolloutBatch(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(raw_actions, dtype=np.int64),
            positions=np.asarray(positions, dtype=np.float32),
            headings=np.asarray(headings, dtype=np.float32),
            metadata={
                "backend": self.backend,
                "geometry": self.geometry_name,
                "canonical_name": self.canonical_name,
                "canonical_display_name": self.canonical_display_name,
                "tile_period": self.tile_period,
                "landmark_density": self.landmark_density,
                "wall_entropy": self.wall_entropy,
                "gradient_weight": self.gradient_weight,
            },
        )


def is_aliasing_controlled_name(env_name) -> bool:
    """Return True when the request should route through the MiniGrid-only aliasing library."""
    if isinstance(env_name, dict):
        return True
    name = str(env_name)
    if name in _SPECIAL_ENVS or name in _BASE_PRESET_NAMES or name in _GEOMETRY_NAMES:
        return True
    if _CANONICAL_RE.match(name):
        return True
    return any(name.endswith(f"_{suffix}") and name[: -len(suffix) - 1] in _GEOMETRY_NAMES for suffix in _ALIAS_SUFFIXES)


def list_prebuilt_environments() -> list[str]:
    """Return the prebuilt MiniGrid-only benchmarking library."""
    combined = [
        f"{geometry}_{alias_suffix}"
        for geometry in ("two_room", "l_shape", "maze_simple")
        for alias_suffix in ("low_alias", "high_alias")
    ]
    return sorted(set(_BASE_PRESET_NAMES) | _GEOMETRY_NAMES | set(_SPECIAL_ENVS) | set(combined))


def make_environment(env_type, *, seed: int = 42, **kwargs) -> AliasingControlledEnv:
    """Instantiate one aliasing-controlled MiniGrid environment from a string or dict spec."""
    spec = _build_spec(env_type, seed=seed, **kwargs)
    return AliasingControlledEnv(spec, seed=seed)
