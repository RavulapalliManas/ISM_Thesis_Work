"""MiniGrid-backed environments matching the paper's discrete observation setup."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from project4_topology_before_geometry.environments.base_env import (
    BaseTopologyEnv,
    RolloutBatch,
    compute_complexity_index_from_mask,
)
from project4_topology_before_geometry.environments.topology_labels import TOPOLOGY_LABELS
from project4_topology_before_geometry.sensory.aliasing_control import generate_tile_pattern

try:  # pragma: no cover - dependency is optional at runtime
    from minigrid.core.mission import MissionSpace
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Floor, Wall

    MINIGRID_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for older package names
    try:
        from gym_minigrid.minigrid import Floor, Grid, MiniGridEnv, Wall

        MINIGRID_AVAILABLE = True
    except ImportError:
        MiniGridEnv = object  # type: ignore[assignment]
        Grid = object  # type: ignore[assignment]
        Floor = object  # type: ignore[assignment]
        Wall = object  # type: ignore[assignment]
        MINIGRID_AVAILABLE = False


_COLOR_TO_SCALE = np.array([10.0, 5.0, 3.0], dtype=np.float32)


@dataclass(frozen=True)
class DiscreteShapeSpec:
    """Static description of one grid-based environment."""

    width: int
    height: int
    mask: np.ndarray
    aliasing_level: str
    aliasing_type: str
    arena_scale: float = 1.0
    complexity_hint: float | None = None


def _full_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _l_shape_mask(size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    cut = size // 2
    mask[:cut, cut:] = False
    return mask


def _circle_mask(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    radius = size / 2.15
    return ((xx - center) ** 2 + (yy - center) ** 2) <= radius ** 2


def _triangle_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for row in range(height):
        max_col = int(np.floor((row + 1) * width / height))
        mask[row, : max(max_col, 1)] = True
    return mask


def _two_room_corridor_mask(height: int = 18, width: int = 18) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, :8] = True
    mask[:, 10:] = True
    mid = height // 2
    mask[mid - 2 : mid + 2, 8:10] = True
    return mask


def _hairpin_mask(height: int = 18, width: int = 12) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    corridor_width = 2
    columns = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    for idx, (start, end) in enumerate(columns):
        if idx % 2 == 0:
            mask[1:16, start:end] = True
        else:
            mask[2:17, start:end] = True
        if idx < len(columns) - 1:
            connector_rows = slice(14, 16) if idx % 2 == 0 else slice(2, 4)
            mask[connector_rows, end : end + corridor_width] = True
    return mask


def _normalize_obs_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return (image / _COLOR_TO_SCALE[None, None, :]).reshape(-1)


class TopologyMiniGridCore(MiniGridEnv):
    """Custom MiniGrid environment whose traversable geometry is defined by a boolean mask."""

    def __init__(self, spec: DiscreteShapeSpec, seed: int = 42):
        if not MINIGRID_AVAILABLE:  # pragma: no cover - validated before construction
            raise ImportError("MiniGrid is not installed.")
        self.spec = spec
        self._rng = np.random.default_rng(seed)
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "topology_before_geometry"),
            width=spec.width + 2,
            height=spec.height + 2,
            see_through_walls=False,
            max_steps=10_000,
            agent_view_size=7,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(0, 0)
        mask = self.spec.mask
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row, col]:
                    self.put_obj(Floor("grey"), col + 1, row + 1)
                else:
                    self.put_obj(Wall(), col + 1, row + 1)

        for col, row, color in generate_tile_pattern(
            env_shape=(mask.shape[0], mask.shape[1]),
            aliasing_level=self.spec.aliasing_level,
            aliasing_type=self.spec.aliasing_type,
            seed=int(self._rng.integers(1, 1_000_000)),
        ):
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col]:
                self.put_obj(Floor(color), col + 1, row + 1)

        traversable = np.argwhere(mask)
        start_idx = int(self._rng.integers(0, len(traversable)))
        row, col = traversable[start_idx]
        self.agent_pos = np.array([col + 1, row + 1], dtype=int)
        self.agent_dir = int(self._rng.integers(0, 4))
        self.mission = "topology_before_geometry"


class MiniGridTopologyEnv(BaseTopologyEnv):
    """Discrete 7x7x3-observation environment that matches the paper's primary backend."""

    backend = "minigrid"

    def __init__(self, env_name: str, spec: DiscreteShapeSpec, seed: int = 42):
        super().__init__(
            env_name,
            obs_dim=147,
            act_dim=5,
            topology_label=TOPOLOGY_LABELS[env_name],
            aliasing_level=spec.aliasing_level,
            aliasing_type=spec.aliasing_type,
            arena_scale=spec.arena_scale,
            seed=seed,
        )
        self.spec = spec
        self._complexity_index = (
            float(spec.complexity_hint)
            if spec.complexity_hint is not None
            else compute_complexity_index_from_mask(spec.mask)
        )
        self.complexity_index = self._complexity_index

    @property
    def traversable_mask(self) -> np.ndarray:
        return self.spec.mask

    @property
    def coordinate_extent(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, float(self.spec.width - 1), float(self.spec.height - 1))

    def make_env(self, seed: int | None = None) -> TopologyMiniGridCore:
        return TopologyMiniGridCore(self.spec, seed=self.seed if seed is None else int(seed))

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

        for _ in range(n_steps):
            action = int(rng.choice(action_values, p=action_probs))
            step_out = env.step(action)
            if len(step_out) == 5:
                observation, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
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
            metadata={"backend": self.backend},
        )


def build_minigrid_specs() -> dict[str, DiscreteShapeSpec]:
    """Create the discrete environment registry used by the factory."""
    square = _full_mask(18, 18)
    specs = {
        "square_low_alias": DiscreteShapeSpec(18, 18, square, "low", "periodic", complexity_hint=1.27),
        "square_high_alias": DiscreteShapeSpec(18, 18, square, "high", "sparse_random", complexity_hint=1.27),
        "rectangle_wide": DiscreteShapeSpec(28, 14, _full_mask(14, 28), "medium", "periodic", complexity_hint=1.46),
        "rectangle_narrow": DiscreteShapeSpec(36, 10, _full_mask(10, 36), "medium", "periodic", complexity_hint=1.91),
        "l_shape_standard": DiscreteShapeSpec(18, 18, _l_shape_mask(18), "medium", "periodic", complexity_hint=1.65),
        "l_shape_large": DiscreteShapeSpec(26, 26, _l_shape_mask(26), "medium", "periodic", complexity_hint=1.65, arena_scale=26 / 18),
        "circle_approx": DiscreteShapeSpec(18, 18, _circle_mask(18), "medium", "periodic", complexity_hint=1.00),
        "triangle_approx": DiscreteShapeSpec(18, 18, _triangle_mask(18, 18), "medium", "periodic", complexity_hint=1.45),
        "two_room_corridor": DiscreteShapeSpec(18, 18, _two_room_corridor_mask(), "medium", "clustered", complexity_hint=2.10),
        "hairpin_maze": DiscreteShapeSpec(12, 18, _hairpin_mask(), "high", "periodic", complexity_hint=3.20),
    }
    return specs


def build_minigrid_env(env_name: str, seed: int = 42) -> MiniGridTopologyEnv:
    """Instantiate one discrete environment by name."""
    if not MINIGRID_AVAILABLE:
        raise ImportError("MiniGrid not found. Install `minigrid` or use the RatInABox fallback.")
    specs = build_minigrid_specs()
    if env_name not in specs:
        raise KeyError(f"`{env_name}` is not a registered MiniGrid environment.")
    return MiniGridTopologyEnv(env_name, specs[env_name], seed=seed)


if not MINIGRID_AVAILABLE:  # pragma: no cover - user-facing warning at import time
    warnings.warn("MiniGrid not found. Using RatInABox for all environments.", stacklevel=2)
