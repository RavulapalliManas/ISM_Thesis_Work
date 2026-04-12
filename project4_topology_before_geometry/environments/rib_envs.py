"""RatInABox-backed environments for topologically non-trivial or periodic arenas."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from project3_generalization.environments.suite_2d import EnvironmentSpec2D
from project4_topology_before_geometry.environments.base_env import (
    BaseTopologyEnv,
    RolloutBatch,
    compute_complexity_index_from_mask,
)
from project4_topology_before_geometry.environments.topology_labels import TOPOLOGY_LABELS

try:  # pragma: no cover - optional until runtime
    from ratinabox.Agent import Agent as RIBAgent
    from ratinabox.Environment import Environment as RIBEnvironment

    RATINABOX_AVAILABLE = True
except ImportError:  # pragma: no cover
    RIBAgent = None
    RIBEnvironment = None
    RATINABOX_AVAILABLE = False


DEFAULT_RIB_AGENT_PARAMS: dict[str, float | bool] = {
    "dt": 0.1,
    "speed_mean": 0.2,
    "speed_coherence_time": 0.7,
    "rotational_velocity_coherence_time": 0.08,
    "rotational_velocity_std": float(np.pi / 6.0),
    "thigmotaxis": 0.2,
    "wall_repel_distance": 0.1,
    "save_history": False,
}


@dataclass(frozen=True)
class RIBSpec:
    """Geometry and observation metadata for one RatInABox-backed environment."""

    spec: EnvironmentSpec2D | None
    env_params: dict[str, object]
    polygon: Polygon
    aliasing_level: str
    aliasing_type: str
    object_channels: dict[str, tuple[tuple[float, float], ...]]
    arena_scale: float = 1.0
    complexity_hint: float | None = None
    periodic_x: bool = False


def _polygon_to_boundary_and_holes(geom: Polygon) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[tuple[float, float], ...], ...]]:
    geom = geom.buffer(0)
    boundary = tuple((float(x), float(y)) for x, y in geom.exterior.coords[:-1])
    holes = tuple(
        tuple((float(x), float(y)) for x, y in ring.coords[:-1])
        for ring in geom.interiors
    )
    return boundary, holes


def _make_spec_from_polygon(
    env_id: str,
    geom: Polygon,
    *,
    aliasing_level: str,
    aliasing_type: str,
    object_channels: dict[str, tuple[tuple[float, float], ...]],
    env_params: dict[str, object] | None = None,
    complexity_hint: float | None = None,
    arena_scale: float = 1.0,
    periodic_x: bool = False,
) -> RIBSpec:
    boundary, holes = _polygon_to_boundary_and_holes(geom)
    spec = EnvironmentSpec2D(
        env_id=env_id,
        name=env_id,
        category="project4",
        boundary=boundary,
        holes=holes,
        objects=tuple(point for points in object_channels.values() for point in points),
        expected_betti1=TOPOLOGY_LABELS[env_id]["betti_1"],
    )
    params = {"boundary": [list(p) for p in boundary], "holes": [[list(p) for p in hole] for hole in holes], "dx": 0.02}
    if env_params:
        params.update(env_params)
    return RIBSpec(
        spec=spec,
        env_params=params,
        polygon=geom.buffer(0),
        aliasing_level=aliasing_level,
        aliasing_type=aliasing_type,
        object_channels=object_channels,
        arena_scale=arena_scale,
        complexity_hint=complexity_hint,
        periodic_x=periodic_x,
    )


def _annulus_spec(inner_radius: float = 0.15) -> RIBSpec:
    outer = Point(0.5, 0.5).buffer(0.5, resolution=96)
    inner = Point(0.5, 0.5).buffer(inner_radius, resolution=96)
    geom = outer.difference(inner)
    object_channels = {
        "red": ((0.5, 0.12),),
        "blue": ((0.82, 0.5),),
        "yellow": ((0.5, 0.88),),
    }
    return _make_spec_from_polygon(
        "annulus_approx",
        geom,
        aliasing_level="medium",
        aliasing_type="clustered",
        object_channels=object_channels,
        complexity_hint=1.80,
        env_params={"objects": [list(p) for pts in object_channels.values() for p in pts]},
    )


def _figure8_spec() -> RIBSpec:
    outer = box(0.0, 0.0, 1.4, 1.0)
    hole_left = Point(0.45, 0.5).buffer(0.23, resolution=96)
    hole_right = Point(0.95, 0.5).buffer(0.23, resolution=96)
    corridor = box(0.58, 0.38, 0.82, 0.62)
    geom = outer.difference(unary_union([hole_left, hole_right])).union(corridor).buffer(0)
    object_channels = {
        "red": ((0.2, 0.22),),
        "blue": ((1.2, 0.22),),
        "yellow": ((0.7, 0.82),),
    }
    return _make_spec_from_polygon(
        "figure8_env",
        geom,
        aliasing_level="medium",
        aliasing_type="clustered",
        object_channels=object_channels,
        complexity_hint=2.20,
        env_params={"objects": [list(p) for pts in object_channels.values() for p in pts]},
    )


def _cylinder_spec() -> RIBSpec:
    geom = box(0.0, 0.0, 1.0, 0.5)
    object_channels = {
        "red": ((0.15, 0.15),),
        "blue": ((0.50, 0.35),),
        "yellow": ((0.85, 0.15),),
    }
    return _make_spec_from_polygon(
        "cylinder_env",
        geom,
        aliasing_level="medium",
        aliasing_type="periodic",
        object_channels=object_channels,
        complexity_hint=None,
        env_params={
            "objects": [list(p) for pts in object_channels.values() for p in pts],
            "boundary_conditions": "periodic",
            "aspect": 2.0,
            "scale": 0.5,
            "boundary": None,
        },
        periodic_x=True,
    )


def build_rib_specs(inner_radius: float = 0.15) -> dict[str, RIBSpec]:
    """Create the continuous environment registry used by the factory."""
    return {
        "annulus_approx": _annulus_spec(inner_radius=inner_radius),
        "figure8_env": _figure8_spec(),
        "cylinder_env": _cylinder_spec(),
    }


class RatInABoxTopologyEnv(BaseTopologyEnv):
    """Continuous environment wrapper with a 144-D FoV-style observation model."""

    backend = "ratinabox"

    def __init__(self, env_name: str, spec: RIBSpec, seed: int = 42):
        super().__init__(
            env_name,
            obs_dim=144,
            act_dim=13,
            topology_label=TOPOLOGY_LABELS[env_name],
            aliasing_level=spec.aliasing_level,
            aliasing_type=spec.aliasing_type,
            arena_scale=spec.arena_scale,
            seed=seed,
        )
        self.spec = spec
        self.geometry = spec.polygon
        self._extent = self._resolve_extent()
        self._mask = self._build_mask()
        self.complexity_index = (
            float(spec.complexity_hint)
            if spec.complexity_hint is not None
            else compute_complexity_index_from_mask(self._mask)
        )
        if env_name == "cylinder_env":
            self.complexity_index = float("nan")

    @property
    def traversable_mask(self) -> np.ndarray:
        return self._mask

    @property
    def coordinate_extent(self) -> tuple[float, float, float, float]:
        return self._extent

    def _resolve_extent(self) -> tuple[float, float, float, float]:
        if self.spec.periodic_x:
            return (0.0, 0.0, 1.0, 0.5)
        min_x, min_y, max_x, max_y = self.geometry.bounds
        return (float(min_x), float(min_y), float(max_x), float(max_y))

    def _build_mask(self, resolution: int = 64) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.coordinate_extent
        xs = np.linspace(min_x, max_x, resolution, endpoint=False) + (max_x - min_x) / (2 * resolution)
        ys = np.linspace(min_y, max_y, resolution, endpoint=False) + (max_y - min_y) / (2 * resolution)
        mask = np.zeros((resolution, resolution), dtype=bool)
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                if self.spec.periodic_x:
                    mask[row, col] = 0.0 <= x <= max_x and 0.0 <= y <= max_y
                else:
                    mask[row, col] = self.geometry.buffer(1e-9).contains(Point(float(x), float(y)))
        return mask

    def build_geodesic_graph(self) -> nx.Graph:
        graph = super().build_geodesic_graph()
        if self.spec.periodic_x:
            mask = self.traversable_mask
            step_x = max((self.coordinate_extent[2] - self.coordinate_extent[0]) / max(mask.shape[1] - 1, 1), 1e-6)
            for row in range(mask.shape[0]):
                if mask[row, 0] and mask[row, -1]:
                    graph.add_edge((row, 0), (row, mask.shape[1] - 1), weight=float(step_x))
        return graph

    def _make_env(self):
        if not RATINABOX_AVAILABLE:  # pragma: no cover - validated before construction
            raise ImportError("RatInABox is not installed.")
        return RIBEnvironment(params=dict(self.spec.env_params))

    def _sample_valid_position(self, rng: np.random.Generator) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.coordinate_extent
        for _ in range(10_000):
            candidate = np.array(
                [rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)],
                dtype=np.float32,
            )
            if self.spec.periodic_x or self.geometry.buffer(1e-9).contains(Point(float(candidate[0]), float(candidate[1]))):
                return candidate
        raise RuntimeError("Failed to sample a valid position inside the RatInABox arena.")

    def _fov_observation(self, position: np.ndarray, heading_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approximate the paper's FoV sensor with 48 sector bins and 3 object channels."""
        radii = np.linspace(0.055, 0.33, 6, dtype=np.float32)
        angle_offsets = np.linspace(-np.pi / 4, np.pi / 4, 8, dtype=np.float32)
        heading_angle = float(np.arctan2(heading_vec[1], heading_vec[0]))

        obs = np.zeros((48, 3), dtype=np.float32)
        wall_signal = np.zeros((48,), dtype=np.float32)
        color_order = ("red", "blue", "yellow")

        cell_idx = 0
        for radius in radii:
            for angle_offset in angle_offsets:
                angle = heading_angle + float(angle_offset)
                sample = position + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                if self.spec.periodic_x:
                    sample[0] = sample[0] % self.coordinate_extent[2]
                inside = self.spec.periodic_x or self.geometry.buffer(1e-9).contains(Point(float(sample[0]), float(sample[1])))
                if not inside:
                    wall_signal[cell_idx] = 1.0
                for color_idx, color in enumerate(color_order):
                    anchors = np.asarray(self.spec.object_channels.get(color, ()), dtype=np.float32)
                    if len(anchors) == 0:
                        continue
                    distances = np.linalg.norm(anchors - sample[None, :], axis=1)
                    obs[cell_idx, color_idx] = float(np.exp(-(distances.min() ** 2) / (2 * 0.06 ** 2)))
                cell_idx += 1
        return obs.reshape(-1), wall_signal

    def sample_rollout(self, n_steps: int, seed: int | None = None) -> RolloutBatch:
        env = self._make_env()
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        agent = RIBAgent(env, params=dict(DEFAULT_RIB_AGENT_PARAMS))
        agent.pos = self._sample_valid_position(rng)
        init_angle = float(rng.uniform(-np.pi, np.pi))
        agent.head_direction = np.array([np.cos(init_angle), np.sin(init_angle)], dtype=np.float32)

        observations: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        headings: list[np.ndarray] = []
        wall_signals: list[np.ndarray] = []
        actions = np.zeros((n_steps, 2), dtype=np.float32)

        for step in range(n_steps + 1):
            heading = np.asarray(agent.head_direction, dtype=np.float32)
            heading_norm = np.linalg.norm(heading)
            if heading_norm == 0:
                heading = np.array([1.0, 0.0], dtype=np.float32)
            else:
                heading = heading / heading_norm

            obs, wall_signal = self._fov_observation(np.asarray(agent.pos, dtype=np.float32), heading)
            observations.append(obs.astype(np.float32))
            positions.append(np.asarray(agent.pos, dtype=np.float32))
            headings.append(heading.astype(np.float32))
            wall_signals.append(wall_signal.astype(np.float32))

            if step < n_steps:
                agent.update(dt=DEFAULT_RIB_AGENT_PARAMS["dt"])
                actions[step] = np.asarray(agent.measured_velocity, dtype=np.float32)

        return RolloutBatch(
            observations=np.asarray(observations, dtype=np.float32),
            actions=actions,
            positions=np.asarray(positions, dtype=np.float32),
            headings=np.asarray(headings, dtype=np.float32),
            metadata={"backend": self.backend, "wall_fov": np.asarray(wall_signals, dtype=np.float32)},
        )


def build_rib_env(env_name: str, seed: int = 42, inner_radius: float = 0.15) -> RatInABoxTopologyEnv:
    """Instantiate one RatInABox environment by name."""
    if not RATINABOX_AVAILABLE:
        raise ImportError("RatInABox is not installed.")
    specs = build_rib_specs(inner_radius=inner_radius)
    if env_name not in specs:
        raise KeyError(f"`{env_name}` is not a registered RatInABox environment.")
    return RatInABoxTopologyEnv(env_name, specs[env_name], seed=seed)
