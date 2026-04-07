from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from shapely.geometry import LineString, Point, Polygon, box
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "project3_generalization.environments.suite_2d requires shapely."
    ) from exc

try:
    from ratinabox.Agent import Agent
    from ratinabox.Environment import Environment
    from ratinabox.Neurons import BoundaryVectorCells, HeadDirectionCells
except ImportError:  # pragma: no cover - optional until runtime
    Agent = None
    Environment = None
    BoundaryVectorCells = None
    HeadDirectionCells = None


DEFAULT_DT = 0.01
DEFAULT_AGENT_PARAMS: dict[str, float | bool] = {
    "dt": DEFAULT_DT,
    "speed_mean": 0.08,
    "speed_std": 0.08,
    "speed_coherence_time": 0.7,
    "rotational_velocity_std": 120 * (np.pi / 180),
    "rotational_velocity_coherence_time": 0.08,
    "thigmotaxis": 0.5,
    "save_history": False,
}
DEFAULT_VECTOR_CELL_PARAMS: dict[str, Any] = {
    "n": 64,
    "reference_frame": "egocentric",
    "save_history": False,
}
DEFAULT_HEAD_DIRECTION_PARAMS: dict[str, Any] = {
    "n": 12,
    "angular_spread_degrees": 45,
    "save_history": False,
}


@dataclass(frozen=True)
class RewardZone:
    center: tuple[float, float]
    radius: float
    value: float = 1.0
    label: str = "reward"


@dataclass(frozen=True)
class EnvironmentSpec2D:
    env_id: str
    name: str
    category: str
    boundary: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    walls: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    reward_zones: tuple[RewardZone, ...] = ()
    objects: tuple[tuple[float, float], ...] = ()
    expected_betti1: int = 0
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_ratinabox_params(self) -> dict[str, Any]:
        params = {
            "boundary": [list(point) for point in self.boundary],
            "holes": [[list(point) for point in hole] for hole in self.holes],
            "walls": [[list(start), list(end)] for start, end in self.walls],
            "objects": [list(obj) for obj in self.objects],
            "dx": 0.02,
        }
        return params

    def build_environment(self) -> Environment:
        _require_ratinabox()
        return Environment(params=self.to_ratinabox_params())

    def create_agent(self, agent_params: Mapping[str, Any] | None = None) -> Agent:
        _require_ratinabox()
        params = dict(DEFAULT_AGENT_PARAMS)
        if agent_params:
            params.update(agent_params)
        return Agent(self.build_environment(), params=params)


@dataclass
class SimulationRollout2D:
    env_id: str
    observations: np.ndarray
    actions: np.ndarray
    positions: np.ndarray
    head_directions: np.ndarray
    velocities: np.ndarray
    rotational_velocities: np.ndarray
    dt: float


def _require_ratinabox() -> None:
    if Environment is None or Agent is None or BoundaryVectorCells is None or HeadDirectionCells is None:
        raise ImportError(
            "RatInABox is required for the 2D environment suite. Install `ratinabox` first."
        )


def _set_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def _to_spec_geometry(
    geom: Polygon,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[tuple[float, float], ...], ...]]:
    cleaned = geom.buffer(0)
    if cleaned.geom_type != "Polygon":
        raise ValueError(f"Expected a Polygon, received {cleaned.geom_type}.")
    boundary = tuple((float(x), float(y)) for x, y in cleaned.exterior.coords[:-1])
    holes = tuple(
        tuple((float(x), float(y)) for x, y in interior.coords[:-1])
        for interior in cleaned.interiors
    )
    return boundary, holes


def _make_spec(
    env_id: str,
    name: str,
    category: str,
    geom: Polygon,
    *,
    walls: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    reward_zones: Sequence[RewardZone] = (),
    objects: Sequence[tuple[float, float]] = (),
    expected_betti1: int = 0,
    notes: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> EnvironmentSpec2D:
    boundary, holes = _to_spec_geometry(geom)
    return EnvironmentSpec2D(
        env_id=env_id,
        name=name,
        category=category,
        boundary=boundary,
        holes=holes,
        walls=tuple(walls),
        reward_zones=tuple(reward_zones),
        objects=tuple(objects),
        expected_betti1=expected_betti1,
        notes=notes,
        metadata=dict(metadata or {}),
    )


def _square(side: float) -> Polygon:
    return box(0.0, 0.0, side, side)


def _rectangle(width: float, height: float) -> Polygon:
    return box(0.0, 0.0, width, height)


def _circle(radius: float, center: tuple[float, float] = (0.0, 0.0), resolution: int = 128) -> Polygon:
    return Point(*center).buffer(radius, resolution=resolution)


def _superellipse(
    a: float,
    b: float,
    exponent: float,
    n_points: int = 256,
    center: tuple[float, float] = (0.0, 0.0),
) -> Polygon:
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / exponent) + center[0]
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / exponent) + center[1]
    return Polygon(np.column_stack([x, y]))


def _l_shape() -> Polygon:
    full = box(0.0, 0.0, 0.75, 0.75)
    cutout = box(0.375, 0.375, 0.75, 0.75)
    return full.difference(cutout)


def _t_maze(arm_length: float = 0.5, corridor_width: float = 0.15) -> Polygon:
    stem_x0 = arm_length
    stem = box(stem_x0, 0.0, stem_x0 + corridor_width, arm_length + corridor_width)
    bar = box(0.0, arm_length, (2 * arm_length) + corridor_width, arm_length + corridor_width)
    return unary_union([stem, bar]).buffer(0)


def _hairpin_maze(corridor_length: float = 0.7, corridor_width: float = 0.1, n_corridors: int = 5) -> Polygon:
    y_levels = corridor_width / 2 + corridor_width * np.arange(n_corridors)
    x_left = corridor_width / 2
    x_right = corridor_length - corridor_width / 2
    points: list[tuple[float, float]] = [(x_left, y_levels[0])]
    for idx, y_level in enumerate(y_levels):
        endpoint = x_right if idx % 2 == 0 else x_left
        points.append((endpoint, y_level))
        if idx < len(y_levels) - 1:
            points.append((endpoint, y_levels[idx + 1]))
    return LineString(points).buffer(corridor_width / 2, cap_style=2, join_style=2).buffer(0)


def _barrier_gap_walls(
    width: float = 0.5,
    barrier_y: float = 0.25,
    gap_center_x: float = 0.25,
    gap_width: float = 0.05,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    left_end = gap_center_x - gap_width / 2
    right_start = gap_center_x + gap_width / 2
    return (
        ((0.0, barrier_y), (left_end, barrier_y)),
        ((right_start, barrier_y), (width, barrier_y)),
    )


def _compartment_walls(
    side: float = 0.6,
    doorway_width: float = 0.15,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    half = side / 2
    gap_half = doorway_width / 2
    return (
        ((half, 0.0), (half, half - gap_half)),
        ((half, half + gap_half), (half, side)),
        ((0.0, half), (half - gap_half, half)),
        ((half + gap_half, half), (side, half)),
    )


def build_suite_2d(include_morph_series: bool = True) -> OrderedDict[str, EnvironmentSpec2D]:
    suite: OrderedDict[str, EnvironmentSpec2D] = OrderedDict()

    suite["A1_square"] = _make_spec(
        "A1_square",
        "Unit Square",
        "A",
        _square(0.5),
        notes="RatInABox square baseline reference.",
        metadata={"width": 0.5, "height": 0.5, "aspect_ratio": 1.0},
    )
    suite["A2_large_square"] = _make_spec(
        "A2_large_square",
        "Large Square",
        "A",
        _square(1.0),
        metadata={"width": 1.0, "height": 1.0, "aspect_ratio": 1.0},
    )
    suite["A3_circle"] = _make_spec(
        "A3_circle",
        "Circle",
        "A",
        _circle(0.35, center=(0.35, 0.35)),
        notes="Circular polygon approximation.",
        metadata={"radius": 0.35},
    )
    suite["A4_rectangle"] = _make_spec(
        "A4_rectangle",
        "Rectangle",
        "A",
        _rectangle(1.0, 0.5),
        metadata={"width": 1.0, "height": 0.5, "aspect_ratio": 2.0},
    )

    suite["B1_l_shape"] = _make_spec(
        "B1_l_shape",
        "L-Shape",
        "B",
        _l_shape(),
        metadata={"width": 0.75, "height": 0.75},
    )
    suite["B2_t_maze"] = _make_spec(
        "B2_t_maze",
        "T-Maze",
        "B",
        _t_maze(),
        notes="Three-arm T-maze with central junction.",
    )
    suite["B3_hairpin"] = _make_spec(
        "B3_hairpin",
        "Hairpin Maze",
        "B",
        _hairpin_maze(),
        notes="Five-corridor zig-zag hairpin layout.",
    )
    suite["B4_compartmentalized"] = _make_spec(
        "B4_compartmentalized",
        "Compartmentalized Arena",
        "B",
        _square(0.6),
        walls=_compartment_walls(),
        notes="Four rooms connected by narrow doorways.",
    )

    suite["C1_center_reward"] = _make_spec(
        "C1_center_reward",
        "Square + Single Central Reward Zone",
        "C",
        _square(0.5),
        reward_zones=(RewardZone(center=(0.25, 0.25), radius=0.05, label="center_reward"),),
        objects=((0.25, 0.25),),
    )
    suite["C2_two_rewards"] = _make_spec(
        "C2_two_rewards",
        "Square + Two Reward Zones",
        "C",
        _square(0.75),
        reward_zones=(
            RewardZone(center=(0.1, 0.1), radius=0.05, label="reward_sw"),
            RewardZone(center=(0.65, 0.65), radius=0.05, label="reward_ne"),
        ),
        objects=((0.1, 0.1), (0.65, 0.65)),
    )
    suite["C3_barrier_gap"] = _make_spec(
        "C3_barrier_gap",
        "Arena with Barrier and Gap",
        "C",
        _square(0.5),
        walls=_barrier_gap_walls(),
        notes="Internal barrier spanning most of the arena width with a narrow central gap.",
    )

    if include_morph_series:
        morph_values = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        for morph_value in morph_values:
            exponent = 20.0 - 18.0 * morph_value
            geom = _superellipse(0.25, 0.25, exponent=exponent, center=(0.25, 0.25))
            env_id = f"C4_morph_{str(morph_value).replace('.', 'p')}"
            suite[env_id] = _make_spec(
                env_id,
                f"Morph Series m={morph_value:.1f}",
                "C",
                geom,
                notes=(
                    "Superellipse interpolation where m=0 is square-like and m=1 "
                    "approaches circular."
                ),
                metadata={"morph": morph_value, "superellipse_exponent": exponent},
            )

    outer_annulus = _circle(0.45, center=(0.45, 0.45))
    inner_annulus = _circle(0.15, center=(0.45, 0.45))
    suite["D1_annulus"] = _make_spec(
        "D1_annulus",
        "Annular Arena",
        "D",
        outer_annulus.difference(inner_annulus),
        expected_betti1=1,
        notes="Outer circular arena with a central circular hole.",
    )
    figure8_outer = _rectangle(1.8, 1.0)
    figure8_hole_left = box(0.2, 0.2, 0.8, 0.8)
    figure8_hole_right = box(1.0, 0.2, 1.6, 0.8)
    suite["D2_figure8"] = _make_spec(
        "D2_figure8",
        "Figure-8 Track",
        "D",
        figure8_outer.difference(unary_union([figure8_hole_left, figure8_hole_right])),
        expected_betti1=2,
        notes="Figure-8 style topology implemented as a connected arena with two holes.",
    )

    return suite


def simulate_random_walk_2d(
    spec: EnvironmentSpec2D,
    n_steps: int,
    *,
    seed: int | None = None,
    agent_params: Mapping[str, Any] | None = None,
    dt: float = DEFAULT_DT,
) -> tuple[Environment, Agent, np.ndarray, np.ndarray, np.ndarray]:
    _require_ratinabox()
    _set_seed(seed)

    env = spec.build_environment()
    params = dict(DEFAULT_AGENT_PARAMS)
    params["dt"] = dt
    if agent_params:
        params.update(agent_params)
    agent = Agent(env, params=params)

    positions = np.zeros((n_steps + 1, 2), dtype=float)
    velocities = np.zeros((n_steps, 2), dtype=float)
    head_directions = np.zeros((n_steps + 1, 2), dtype=float)
    positions[0] = agent.pos
    head_directions[0] = agent.head_direction

    for step in range(n_steps):
        agent.update(dt=dt)
        positions[step + 1] = agent.pos
        head_directions[step + 1] = agent.head_direction
        velocities[step] = agent.measured_velocity

    return env, agent, positions, velocities, head_directions


def collect_rollout_2d(
    spec: EnvironmentSpec2D,
    n_steps: int,
    *,
    seed: int | None = None,
    dt: float = DEFAULT_DT,
    agent_params: Mapping[str, Any] | None = None,
    vector_cell_params: Mapping[str, Any] | None = None,
    head_direction_params: Mapping[str, Any] | None = None,
) -> SimulationRollout2D:
    _require_ratinabox()
    _set_seed(seed)

    env = spec.build_environment()
    params = dict(DEFAULT_AGENT_PARAMS)
    params["dt"] = dt
    if agent_params:
        params.update(agent_params)
    agent = Agent(env, params=params)

    bvc_params = dict(DEFAULT_VECTOR_CELL_PARAMS)
    if vector_cell_params:
        bvc_params.update(vector_cell_params)
    hd_params = dict(DEFAULT_HEAD_DIRECTION_PARAMS)
    if head_direction_params:
        hd_params.update(head_direction_params)

    bvcs = BoundaryVectorCells(agent, params=bvc_params)
    hdc = HeadDirectionCells(agent, params=hd_params)

    observations = np.zeros((n_steps + 1, bvcs.n + hdc.n), dtype=np.float32)
    actions = np.zeros((n_steps, 3), dtype=np.float32)
    positions = np.zeros((n_steps + 1, 2), dtype=np.float32)
    head_directions = np.zeros((n_steps + 1, 2), dtype=np.float32)
    velocities = np.zeros((n_steps, 2), dtype=np.float32)
    rotational_velocities = np.zeros((n_steps,), dtype=np.float32)

    for step in range(n_steps + 1):
        bvcs.update()
        hdc.update()
        observations[step, : bvcs.n] = bvcs.firingrate.astype(np.float32)
        observations[step, bvcs.n :] = hdc.firingrate.astype(np.float32)
        positions[step] = agent.pos.astype(np.float32)
        head_directions[step] = agent.head_direction.astype(np.float32)

        if step < n_steps:
            agent.update(dt=dt)
            velocities[step] = agent.measured_velocity.astype(np.float32)
            rotational_velocity = agent.measured_rotational_velocity
            rotational_velocities[step] = 0.0 if np.isnan(rotational_velocity) else float(rotational_velocity)
            actions[step, :2] = velocities[step]
            actions[step, 2] = rotational_velocities[step]

    return SimulationRollout2D(
        env_id=spec.env_id,
        observations=observations,
        actions=actions,
        positions=positions,
        head_directions=head_directions,
        velocities=velocities,
        rotational_velocities=rotational_velocities,
        dt=dt,
    )


def _grid_centers(extent: Sequence[float], grid_size: int) -> np.ndarray:
    left, right, bottom, top = extent
    xs = np.linspace(left, right, grid_size, endpoint=False) + (right - left) / (2 * grid_size)
    ys = np.linspace(bottom, top, grid_size, endpoint=False) + (top - bottom) / (2 * grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def validate_environment_2d(
    spec: EnvironmentSpec2D,
    *,
    n_steps: int = 10_000,
    seed: int = 0,
    grid_size: int = 50,
) -> dict[str, Any]:
    _require_ratinabox()
    env, agent, positions, _, _ = simulate_random_walk_2d(spec, n_steps=n_steps, seed=seed)
    inside = np.array([env.check_if_position_is_in_environment(pos) for pos in positions], dtype=bool)

    grid_centers = _grid_centers(env.extent, grid_size)
    valid_mask = np.array(
        [env.check_if_position_is_in_environment(pos) for pos in grid_centers],
        dtype=bool,
    )

    xs = np.clip(
        ((positions[:, 0] - env.extent[0]) / max(env.extent[1] - env.extent[0], 1e-9) * grid_size).astype(int),
        0,
        grid_size - 1,
    )
    ys = np.clip(
        ((positions[:, 1] - env.extent[2]) / max(env.extent[3] - env.extent[2], 1e-9) * grid_size).astype(int),
        0,
        grid_size - 1,
    )
    occupancy = np.zeros((grid_size, grid_size), dtype=float)
    np.add.at(occupancy, (ys, xs), 1.0)
    valid_occupancy = occupancy.ravel()[valid_mask]
    coverage_score = float(valid_occupancy.mean() / (valid_occupancy.std() + 1e-6))

    allocentric_bvcs = BoundaryVectorCells(
        agent,
        params={"n": 16, "reference_frame": "allocentric", "save_history": False},
    )
    bvc_rates = allocentric_bvcs.get_head_direction_averaged_state(evaluate_at="all")
    bvc_dynamic_range = (bvc_rates.max(axis=1) - bvc_rates.min(axis=1)) / (bvc_rates.mean(axis=1) + 1e-6)
    tuned_fraction = float(np.mean(bvc_dynamic_range > 0.5))

    return {
        "env_id": spec.env_id,
        "trajectory_respects_boundaries": bool(np.all(inside)),
        "coverage_score": coverage_score,
        "coverage_histogram": occupancy,
        "coverage_valid_mask": valid_mask.reshape(grid_size, grid_size),
        "bvc_tuned_fraction": tuned_fraction,
        "bvc_dynamic_range": bvc_dynamic_range,
        "positions": positions,
        "extent": np.array(env.extent, dtype=float),
    }


__all__ = [
    "DEFAULT_DT",
    "DEFAULT_AGENT_PARAMS",
    "DEFAULT_VECTOR_CELL_PARAMS",
    "DEFAULT_HEAD_DIRECTION_PARAMS",
    "EnvironmentSpec2D",
    "RewardZone",
    "SimulationRollout2D",
    "build_suite_2d",
    "collect_rollout_2d",
    "simulate_random_walk_2d",
    "validate_environment_2d",
]
