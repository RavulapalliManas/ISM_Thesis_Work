"""Ground-truth topology labels for both legacy and aliasing-controlled environments."""

from __future__ import annotations

from copy import deepcopy
import re


_BASE_TOPOLOGY_LABELS: dict[str, dict[str, int]] = {
    "square_low_alias": {"betti_0": 1, "betti_1": 0},
    "square_high_alias": {"betti_0": 1, "betti_1": 0},
    "rectangle_wide": {"betti_0": 1, "betti_1": 0},
    "rectangle_narrow": {"betti_0": 1, "betti_1": 0},
    "l_shape_standard": {"betti_0": 1, "betti_1": 0},
    "l_shape_large": {"betti_0": 1, "betti_1": 0},
    "circle_approx": {"betti_0": 1, "betti_1": 0},
    "triangle_approx": {"betti_0": 1, "betti_1": 0},
    "two_room_corridor": {"betti_0": 1, "betti_1": 0},
    "hairpin_maze": {"betti_0": 1, "betti_1": 0},
    "annulus_approx": {"betti_0": 1, "betti_1": 1},
    "figure8_env": {"betti_0": 1, "betti_1": 2},
    "cylinder_env": {"betti_0": 1, "betti_1": 1},
}

_GEOMETRY_TOPOLOGY: dict[str, dict[str, int]] = {
    "square": {"betti_0": 1, "betti_1": 0},
    "rectangle": {"betti_0": 1, "betti_1": 0},
    "circle_approx": {"betti_0": 1, "betti_1": 0},
    "l_shape": {"betti_0": 1, "betti_1": 0},
    "u_shape": {"betti_0": 1, "betti_1": 0},
    "corridor": {"betti_0": 1, "betti_1": 0},
    "t_maze": {"betti_0": 1, "betti_1": 0},
    "plus_maze": {"betti_0": 1, "betti_1": 0},
    "two_room": {"betti_0": 1, "betti_1": 0},
    "three_room": {"betti_0": 1, "betti_1": 0},
    "four_room": {"betti_0": 1, "betti_1": 0},
    "bottleneck_room": {"betti_0": 1, "betti_1": 0},
    "maze_simple": {"betti_0": 1, "betti_1": 0},
    "maze_medium": {"betti_0": 1, "betti_1": 0},
    "loop_corridor": {"betti_0": 1, "betti_1": 1},
    "spiral_maze": {"betti_0": 1, "betti_1": 0},
    "dead_end_maze": {"betti_0": 1, "betti_1": 0},
    "branching_tree": {"betti_0": 1, "betti_1": 0},
    "figure_8": {"betti_0": 1, "betti_1": 2},
    "double_loop": {"betti_0": 1, "betti_1": 2},
    "nested_rooms": {"betti_0": 1, "betti_1": 2},
    "room_with_island": {"betti_0": 1, "betti_1": 1},
    "uniform_box": {"betti_0": 1, "betti_1": 0},
    "repeating_stripes": {"betti_0": 1, "betti_1": 0},
    "checkerboard_large_period": {"betti_0": 1, "betti_1": 0},
    "symmetry_trap": {"betti_0": 1, "betti_1": 0},
    "long_corridor_alias": {"betti_0": 1, "betti_1": 0},
    "ambiguous_junctions": {"betti_0": 1, "betti_1": 0},
    "perceptual_alias_maze": {"betti_0": 1, "betti_1": 0},
    "no_landmarks": {"betti_0": 1, "betti_1": 0},
    "boundary_only_landmarks": {"betti_0": 1, "betti_1": 0},
    "center_only_landmark": {"betti_0": 1, "betti_1": 0},
    "sparse_random_landmarks": {"betti_0": 1, "betti_1": 0},
    "zero_alias": {"betti_0": 1, "betti_1": 0},
    "low_alias": {"betti_0": 1, "betti_1": 0},
    "medium_alias": {"betti_0": 1, "betti_1": 0},
    "high_alias": {"betti_0": 1, "betti_1": 0},
    "maximum_alias": {"betti_0": 1, "betti_1": 0},
}

_ALIAS_SUFFIXES = (
    "zero_alias",
    "low_alias",
    "medium_alias",
    "high_alias",
    "maximum_alias",
)
_CANONICAL_RE = re.compile(
    r"^(?P<geometry>.+?)(?:_(?:τ|tau)=(?P<tau>[^_]+)_(?:λ|lambda)=(?P<lambda>[^_]+)_H=(?P<H>[^_]+)_(?:ω|omega)=(?P<omega>[^_]+))$"
)


def _canonical_geometry_name(env_name: str) -> str:
    name = str(env_name)
    if name in _BASE_TOPOLOGY_LABELS or name in _GEOMETRY_TOPOLOGY:
        return name

    canonical_match = _CANONICAL_RE.match(name)
    if canonical_match:
        return canonical_match.group("geometry")

    for suffix in _ALIAS_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            return name[: -len(token)]

    return name


def get_topology_label(env_name: str) -> dict[str, int]:
    """Resolve topology labels for legacy, combined, and canonical aliasing env names."""
    if env_name in _BASE_TOPOLOGY_LABELS:
        return deepcopy(_BASE_TOPOLOGY_LABELS[env_name])

    geometry_name = _canonical_geometry_name(env_name)
    if geometry_name in _BASE_TOPOLOGY_LABELS:
        return deepcopy(_BASE_TOPOLOGY_LABELS[geometry_name])
    if geometry_name in _GEOMETRY_TOPOLOGY:
        return deepcopy(_GEOMETRY_TOPOLOGY[geometry_name])

    raise KeyError(f"Unknown topology label for environment `{env_name}`.")


TOPOLOGY_LABELS: dict[str, dict[str, int]] = {
    **_BASE_TOPOLOGY_LABELS,
    **_GEOMETRY_TOPOLOGY,
}

