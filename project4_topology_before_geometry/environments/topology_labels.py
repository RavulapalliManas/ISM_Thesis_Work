"""Ground-truth topology labels for the environment suite."""

TOPOLOGY_LABELS: dict[str, dict[str, int]] = {
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

