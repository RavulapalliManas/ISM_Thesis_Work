"""Hairpin-maze convenience constructor."""

from project4_topology_before_geometry.environments.minigrid_envs import build_minigrid_env


def make_hairpin_maze(seed: int = 42):
    """Return the periodic-aliasing hairpin-maze environment."""
    return build_minigrid_env("hairpin_maze", seed=seed)

