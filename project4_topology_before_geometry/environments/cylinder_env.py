"""Cylinder-environment convenience constructor."""

from project4_topology_before_geometry.environments.rib_envs import build_rib_env


def make_cylinder_env(seed: int = 42):
    """Return the periodic cylinder environment."""
    return build_rib_env("cylinder_env", seed=seed)

