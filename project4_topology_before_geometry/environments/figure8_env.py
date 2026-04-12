"""Figure-8 convenience constructor."""

from project4_topology_before_geometry.environments.rib_envs import build_rib_env


def make_figure8_env(seed: int = 42):
    """Return the B1=2 figure-8 environment."""
    return build_rib_env("figure8_env", seed=seed)

