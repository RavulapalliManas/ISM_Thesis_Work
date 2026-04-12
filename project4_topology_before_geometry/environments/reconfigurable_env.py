"""Helpers for post-training tile reconfiguration experiments."""

from __future__ import annotations

from project4_topology_before_geometry.environments.minigrid_envs import build_minigrid_env


def make_reconfigurable_square(seed: int = 42):
    """Base square used for the rapid-remapping experiment."""
    return build_minigrid_env("square_low_alias", seed=seed)

