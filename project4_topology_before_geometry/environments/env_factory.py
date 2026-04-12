"""Single environment-instantiation entry point for project 4."""

from __future__ import annotations

from project4_topology_before_geometry.environments.aliasing_controlled_envs import (
    is_aliasing_controlled_name,
    list_prebuilt_environments,
    make_environment,
)
from project4_topology_before_geometry.environments.minigrid_envs import (
    MINIGRID_AVAILABLE,
    build_minigrid_env,
    build_minigrid_specs,
)
from project4_topology_before_geometry.environments.rib_envs import build_rib_env, build_rib_specs


def list_environments() -> list[str]:
    """Return the union of registered MiniGrid and RatInABox environment names."""
    return sorted(set(build_minigrid_specs()) | set(build_rib_specs()) | set(list_prebuilt_environments()))


def get_env(env_name: str | dict, cfg: dict | None = None):
    """Instantiate one environment, preferring MiniGrid and falling back to RatInABox."""
    cfg = dict(cfg or {})
    if isinstance(env_name, dict):
        env_request = dict(env_name)
        env_name = str(env_request.get("env_type") or env_request.get("name") or env_request.get("geometry") or "square")
        cfg = {**cfg, **env_request}

    env_backend = dict(cfg.get("env_backend", {}))
    requested_backend = env_backend.get(env_name)
    seed = int(cfg.get("seed", 42))
    inner_radius = float(cfg.get("inner_radius", 0.15))
    factory_cfg = {key: value for key, value in cfg.items() if key != "seed"}

    if is_aliasing_controlled_name(env_name):
        return make_environment(env_name, seed=seed, **factory_cfg)

    discrete_names = set(build_minigrid_specs())
    continuous_names = set(build_rib_specs(inner_radius=inner_radius))

    if requested_backend == "minigrid":
        return build_minigrid_env(env_name, seed=seed)
    if requested_backend == "ratinabox":
        return build_rib_env(env_name, seed=seed, inner_radius=inner_radius)

    if env_name in discrete_names and MINIGRID_AVAILABLE:
        return build_minigrid_env(env_name, seed=seed)
    if env_name in continuous_names:
        return build_rib_env(env_name, seed=seed, inner_radius=inner_radius)
    if env_name in discrete_names:
        raise ImportError(
            f"MiniGrid is required for `{env_name}` in the current build. Install `minigrid` to use the paper-faithful discrete backend."
        )
    raise KeyError(f"Unknown environment `{env_name}`.")
