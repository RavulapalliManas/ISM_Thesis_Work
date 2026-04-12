#!/usr/bin/env python3
"""Remote runner for RTX 4070-oriented topology-before-geometry training."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from project4_topology_before_geometry.analysis.phase_transition import plot_convergence_curves
from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.environments.topology_labels import get_topology_label
from project4_topology_before_geometry.evaluation.geometric_metrics import precompute_geodesic_matrix
from project4_topology_before_geometry.evaluation.convergence_tracker import ConvergenceTracker
from project4_topology_before_geometry.models.objectives import LossFactory
from project4_topology_before_geometry.models.prnn import RolloutPRNN, maybe_compile
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.training.trainer import Trainer

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="pygame.pkgdata",
)


def _load_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setup(run_cfg: dict, env_name: str, seed: int, device: torch.device):
    env = get_env(env_name, run_cfg)
    precompute_start = time.perf_counter()
    precompute_geodesic_matrix(env, resolution=float(run_cfg.get("geodesic_resolution", 0.05)))
    precompute_elapsed = time.perf_counter() - precompute_start
    print(f"Precomputed geodesic matrix for {env_name} in {precompute_elapsed:.2f}s")

    backend = run_cfg.get("env_backend", {}).get(env_name, "minigrid")
    act_enc = ActionEncoder(backend=backend)
    model = RolloutPRNN(
        obs_dim=env.obs_dim,
        act_dim=act_enc.act_dim,
        hidden_dim=run_cfg["hidden_dim"],
        rollout_k=run_cfg["rollout_k"],
        sigma=run_cfg["sigma_noise"],
        time_mode=run_cfg["time_mode"],
        recurrence_scale=run_cfg["recurrence_scale"],
        dropout=run_cfg["dropout"],
        neural_timescale=run_cfg["neural_timescale"],
        device=device,
    )
    model = maybe_compile(model, run_cfg)
    loss_fn = LossFactory.get_loss(run_cfg["loss_type"], run_cfg["rollout_k"], device)
    tracker = ConvergenceTracker(env, model, get_topology_label(env_name), run_cfg)
    trainer = Trainer(run_cfg, env, model, act_enc, loss_fn, tracker, device=device)
    return env, model, tracker, trainer


def teardown(trainer: Trainer, tracker: ConvergenceTracker, update_count: int, trajectories_seen: int):
    trainer.save_checkpoint(update_count, trajectories_seen=trajectories_seen)
    return tracker.finalize()


def analyze(run_cfg: dict, env_name: str, seed: int):
    log_path = Path(run_cfg["log_dir"]) / f"{env_name}_{seed}.csv"
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    return plot_convergence_curves(df, f"{env_name}_seed{seed}", get_topology_label(env_name), Path(run_cfg["figures_dir"]))


def _seed_list(seed_value) -> list[int]:
    if isinstance(seed_value, list):
        return [int(seed) for seed in seed_value]
    return [int(seed_value)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "remote_config.yaml"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn("CUDA not found. Remote script is intended for the RTX 4070 workstation.")
    print(f"Using device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  {torch.cuda.get_device_name(0)}")
        print(f"  {props.total_memory / 1e9:.1f} GB VRAM")

    cfg = _load_config(Path(args.config))
    base_cfg = dict(cfg)
    base_cfg["device"] = str(device)
    envs = [args.env] if args.env is not None else list(base_cfg.get("environments", []))
    if not envs:
        raise ValueError("No environments specified. Pass --env or define `environments` in the config.")
    seeds = [int(args.seed)] if args.seed is not None else _seed_list(base_cfg.get("seed", 42))
    for directory in (base_cfg["checkpoint_dir"], base_cfg["log_dir"], base_cfg["figures_dir"]):
        Path(directory).mkdir(parents=True, exist_ok=True)

    for env_name, seed in product(envs, seeds):
        run_cfg = dict(base_cfg)
        run_cfg["seed"] = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        env = model = tracker = trainer = None
        finalized = False
        try:
            print(f"\n{'=' * 60}\nRun: env={env_name} seed={seed}\n{'=' * 60}")
            print("Setting up environment, caches, workers, and model...")
            env, model, tracker, trainer = setup(run_cfg, env_name, seed, device)

            print("Starting training loop...")
            total_updates = int(run_cfg["n_updates"])
            total_trajectories = total_updates * int(run_cfg.get("n_accumulate", 1))
            train_start = time.perf_counter()
            trainer.train(total_trajectories)
            train_elapsed = time.perf_counter() - train_start

            print("Tearing down workers and finalizing logs...")
            results = teardown(trainer, tracker, total_updates, total_trajectories)
            finalized = True
            figure_path = analyze(run_cfg, env_name, seed)

            print(f"Training time: {train_elapsed / 3600:.2f}h")
            print(f"gap={results.get('gap')} T_topo={results.get('T_topology')} T_geo={results.get('T_geometry')}")
            if figure_path is not None:
                print(f"Convergence figure -> {figure_path}")
        finally:
            if trainer is not None:
                trainer.close()
            if tracker is not None and not finalized:
                tracker.close()


if __name__ == "__main__":
    main()
