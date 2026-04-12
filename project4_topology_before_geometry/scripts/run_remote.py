#!/usr/bin/env python3
"""Remote runner for RTX 4070-oriented topology-before-geometry training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from project4_topology_before_geometry.analysis.phase_transition import plot_convergence_curves
from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.environments.topology_labels import TOPOLOGY_LABELS
from project4_topology_before_geometry.evaluation.geometric_metrics import precompute_geodesic_matrix
from project4_topology_before_geometry.evaluation.convergence_tracker import ConvergenceTracker
from project4_topology_before_geometry.models.objectives import LossFactory
from project4_topology_before_geometry.models.prnn import RolloutPRNN, maybe_compile
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.training.trainer import Trainer


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

    act_enc = ActionEncoder(backend=run_cfg["env_backend"][env_name])
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
    tracker = ConvergenceTracker(env, model, TOPOLOGY_LABELS[env_name], run_cfg)
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
    return plot_convergence_curves(df, f"{env_name}_seed{seed}", TOPOLOGY_LABELS[env_name], Path(run_cfg["figures_dir"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    run_cfg = dict(cfg)
    run_cfg["device"] = str(device)
    run_cfg["seed"] = int(args.seed)
    for directory in (run_cfg["checkpoint_dir"], run_cfg["log_dir"], run_cfg["figures_dir"]):
        Path(directory).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = model = tracker = trainer = None
    finalized = False
    try:
        print("Setting up environment, caches, workers, and model...")
        env, model, tracker, trainer = setup(run_cfg, args.env, args.seed, device)

        print("Starting training loop...")
        total_updates = int(run_cfg["n_updates"])
        total_trajectories = total_updates * int(run_cfg.get("n_accumulate", 1))
        train_start = time.perf_counter()
        trainer.train(total_trajectories)
        train_elapsed = time.perf_counter() - train_start

        print("Tearing down workers and finalizing logs...")
        results = teardown(trainer, tracker, total_updates, total_trajectories)
        finalized = True
        figure_path = analyze(run_cfg, args.env, args.seed)

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
