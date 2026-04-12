#!/usr/bin/env python3
"""Remote runner for full topology-before-geometry sweeps."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from project4_topology_before_geometry.analysis.phase_transition import (
    plot_convergence_curves,
    plot_cross_env_summary,
    plot_gap_vs_aliasing,
    plot_gap_vs_complexity,
)
from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.environments.topology_labels import TOPOLOGY_LABELS
from project4_topology_before_geometry.evaluation.convergence_tracker import ConvergenceTracker
from project4_topology_before_geometry.models.objectives import LossFactory
from project4_topology_before_geometry.models.prnn import RolloutPRNN
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.training.trainer import Trainer


def _load_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn("CUDA not found.")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "remote_config.yaml"
    cfg = _load_config(cfg_path)
    cfg["device"] = str(device)
    if args.loss_type is not None:
        cfg["loss_type"] = args.loss_type

    envs = [args.env] if args.env else cfg["environments"]
    seeds = [args.seed] if args.seed is not None else cfg["seed"]
    for directory in (cfg["checkpoint_dir"], cfg["log_dir"], cfg["figures_dir"]):
        Path(directory).mkdir(parents=True, exist_ok=True)

    all_results = {}
    run_iterator = tqdm(
        list(product(envs, seeds)),
        desc="remote-runs",
        disable=not bool(cfg.get("use_tqdm", True)),
    )
    for env_name, seed in run_iterator:
        run_cfg = {**cfg, "seed": seed}
        run_id = f"{env_name}_{run_cfg['loss_type']}_seed{seed}"
        print(f"\n{'=' * 60}\n{run_id}\n{'=' * 60}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        env = get_env(env_name, run_cfg)
        act_enc = ActionEncoder(backend=cfg["env_backend"][env_name])
        model = RolloutPRNN(
            obs_dim=env.obs_dim,
            act_dim=act_enc.act_dim,
            hidden_dim=cfg["hidden_dim"],
            rollout_k=cfg["rollout_k"],
            sigma=cfg["sigma_noise"],
            time_mode=cfg["time_mode"],
            recurrence_scale=cfg["recurrence_scale"],
            dropout=cfg["dropout"],
            neural_timescale=cfg["neural_timescale"],
            device=device,
        )
        loss_fn = LossFactory.get_loss(cfg["loss_type"], cfg["rollout_k"], device)
        tracker = ConvergenceTracker(env, model, TOPOLOGY_LABELS[env_name], run_cfg)
        trainer = Trainer(run_cfg, env, model, act_enc, loss_fn, tracker, device=device)
        ckpt_path = Path(cfg["checkpoint_dir"]) / f"{run_id}_latest.pt"
        if args.resume and ckpt_path.exists():
            trainer.load_checkpoint(ckpt_path)
        trainer.train(n_trials=cfg["n_trials"])
        results = tracker.finalize()
        results["complexity_index"] = env.complexity_index
        results["geo_euclidean_discrepancy"] = env.geo_euclidean_discrepancy
        results["env_name"] = env_name
        results["loss_type"] = cfg["loss_type"]
        results["seed"] = seed
        results["betti_1_gt"] = TOPOLOGY_LABELS[env_name]["betti_1"]
        all_results[run_id] = results
        log_path = Path(cfg["log_dir"]) / f"{env_name}_{seed}.csv"
        if log_path.exists():
            df = pd.read_csv(log_path)
            plot_convergence_curves(df, run_id, TOPOLOGY_LABELS[env_name], Path(cfg["figures_dir"]))
        run_iterator.set_postfix(run=run_id, gap=results.get("gap"))
        print(f"gap={results.get('gap')} T_topo={results.get('T_topology')} T_geo={results.get('T_geometry')}")

    summary_df = pd.DataFrame(list(all_results.values()))
    summary_df.to_csv(Path(cfg["log_dir"]) / "summary.csv", index=False)
    plot_gap_vs_complexity(all_results, Path(cfg["figures_dir"]))
    plot_gap_vs_aliasing(all_results, Path(cfg["figures_dir"]))
    plot_cross_env_summary(all_results, Path(cfg["figures_dir"]))
    print(f"All figures -> {cfg['figures_dir']}")


if __name__ == "__main__":
    main()
