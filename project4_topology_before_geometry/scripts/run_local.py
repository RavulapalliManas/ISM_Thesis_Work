#!/usr/bin/env python3
"""Local prototype runner for the topology-before-geometry project."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from project4_topology_before_geometry.analysis.phase_transition import plot_convergence_curves
from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.environments.topology_labels import TOPOLOGY_LABELS
from project4_topology_before_geometry.models.objectives import LossFactory
from project4_topology_before_geometry.models.prnn import RolloutPRNN
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.training.trainer import Trainer
from project4_topology_before_geometry.evaluation.convergence_tracker import ConvergenceTracker


def _load_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn("CUDA not found. Training will be very slow.")
    print(f"Using device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  {torch.cuda.get_device_name(0)}")
        print(f"  {props.total_memory / 1e9:.1f} GB VRAM")

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "local_config.yaml"
    cfg = _load_config(cfg_path)
    cfg["device"] = str(device)
    for directory in (cfg["checkpoint_dir"], cfg["log_dir"], cfg["figures_dir"]):
        Path(directory).mkdir(parents=True, exist_ok=True)

    env_iterator = tqdm(
        cfg["environments"],
        desc="local-envs",
        disable=not bool(cfg.get("use_tqdm", True)),
    )
    for env_name in env_iterator:
        print(f"\n{'=' * 60}\nEnvironment: {env_name}\n{'=' * 60}")
        tracker = trainer = None
        finalized = False
        try:
            env = get_env(env_name, cfg)
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
            print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
            loss_fn = LossFactory.get_loss(cfg["loss_type"], cfg["rollout_k"], device)
            tracker = ConvergenceTracker(env, model, TOPOLOGY_LABELS[env_name], cfg)
            trainer = Trainer(cfg, env, model, act_enc, loss_fn, tracker, device=device)
            trainer.train(n_trials=cfg["n_trials"])
            results = tracker.finalize()
            finalized = True
            env_iterator.set_postfix(env=env_name, gap=results.get("gap"))
            print(f"Results: T_topo={results.get('T_topology')} T_geo={results.get('T_geometry')} gap={results.get('gap')}")
            import pandas as pd

            log_df = pd.read_csv(Path(cfg["log_dir"]) / f"{env_name}_{seed}.csv")
            plot_convergence_curves(log_df, env_name, TOPOLOGY_LABELS[env_name], Path(cfg["figures_dir"]))
        finally:
            if trainer is not None:
                trainer.close()
            if tracker is not None and not finalized:
                tracker.close()


if __name__ == "__main__":
    main()
