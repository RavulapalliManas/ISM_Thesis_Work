"""Rollout training loop for the topology-before-geometry experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


class Trainer:
    """Single-environment trainer that reuses the legacy rollout core."""

    def __init__(self, cfg: dict[str, Any], env, model, act_enc, loss_fn, tracker, device: torch.device | str | None = None):
        self.cfg = dict(cfg)
        self.env = env
        self.model = model
        self.act_enc = act_enc
        self.loss_fn = loss_fn
        self.tracker = tracker
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seq_duration = int(self.cfg.get("seq_duration", 100))
        self.n_accumulate = max(int(self.cfg.get("n_accumulate", 1)), 1)
        self.gradient_clip = float(self.cfg.get("gradient_clip", 1.0))
        self.checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "project4_topology_before_geometry/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_batch(self, seed: int):
        rollout = self.env.sample_rollout(self.seq_duration, seed=seed)
        actions = self.act_enc.encode(rollout.actions, rollout.headings)
        obs_t = torch.as_tensor(rollout.observations[None, ...], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions[None, ...], dtype=torch.float32, device=self.device)
        pos_t = torch.as_tensor(rollout.positions, dtype=torch.float32, device=self.device)
        return rollout, obs_t, act_t, pos_t

    @staticmethod
    def _positions_for_loss(positions: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        target_len = pred.shape[1] if pred.ndim >= 2 else pred.shape[0]
        aligned = positions[:target_len]
        if pred.ndim == 3:
            return aligned.unsqueeze(0).expand(pred.shape[0], -1, -1)
        if pred.ndim == 4:
            return aligned.unsqueeze(0).unsqueeze(0).expand(pred.shape[0], pred.shape[1], -1, -1)
        return aligned

    def train_trial(
        self,
        trial_idx: int,
        *,
        accumulate_count: int | None = None,
        trajectory_offset: int = 0,
    ) -> dict[str, float]:
        self.model.train()
        self.model.zero_grad()
        accumulate_count = self.n_accumulate if accumulate_count is None else max(int(accumulate_count), 1)
        accumulated_loss = 0.0

        for acc_step in range(accumulate_count):
            trajectory_index = trajectory_offset + acc_step + 1
            _, obs_t, act_t, pos_t = self._prepare_batch(seed=int(self.cfg.get("seed", 42)) + trajectory_index)
            outputs = self.model.forward_sequence(obs_t, act_t, training=True)
            pred = outputs["decoded_predictions"]
            target = outputs["raw_targets"]
            hidden = outputs["hidden"]
            if not all(isinstance(item, torch.Tensor) for item in (pred, target, hidden)):
                raise RuntimeError("Model forward_sequence returned non-tensor outputs.")

            positions_for_loss = self._positions_for_loss(pos_t, pred)
            raw_loss = self.loss_fn(pred, target, hidden=hidden, positions=positions_for_loss)
            (raw_loss / float(accumulate_count)).backward()
            accumulated_loss += float(raw_loss.detach().cpu())

        self.model.optimizer_step(gradient_clip=self.gradient_clip)
        return {
            "loss": accumulated_loss / float(accumulate_count),
            "n_accumulate": float(accumulate_count),
        }

    def save_checkpoint(self, trial_idx: int, *, trajectories_seen: int | None = None) -> Path:
        path = self.checkpoint_dir / f"{self.env.env_name}_{self.cfg.get('loss_type', 'rollout_mse')}_seed{self.cfg.get('seed', 42)}_latest.pt"
        torch.save(
            {
                "trial": int(trajectories_seen if trajectories_seen is not None else trial_idx),
                "update": int(trial_idx),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
                "n_accumulate": int(self.n_accumulate),
                "cfg": self.cfg,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def get_hidden_snapshot(self, trial: int, **kwargs):
        return self.tracker.get_hidden_snapshot(trial, **kwargs)

    def train(self, n_trials: int):
        total_trajectories = int(n_trials)
        total_updates = math.ceil(total_trajectories / self.n_accumulate)
        progress = tqdm(
            range(1, total_updates + 1),
            desc=f"train:{self.env.env_name}",
            unit="update",
            disable=not bool(self.cfg.get("use_tqdm", True)),
        )
        trajectories_seen = 0
        for trial_idx in progress:
            accumulate_count = min(self.n_accumulate, total_trajectories - trajectories_seen)
            metrics = self.train_trial(
                trial_idx,
                accumulate_count=accumulate_count,
                trajectory_offset=trajectories_seen,
            )
            trajectories_seen += accumulate_count
            updated = self.tracker.maybe_update(
                trial_idx,
                metrics["loss"],
                trajectories_seen=trajectories_seen,
            )
            if updated is not None:
                self.save_checkpoint(trial_idx, trajectories_seen=trajectories_seen)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    traj=trajectories_seen,
                    srsa_geo=f"{updated.get('srsa_geodesic', float('nan')):.3f}",
                )
            else:
                progress.set_postfix(loss=f"{metrics['loss']:.4f}", traj=trajectories_seen)
