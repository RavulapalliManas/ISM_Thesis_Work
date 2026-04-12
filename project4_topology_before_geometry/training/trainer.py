"""Rollout training loop for the topology-before-geometry experiments."""

from __future__ import annotations

import concurrent.futures
import csv
from contextlib import nullcontext
import math
from pathlib import Path
import queue
import threading
from typing import Any
import warnings

import torch
from tqdm.auto import tqdm

from project4_topology_before_geometry.evaluation.geometric_metrics import precompute_geodesic_matrix


class AsyncLogger:
    """Append CSV rows from a background thread so training never blocks on disk I/O."""

    def __init__(self, path: str | Path, fieldnames: list[str] | None = None):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = list(fieldnames) if fieldnames is not None else None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._closed = False
        self._thread.start()

    def log(self, row: dict[str, Any]) -> None:
        self._queue.put(dict(row))

    def _worker(self) -> None:
        handle = None
        writer = None
        try:
            while True:
                row = self._queue.get()
                if row is None:
                    break
                if handle is None:
                    self._fieldnames = list(self._fieldnames or row.keys())
                    handle = self._path.open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                    writer.writeheader()
                assert writer is not None and handle is not None
                writer.writerow({key: row.get(key, "") for key in self._fieldnames})
                handle.flush()
        finally:
            if handle is not None:
                handle.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10)


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
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.seq_duration = int(self.cfg.get("seq_duration", 100))
        self.n_accumulate = max(int(self.cfg.get("n_accumulate", 1)), 1)
        self.gradient_clip = float(self.cfg.get("gradient_clip", 1.0))
        self.checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "project4_topology_before_geometry/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_workers = max(int(self.cfg.get("n_rollout_workers", 1)), 0)
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=self.rollout_workers)
            if self.rollout_workers > 0
            else None
        )
        self.pin_memory = bool(self.cfg.get("pin_memory", False) and self.device.type == "cuda")
        self.use_mixed_precision = bool(self.cfg.get("mixed_precision", False) and self.device.type == "cuda")
        self.precision = str(self.cfg.get("precision", "bf16")).lower()
        self._autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.cuda_device_index = None
        if self.device.type == "cuda":
            self.cuda_device_index = int(self.device.index if self.device.index is not None else 0)
            torch.cuda.set_device(self.cuda_device_index)
        self.logger = AsyncLogger(self.tracker.log_path)
        if hasattr(self.tracker, "attach_logger"):
            self.tracker.attach_logger(self.logger)
        self.geodesic_matrix = precompute_geodesic_matrix(
            self.env,
            resolution=float(self.cfg.get("geodesic_resolution", 0.05)),
        )

    def _autocast_context(self):
        if not self.use_mixed_precision:
            return nullcontext()
        if self.cuda_device_index is not None:
            torch.cuda.set_device(self.cuda_device_index)
        try:
            return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
        except (AssertionError, RuntimeError) as exc:
            warnings.warn(
                f"Disabling mixed precision after autocast setup failed ({exc}). Training will continue in float32.",
                stacklevel=2,
            )
            self.use_mixed_precision = False
            return nullcontext()

    def _rollout_to_tensors(self, rollout):
        actions = self.act_enc.encode(rollout.actions, rollout.headings)
        obs_cpu = torch.as_tensor(rollout.observations[None, ...], dtype=torch.float32)
        act_cpu = torch.as_tensor(actions[None, ...], dtype=torch.float32)
        pos_cpu = torch.as_tensor(rollout.positions, dtype=torch.float32)
        if self.pin_memory:
            obs_cpu = obs_cpu.pin_memory()
            act_cpu = act_cpu.pin_memory()
            pos_cpu = pos_cpu.pin_memory()
        obs_t = obs_cpu.to(self.device, non_blocking=self.pin_memory)
        act_t = act_cpu.to(self.device, non_blocking=self.pin_memory)
        pos_t = pos_cpu.to(self.device, non_blocking=self.pin_memory)
        return obs_t, act_t, pos_t

    def _positions_for_loss(self, positions: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        target_len = pred.shape[1] if pred.ndim >= 2 else pred.shape[0]
        aligned = positions[:target_len]
        if pred.ndim == 3:
            return aligned.unsqueeze(0).expand(pred.shape[0], -1, -1)
        if pred.ndim == 4:
            return aligned.unsqueeze(0).unsqueeze(0).expand(pred.shape[0], pred.shape[1], -1, -1)
        return aligned

    def _submit_rollout(self, seed: int):
        if self.executor is None:
            return None
        return self.executor.submit(self.env.sample_rollout, self.seq_duration, seed=seed)

    def _consume_rollout(self, future, seed: int):
        if future is None:
            return self.env.sample_rollout(self.seq_duration, seed=seed)
        return future.result()

    def _backward_on_rollout(self, rollout, accumulate_count: int) -> float:
        obs_t, act_t, pos_t = self._rollout_to_tensors(rollout)
        with self._autocast_context():
            outputs = self.model.forward_sequence(obs_t, act_t, training=True)
            pred = outputs["decoded_predictions"]
            target = outputs["raw_targets"]
            hidden = outputs["hidden"]
            if not all(isinstance(item, torch.Tensor) for item in (pred, target, hidden)):
                raise RuntimeError("Model forward_sequence returned non-tensor outputs.")
            positions_for_loss = self._positions_for_loss(pos_t, pred)
            raw_loss = self.loss_fn(pred, target, hidden=hidden, positions=positions_for_loss)
            scaled_loss = raw_loss / float(accumulate_count)
        scaled_loss.backward()
        return float(raw_loss.detach().cpu())

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
            rollout = self.env.sample_rollout(self.seq_duration, seed=int(self.cfg.get("seed", 42)) + trajectory_index)
            accumulated_loss += self._backward_on_rollout(rollout, accumulate_count)

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

        seed_base = int(self.cfg.get("seed", 42))
        trajectories_seen = 0
        next_seed = seed_base + 1
        future = self._submit_rollout(next_seed)
        if future is not None:
            next_seed += 1

        try:
            for update_idx in progress:
                self.model.train()
                self.model.zero_grad()
                accumulate_count = min(self.n_accumulate, total_trajectories - trajectories_seen)
                accumulated_loss = 0.0

                for _ in range(accumulate_count):
                    seed_for_rollout = seed_base + trajectories_seen + 1
                    rollout = self._consume_rollout(future, seed_for_rollout)
                    trajectories_seen += 1

                    if trajectories_seen < total_trajectories:
                        future = self._submit_rollout(next_seed)
                        if future is not None:
                            next_seed += 1
                    else:
                        future = None

                    accumulated_loss += self._backward_on_rollout(rollout, accumulate_count)

                self.model.optimizer_step(gradient_clip=self.gradient_clip)
                mean_loss = accumulated_loss / float(accumulate_count)
                updated = self.tracker.maybe_update(
                    update_idx,
                    mean_loss,
                    trajectories_seen=trajectories_seen,
                )
                if updated is not None:
                    self.save_checkpoint(update_idx, trajectories_seen=trajectories_seen)
                    progress.set_postfix(
                        loss=f"{mean_loss:.4f}",
                        traj=trajectories_seen,
                        srsa_geo=f"{updated.get('srsa_geodesic', float('nan')):.3f}",
                    )
                else:
                    progress.set_postfix(loss=f"{mean_loss:.4f}", traj=trajectories_seen)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
