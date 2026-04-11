"""
File: project3_generalization/visual_rnn/train.py

Description:
Single-run training pipeline for visual-input predictive-RNN experiments.

Role in system:
This is the highest-level implementation of the visual branch. It glues
together environment rollouts, tile-map rendering, model training, dashboard
logging, checkpointing, and post-run analysis into one reproducible run.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

from project3_generalization.environments.suite_2d import EnvironmentSpec2D, build_suite_2d, collect_rollout_2d
from project3_generalization.evaluation.metrics import fraction_spatially_tuned
from project3_generalization.hardware import gpu_memory_snapshot
from project3_generalization.models.hippocampal_module import HippocampalConfig, HippocampalPredictiveRNN
from project3_generalization.visual_rnn.analysis import (
    generate_post_run_analysis,
    plot_hidden_embedding,
    plot_trajectory_reconstruction,
    plot_visual_prediction,
)
from project3_generalization.visual_rnn.model import build_visual_model_config
from project3_generalization.visual_rnn.renderer import TileMapConfig, build_tile_map


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload to disk, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _maybe_tensorboard_writer(log_dir: Path):
    """Create a TensorBoard writer when the dependency is available."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(log_dir))


class DashboardLogger:
    """Small wrapper for live TensorBoard or W&B logging."""

    def __init__(self, config: "DashboardConfig", run_dir: Path, run_id: str):
        """Initialize optional dashboard backends for one training run."""
        self.backend = config.backend
        self.enabled = bool(config.enabled)
        self.update_every_n_steps = max(int(config.update_every_n_steps), 1)
        self.writer = None
        self._wandb = None
        if not self.enabled:
            return

        if self.backend == "tensorboard":
            self.writer = _maybe_tensorboard_writer(run_dir / "dashboard")
            if self.writer is None:
                self.enabled = False
        elif self.backend == "wandb":
            try:
                import wandb
            except ImportError:
                self.enabled = False
            else:
                self._wandb = wandb
                wandb.init(
                    project=config.project,
                    name=config.run_name or run_id,
                    dir=str(run_dir),
                    mode=config.wandb_mode,
                    config={"run_id": run_id},
                )
        else:
            raise ValueError(f"Unsupported dashboard backend `{self.backend}`.")

    def log_scalars(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics to the configured backend, if enabled."""
        if not self.enabled:
            return
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, float(value), step)
            self.writer.flush()
        elif self._wandb is not None:
            self._wandb.log({key: float(value) for key, value in metrics.items()}, step=step)

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """Log a Matplotlib figure and ensure it is closed afterward."""
        if not self.enabled:
            plt.close(figure)
            return
        if self.writer is not None:
            self.writer.add_figure(tag, figure, global_step=step)
            self.writer.flush()
        elif self._wandb is not None:
            self._wandb.log({tag: self._wandb.Image(figure)}, step=step)
        plt.close(figure)

    def close(self) -> None:
        """Close any open dashboard resources."""
        if self.writer is not None:
            self.writer.close()
        if self._wandb is not None:
            self._wandb.finish()


@dataclass
class DashboardConfig:
    """Configuration for optional experiment dashboards such as TensorBoard or W&B."""

    enabled: bool = True
    backend: str = "tensorboard"
    update_every_n_steps: int = 5
    project: str = "predictive-rnn-visual"
    run_name: str | None = None
    wandb_mode: str = "offline"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "DashboardConfig":
        """Construct a dashboard config from a decoded mapping."""
        return cls(**dict(mapping or {}))


@dataclass
class ExperimentConfig:
    """Configuration for one visual-input predictive-RNN run."""

    run_name: str = "visual_predictive_rnn"
    output_root: str = "results"
    env_id: str = "B1_l_shape"
    seed: int = 7
    epochs: int = 20
    sequence_length: int = 64
    train_sequences_per_epoch: int = 24
    val_sequences_per_epoch: int = 6
    batch_size: int = 4
    eval_rollout_steps: int = 768
    replay_steps: int = 256
    gradient_clip: float | None = 1.0
    checkpoint_every_n_epochs: int = 5
    early_stopping_patience: int = 8
    observation_mode: str = "visual"
    include_head_direction: bool = True
    agent_params: dict[str, Any] = field(default_factory=dict)
    tile_map: TileMapConfig = field(default_factory=TileMapConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    model: HippocampalConfig = field(default_factory=build_visual_model_config)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "ExperimentConfig":
        """Build an experiment config from a JSON-like mapping."""
        mapping = dict(mapping or {})
        include_head_direction = bool(mapping.get("include_head_direction", True))
        default_model = asdict(build_visual_model_config(include_head_direction=include_head_direction))
        default_model.update(mapping.get("model", {}))
        return cls(
            run_name=str(mapping.get("run_name", "visual_predictive_rnn")),
            output_root=str(mapping.get("output_root", "results")),
            env_id=str(mapping.get("env_id", "B1_l_shape")),
            seed=int(mapping.get("seed", 7)),
            epochs=int(mapping.get("epochs", 20)),
            sequence_length=int(mapping.get("sequence_length", 64)),
            train_sequences_per_epoch=int(mapping.get("train_sequences_per_epoch", 24)),
            val_sequences_per_epoch=int(mapping.get("val_sequences_per_epoch", 6)),
            batch_size=int(mapping.get("batch_size", 4)),
            eval_rollout_steps=int(mapping.get("eval_rollout_steps", 768)),
            replay_steps=int(mapping.get("replay_steps", 256)),
            gradient_clip=mapping.get("gradient_clip", 1.0),
            checkpoint_every_n_epochs=int(mapping.get("checkpoint_every_n_epochs", 5)),
            early_stopping_patience=int(mapping.get("early_stopping_patience", 8)),
            observation_mode=str(mapping.get("observation_mode", "visual")),
            include_head_direction=include_head_direction,
            agent_params=dict(mapping.get("agent_params", {})),
            tile_map=TileMapConfig(**dict(mapping.get("tile_map", {}))),
            dashboard=DashboardConfig.from_mapping(mapping.get("dashboard")),
            model=HippocampalConfig(**default_model),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load an experiment config from a JSON file on disk."""
        return cls.from_mapping(json.loads(Path(path).read_text()))


@dataclass
class RunResult:
    """Summary object returned after a complete visual-input run."""

    run_id: str
    run_dir: str
    best_checkpoint: str
    history: list[dict[str, float]]
    summary: dict[str, Any]


def set_global_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_run_dir(config: ExperimentConfig) -> tuple[str, Path]:
    """Create the run directory structure and return its identifier."""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.run_name}_{config.env_id}_seed{config.seed}"
    run_dir = Path(config.output_root) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _rollout_to_tensors(rollout, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one rollout into batched tensors on the target device."""
    obs = torch.as_tensor(rollout.observations[None, ...], dtype=torch.float32, device=device)
    act = torch.as_tensor(rollout.actions[None, ...], dtype=torch.float32, device=device)
    return obs, act


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-batch metric dictionaries."""
    keys = metrics[0].keys()
    return {key: float(np.mean([entry[key] for entry in metrics])) for key in keys}


def _trajectory_smoothness(hidden_states: np.ndarray) -> float:
    """Score hidden trajectories by penalizing large second-order temporal differences."""
    if len(hidden_states) < 3:
        return 1.0
    step_delta = np.diff(hidden_states, axis=0)
    jerk = np.diff(step_delta, axis=0)
    return float(np.exp(-np.mean(np.linalg.norm(jerk, axis=1)) / (np.mean(np.linalg.norm(step_delta, axis=1)) + 1e-6)))


def _position_decoder(hidden_states: np.ndarray, positions: np.ndarray) -> tuple[Ridge, dict[str, float], np.ndarray]:
    """Fit a simple linear decoder from hidden states to 2-D position."""
    split = max(int(0.7 * len(hidden_states)), 4)
    split = min(split, len(hidden_states) - 1)
    decoder = Ridge(alpha=1.0)
    decoder.fit(hidden_states[:split], positions[:split])
    decoded = decoder.predict(hidden_states)
    holdout_decoded = decoded[split:]
    holdout_true = positions[split:]
    diag = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) + 1e-6
    return decoder, {
        "decoder_r2": float(r2_score(holdout_true, holdout_decoded, multioutput="variance_weighted")),
        "decoder_normalized_rmse": float(np.sqrt(mean_squared_error(holdout_true, holdout_decoded)) / diag),
    }, decoded


def _hidden_embedding(hidden_states: np.ndarray, seed: int) -> tuple[np.ndarray, str, float]:
    """Compute a 2-D embedding for visualization, preferring UMAP when available."""
    sample_limit = min(len(hidden_states), 1500)
    if len(hidden_states) > sample_limit:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(hidden_states), size=sample_limit, replace=False))
        hidden_states = hidden_states[keep]
    try:
        import umap.umap_ as umap
    except ImportError:
        embedding = PCA(n_components=2).fit_transform(hidden_states)
        explained = float(PCA(n_components=min(3, hidden_states.shape[1], len(hidden_states))).fit(hidden_states).explained_variance_ratio_[:3].sum())
        return embedding, "pca", explained
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=seed)
    embedding = reducer.fit_transform(hidden_states)
    explained = float(PCA(n_components=min(3, hidden_states.shape[1], len(hidden_states))).fit(hidden_states).explained_variance_ratio_[:3].sum())
    return embedding, "umap", explained


def _evaluate_epoch(
    model: HippocampalPredictiveRNN,
    spec: EnvironmentSpec2D,
    config: ExperimentConfig,
    tile_map,
    *,
    epoch: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run validation and produce both scalar metrics and plotting artifacts."""
    val_batches: list[dict[str, float]] = []
    for seq_idx in range(config.val_sequences_per_epoch):
        val_rollout = collect_rollout_2d(
            spec,
            config.sequence_length,
            seed=config.seed + 40_000 + epoch * 100 + seq_idx,
            observation_mode=config.observation_mode,
            tile_map=tile_map,
            include_head_direction=config.include_head_direction,
            agent_params=config.agent_params,
        )
        obs_val, act_val = _rollout_to_tensors(val_rollout, model.device)
        val_batches.append(model.evaluate_on_batch(obs_val, act_val))
    val_losses = _mean_metrics(val_batches)

    eval_rollout = collect_rollout_2d(
        spec,
        config.eval_rollout_steps,
        seed=config.seed + 50_000 + epoch,
        observation_mode=config.observation_mode,
        tile_map=tile_map,
        include_head_direction=config.include_head_direction,
        agent_params=config.agent_params,
    )
    obs_eval, act_eval = _rollout_to_tensors(eval_rollout, model.device)
    pred, target, hidden = model.predict_batch(obs_eval, act_eval, reduce_hidden=None, no_grad=True)

    pred_np = pred.squeeze(0).detach().cpu().numpy()
    target_np = target.squeeze(0).detach().cpu().numpy()
    hidden_np = hidden.squeeze(0).detach().cpu().numpy()
    positions = eval_rollout.positions[: len(hidden_np)]

    decoder, decoder_metrics, decoded_positions = _position_decoder(hidden_np, positions)
    tuning = fraction_spatially_tuned(hidden_np, positions, grid_size=20, spatial_information_threshold=0.1)
    norms = np.linalg.norm(hidden_np, axis=1)
    embedding, embedding_method, variance_explained = _hidden_embedding(hidden_np, config.seed + epoch)

    _, sleep_hidden = model.spontaneous(config.replay_steps, reduce_hidden=None)
    sleep_hidden_np = sleep_hidden.squeeze(0).detach().cpu().numpy()
    replay_positions = decoder.predict(sleep_hidden_np)

    visual_size = int(tile_map.visual_vector_size)
    sample_index = int(min(len(pred_np) - 1, max(len(pred_np) // 2, 0)))
    pred_patch = pred_np[sample_index, :visual_size].reshape(tile_map.config.patch_size, tile_map.config.patch_size, tile_map.config.channels)
    true_patch = target_np[sample_index, :visual_size].reshape(tile_map.config.patch_size, tile_map.config.patch_size, tile_map.config.channels)

    metrics = {
        "val_prediction_loss": float(val_losses["prediction_loss"]),
        "val_rollout_loss": float(val_losses["rollout_loss"]),
        "val_total_loss": float(val_losses["loss"]),
        "decoder_r2": float(decoder_metrics["decoder_r2"]),
        "decoder_normalized_rmse": float(decoder_metrics["decoder_normalized_rmse"]),
        "fraction_spatially_tuned": float(tuning["fraction"]),
        "mean_spatial_information": float(np.mean(tuning["spatial_information"])),
        "pca_variance_explained": float(variance_explained),
        "hidden_norm_mean": float(np.mean(norms)),
        "hidden_norm_std": float(np.std(norms)),
        "trajectory_smoothness": float(_trajectory_smoothness(hidden_np)),
        "visual_patch_mae": float(np.mean(np.abs(pred_patch - true_patch))),
        "gpu_memory_allocated_mb": float(gpu_memory_snapshot(model.device)["memory_allocated_mb"]),
    }
    artifacts = {
        "pred_patch": pred_patch,
        "true_patch": true_patch,
        "decoded_positions": decoded_positions,
        "true_positions": positions,
        "replay_positions": replay_positions,
        "hidden_embedding": embedding,
        "embedding_method": embedding_method,
        "tuning_curves": tuning["tuning_curves"],
        "spatial_information": tuning["spatial_information"],
    }
    return metrics, artifacts


def _append_history_row(csv_path: Path, row: Mapping[str, float], *, write_header: bool) -> None:
    """Append one metrics row to the CSV training log."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_single_experiment(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    spec: EnvironmentSpec2D | None = None,
) -> RunResult:
    """Train one visual predictive-RNN run and write logs, checkpoints, and plots."""

    config = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    if spec is None:
        suite = build_suite_2d()
        spec = suite[config.env_id]

    set_global_seeds(config.seed)
    run_id, run_dir = _make_run_dir(config)
    _write_json(run_dir / "config.json", asdict(config))

    tile_map = build_tile_map(spec, config.tile_map)
    model = HippocampalPredictiveRNN(config.model)
    dashboard = DashboardLogger(config.dashboard, run_dir, run_id)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    epochs_without_improvement = 0
    global_step = 0
    csv_path = run_dir / "logs" / "metrics.csv"

    progress = tqdm(range(1, config.epochs + 1), desc=f"train:{spec.env_id}", leave=True)
    final_artifacts: dict[str, Any] | None = None

    try:
        for epoch in progress:
            train_metrics: list[dict[str, float]] = []
            model.train()
            batch_metrics: list[dict[str, float]] = []
            batch_count = 0

            for seq_idx in range(config.train_sequences_per_epoch):
                rollout = collect_rollout_2d(
                    spec,
                    config.sequence_length,
                    seed=config.seed + epoch * 10_000 + seq_idx,
                    observation_mode=config.observation_mode,
                    tile_map=tile_map,
                    include_head_direction=config.include_head_direction,
                    agent_params=config.agent_params,
                )
                obs, act = _rollout_to_tensors(rollout, model.device)
                if batch_count == 0:
                    model.zero_grad()
                batch_metrics.append(
                    model.backward_on_batch(
                        obs,
                        act,
                        loss_scale=1.0 / float(config.batch_size),
                    )
                )
                batch_count += 1
                if batch_count == config.batch_size or seq_idx == config.train_sequences_per_epoch - 1:
                    # Gradients are accumulated over multiple short rollouts to emulate batch training.
                    model.optimizer_step(gradient_clip=config.gradient_clip)
                    optimizer_metrics = _mean_metrics(batch_metrics)
                    train_metrics.append(optimizer_metrics)
                    global_step += 1
                    if global_step % dashboard.update_every_n_steps == 0:
                        dashboard.log_scalars(
                            {
                                "train/loss": optimizer_metrics["loss"],
                                "train/prediction_loss": optimizer_metrics["prediction_loss"],
                                "train/rollout_loss": optimizer_metrics["rollout_loss"],
                            },
                            global_step,
                        )
                    batch_count = 0
                    batch_metrics = []

            train_epoch = _mean_metrics(train_metrics)
            eval_metrics, eval_artifacts = _evaluate_epoch(model, spec, config, tile_map, epoch=epoch)
            final_artifacts = eval_artifacts

            epoch_row = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "train_loss": train_epoch["loss"],
                "train_prediction_loss": train_epoch["prediction_loss"],
                "train_rollout_loss": train_epoch["rollout_loss"],
                "train_latent_loss": train_epoch["latent_loss"],
                **eval_metrics,
            }
            history.append(epoch_row)
            _append_history_row(csv_path, epoch_row, write_header=(epoch == 1))
            _write_json(run_dir / "logs" / "history.json", {"history": history})

            progress.set_postfix(
                train_loss=f"{epoch_row['train_prediction_loss']:.4f}",
                val_loss=f"{epoch_row['val_prediction_loss']:.4f}",
                r2=f"{epoch_row['decoder_r2']:.3f}",
            )

            dashboard.log_scalars(
                {
                    "epoch/train_prediction_loss": epoch_row["train_prediction_loss"],
                    "epoch/val_prediction_loss": epoch_row["val_prediction_loss"],
                    "epoch/val_rollout_loss": epoch_row["val_rollout_loss"],
                    "epoch/decoder_r2": epoch_row["decoder_r2"],
                    "epoch/fraction_spatially_tuned": epoch_row["fraction_spatially_tuned"],
                    "epoch/trajectory_smoothness": epoch_row["trajectory_smoothness"],
                },
                epoch,
            )

            if epoch % dashboard.update_every_n_steps == 0:
                visual_fig, _ = plot_visual_prediction(eval_artifacts["true_patch"], eval_artifacts["pred_patch"])
                dashboard.log_figure("visual/prediction_vs_truth", visual_fig, epoch)

                trajectory_fig, _ = plot_trajectory_reconstruction(
                    eval_artifacts["true_positions"],
                    eval_artifacts["decoded_positions"],
                    eval_artifacts["replay_positions"],
                )
                dashboard.log_figure("trajectory/decoded_vs_true", trajectory_fig, epoch)

                embedding_fig, _ = plot_hidden_embedding(
                    eval_artifacts["hidden_embedding"],
                    eval_artifacts["true_positions"][: len(eval_artifacts["hidden_embedding"])],
                    method=eval_artifacts["embedding_method"],
                )
                dashboard.log_figure("hidden/embedding", embedding_fig, epoch)

            model.save_checkpoint(run_dir / "checkpoints" / "last.pt")
            if epoch_row["val_prediction_loss"] < best_val_loss:
                best_val_loss = epoch_row["val_prediction_loss"]
                epochs_without_improvement = 0
                model.save_checkpoint(best_checkpoint)
            else:
                epochs_without_improvement += 1

            if epoch % config.checkpoint_every_n_epochs == 0:
                model.save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

            if epochs_without_improvement >= config.early_stopping_patience:
                break
    finally:
        progress.close()
        dashboard.close()

    if final_artifacts is None:
        raise RuntimeError("Training loop finished without evaluation artifacts.")

    final_artifacts["best_checkpoint"] = str(best_checkpoint)
    summary = generate_post_run_analysis(run_dir, history, final_artifacts)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "env_id": config.env_id,
            "seed": config.seed,
            "best_checkpoint": str(best_checkpoint),
            "history_rows": len(history),
            "device": str(model.device),
        },
    )
    return RunResult(
        run_id=run_id,
        run_dir=str(run_dir),
        best_checkpoint=str(best_checkpoint),
        history=history,
        summary=summary,
    )


__all__ = [
    "DashboardConfig",
    "ExperimentConfig",
    "RunResult",
    "run_single_experiment",
    "set_global_seeds",
]
