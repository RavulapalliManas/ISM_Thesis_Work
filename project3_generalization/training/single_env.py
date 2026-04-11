"""
File: project3_generalization/training/single_env.py

Description:
Single-environment training and evaluation loop for Project 3.

Role in system:
This module is the baseline optimization engine used directly by the baseline
runner and indirectly by curriculum and ablation experiments. It also handles
resource adaptation for constrained hardware and assembles the primary metrics
used throughout the package.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from project3_generalization.environments.similarity import SimilarityConfig, estimate_environment_structure
from project3_generalization.environments.suite_2d import EnvironmentSpec2D, build_suite_2d, collect_rollout_2d
from project3_generalization.evaluation.metrics import (
    current_environment_sr_error,
    fraction_spatially_tuned,
    participation_ratio,
    replay_quality,
    RG1_sRSA,
)
from project3_generalization.models.hippocampal_module import HippocampalConfig, HippocampalPredictiveRNN


@dataclass
class SingleEnvironmentConfig:
    """Configuration for one baseline predictive-RNN training run."""

    total_steps: int = 100_000
    sequence_length: int = 64
    min_sequence_length: int = 32
    rollout_batch_size: int = 2
    min_rollout_batch_size: int = 1
    rollout_workers: int = 1
    evaluation_interval: int = 10_000
    evaluation_rollout_steps: int = 2_000
    replay_rollout_steps: int = 400
    replay_noise_std: float = 0.03
    spatial_information_threshold: float = 0.1
    tuning_grid_size: int = 20
    hidden_state_sample_limit: int = 2_000
    srsa_max_samples: int = 2_000
    cka_batch_size: int = 256
    betti_max_points: int = 1_000
    sr_grid_size: int = 30
    sr_num_steps: int = 25_000
    gradient_clip: float | None = 1.0
    slow_batch_threshold_seconds: float = 8.0
    rolling_window_size: int = 3
    seeds: tuple[int, ...] = (0,)
    checkpoint_dir: str | Path | None = None
    validate_lshape_first: bool = False
    store_hidden_states: bool = True
    model_config: HippocampalConfig = field(default_factory=HippocampalConfig)


@dataclass
class SingleEnvironmentResult:
    """Training outputs and diagnostics for one environment/seed pair."""

    env_id: str
    seed: int
    history: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    steps_to_criterion: float
    model: HippocampalPredictiveRNN
    checkpoint_path: str | None = None
    runtime_summary: dict[str, Any] = field(default_factory=dict)
    resource_events: list[dict[str, Any]] = field(default_factory=list)


def _collect_rollout_job(
    spec: EnvironmentSpec2D,
    steps: int,
    seed: int,
) -> Any:
    """Worker helper for collecting one rollout in a subprocess."""
    return collect_rollout_2d(spec, steps, seed=seed)


def _rollout_to_batch_tensors(rollout: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a rollout object into batched observation/action tensors."""
    observations = torch.as_tensor(rollout.observations[None, ...], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions[None, ...], dtype=torch.float32, device=device)
    return observations, actions


def _rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a causal rolling average with edge padding."""
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _history_steps_to_criterion(
    history: Sequence[Mapping[str, Any]],
    *,
    criterion: float = 0.9,
    rolling_window_size: int = 3,
) -> float:
    """Estimate when a learning curve first reaches a fraction of its asymptote."""
    curve = np.asarray([entry["sRSA"] for entry in history], dtype=float)
    checkpoint_steps = np.asarray([entry["trained_steps"] for entry in history], dtype=float)
    smoothed = _rolling_average(curve, rolling_window_size)
    asymptote = np.nanmax(smoothed)
    threshold = criterion * asymptote
    reached = np.flatnonzero(smoothed >= threshold)
    if reached.size == 0:
        return float(np.inf)
    return float(checkpoint_steps[reached[0]])


def _subsample_hidden_logging(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample hidden states and positions before storing them in results."""
    if len(hidden_states) <= max_points:
        return hidden_states, positions
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(hidden_states), size=max_points, replace=False))
    return hidden_states[keep], positions[keep]


def evaluate_model_in_environment(
    model: HippocampalPredictiveRNN,
    spec: EnvironmentSpec2D,
    config: SingleEnvironmentConfig,
    *,
    seed: int,
    include_hidden_states: bool = False,
    include_tuning_details: bool = False,
    include_sr_error: bool = False,
) -> dict[str, Any]:
    """Evaluate one trained model in one environment and compute summary metrics."""
    rollout = collect_rollout_2d(spec, config.evaluation_rollout_steps, seed=seed)
    obs, act = _rollout_to_batch_tensors(rollout, model.device)
    _, _, hidden = model.predict_batch(obs, act, reduce_hidden="mean")
    hidden_np = hidden.detach().cpu().numpy()
    positions = rollout.positions[: hidden_np.shape[0]]

    tuning = fraction_spatially_tuned(
        hidden_np,
        positions,
        grid_size=config.tuning_grid_size,
        spatial_information_threshold=config.spatial_information_threshold,
    )
    extent = spec.build_environment().extent
    _, spontaneous_hidden = model.spontaneous(
        config.replay_rollout_steps,
        noise_std=config.replay_noise_std,
        reduce_hidden="mean",
    )
    spontaneous_hidden_np = spontaneous_hidden.detach().cpu().numpy()

    results: dict[str, Any] = {
        "sRSA": RG1_sRSA(hidden_np, positions, max_samples=config.srsa_max_samples, seed=seed),
        "fraction_spatially_tuned": tuning["fraction"],
        "participation_ratio": participation_ratio(hidden_np),
        "replay_quality": replay_quality(hidden_np, positions, spontaneous_hidden_np, extent=extent),
        "gpu_memory": model.memory_stats(),
    }

    if include_sr_error:
        # SR estimation is expensive, so it is only computed for final summaries by default.
        structure = estimate_environment_structure(
            spec,
            SimilarityConfig(
                num_steps=config.sr_num_steps,
                grid_size=config.sr_grid_size,
                num_workers=1,
                use_memmap=False,
            ),
            seed=seed,
        )
        true_sr = structure.load_successor_representation()
        results["sr_error"] = current_environment_sr_error(
            hidden_np,
            positions,
            true_sr,
            extent=extent,
            grid_size=config.sr_grid_size,
        )

    if include_tuning_details:
        results.update(
            {
                "peak_bins": tuning["peak_bins"],
                "tuned_mask": tuning["tuned_mask"],
                "spatial_information": tuning["spatial_information"],
                "tuning_curves": tuning["tuning_curves"],
                "occupancy": tuning["occupancy"],
            }
        )

    if include_hidden_states:
        hidden_log, position_log = _subsample_hidden_logging(
            hidden_np,
            positions,
            max_points=config.hidden_state_sample_limit,
            seed=seed,
        )
        results["hidden_states"] = hidden_log
        results["positions"] = position_log

    return results


def _mean_metric(metrics: Sequence[Mapping[str, float]], key: str) -> float:
    """Average one scalar metric across micro-batches."""
    return float(np.mean([entry[key] for entry in metrics])) if metrics else float("nan")


def _adapt_resource_settings(
    sequence_length: int,
    rollout_batch_size: int,
    store_hidden_states: bool,
    config: SingleEnvironmentConfig,
    reason: str,
) -> tuple[int, int, bool, dict[str, Any] | None]:
    """Relax resource usage after an OOM or a persistently slow batch."""
    if rollout_batch_size > config.min_rollout_batch_size:
        new_value = rollout_batch_size - 1
        return sequence_length, new_value, store_hidden_states, {
            "reason": reason,
            "adjustment": "rollout_batch_size",
            "old_value": rollout_batch_size,
            "new_value": new_value,
        }
    if sequence_length > config.min_sequence_length:
        new_value = max(config.min_sequence_length, sequence_length // 2)
        return new_value, rollout_batch_size, store_hidden_states, {
            "reason": reason,
            "adjustment": "sequence_length",
            "old_value": sequence_length,
            "new_value": new_value,
        }
    if store_hidden_states:
        return sequence_length, rollout_batch_size, False, {
            "reason": reason,
            "adjustment": "store_hidden_states",
            "old_value": True,
            "new_value": False,
        }
    return sequence_length, rollout_batch_size, store_hidden_states, None


def _is_oom_error(error: RuntimeError) -> bool:
    """Heuristically identify CUDA out-of-memory failures."""
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def train_single_environment(
    spec: EnvironmentSpec2D,
    config: SingleEnvironmentConfig,
    *,
    seed: int,
    model: HippocampalPredictiveRNN | None = None,
    ewc_regularizer: Any | None = None,
    frozen_readout_only: bool = False,
) -> SingleEnvironmentResult:
    """Train or fine-tune a predictive RNN on a single environment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model is None:
        model = HippocampalPredictiveRNN(config.model_config)
    if frozen_readout_only:
        model.freeze_all_but_readout()
    else:
        model.unfreeze_all()

    history: list[dict[str, Any]] = []
    resource_events: list[dict[str, Any]] = []
    trained_steps = 0
    batch_index = 0
    last_eval_steps = 0
    adaptive_sequence_length = config.sequence_length
    adaptive_rollout_batch_size = config.rollout_batch_size
    store_hidden_states = config.store_hidden_states
    peak_memory_allocated_mb = 0.0
    training_start = time.perf_counter()
    progress = tqdm(
        total=config.total_steps,
        desc=f"train:{spec.env_id}:seed{seed}",
        unit="step",
        leave=True,
    )
    rollout_executor = ProcessPoolExecutor(max_workers=config.rollout_workers) if config.rollout_workers > 1 else None

    try:
        while trained_steps < config.total_steps:
            current_length = min(adaptive_sequence_length, config.total_steps - trained_steps)
            remaining_steps = config.total_steps - trained_steps
            micro_lengths: list[int] = []
            while remaining_steps > 0 and len(micro_lengths) < adaptive_rollout_batch_size:
                micro_length = min(current_length, remaining_steps)
                micro_lengths.append(micro_length)
                remaining_steps -= micro_length
            batch_start = time.perf_counter()
            try:
                model.zero_grad()
                batch_metrics: list[dict[str, float]] = []
                if rollout_executor is not None:
                    futures = [
                        rollout_executor.submit(_collect_rollout_job, spec, micro_length, seed + batch_index + micro_idx)
                        for micro_idx, micro_length in enumerate(micro_lengths)
                    ]
                    rollouts = [future.result() for future in futures]
                else:
                    rollouts = [
                        collect_rollout_2d(spec, micro_length, seed=seed + batch_index + micro_idx)
                        for micro_idx, micro_length in enumerate(micro_lengths)
                    ]
                for rollout in rollouts:
                    obs, act = _rollout_to_batch_tensors(rollout, model.device)
                    ewc_penalty = None if ewc_regularizer is None else ewc_regularizer.penalty(model)
                    batch_metrics.append(
                        model.backward_on_batch(
                            obs,
                            act,
                            ewc_penalty=ewc_penalty,
                            loss_scale=1.0 / float(len(micro_lengths)),
                        )
                    )
                model.optimizer_step(gradient_clip=config.gradient_clip)
                batch_index += len(micro_lengths)
                trained_steps += int(sum(micro_lengths))
                progress.update(int(sum(micro_lengths)))
            except RuntimeError as error:
                if not _is_oom_error(error):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Back off on batch size, sequence length, or logging footprint before retrying.
                adaptive_sequence_length, adaptive_rollout_batch_size, store_hidden_states, event = _adapt_resource_settings(
                    adaptive_sequence_length,
                    adaptive_rollout_batch_size,
                    store_hidden_states,
                    config,
                    "oom",
                )
                if event is None:
                    raise
                resource_events.append(event)
                progress.set_postfix(
                    seq_len=adaptive_sequence_length,
                    batch=adaptive_rollout_batch_size,
                    mode="oom-retry",
                )
                continue

            batch_duration = time.perf_counter() - batch_start
            peak_memory_allocated_mb = max(peak_memory_allocated_mb, model.memory_stats()["max_memory_allocated_mb"])
            progress.set_postfix(
                loss=f"{_mean_metric(batch_metrics, 'loss'):.4f}",
                seq_len=adaptive_sequence_length,
                batch=adaptive_rollout_batch_size,
            )
            if batch_duration > config.slow_batch_threshold_seconds:
                adaptive_sequence_length, adaptive_rollout_batch_size, store_hidden_states, event = _adapt_resource_settings(
                    adaptive_sequence_length,
                    adaptive_rollout_batch_size,
                    store_hidden_states,
                    config,
                    "slowdown",
                )
                if event is not None:
                    event["batch_duration_seconds"] = float(batch_duration)
                    resource_events.append(event)

            if (trained_steps - last_eval_steps) >= config.evaluation_interval or trained_steps >= config.total_steps:
                eval_metrics = evaluate_model_in_environment(
                    model,
                    spec,
                    config,
                    seed=seed + 10_000 + batch_index,
                    include_hidden_states=False,
                    include_tuning_details=False,
                    include_sr_error=False,
                )
                eval_metrics.update(
                    {
                        "loss": _mean_metric(batch_metrics, "loss"),
                        "prediction_loss": _mean_metric(batch_metrics, "prediction_loss"),
                        "trained_steps": trained_steps,
                    }
                )
                history.append(eval_metrics)
                last_eval_steps = trained_steps
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=False)
        progress.close()

    final_metrics = evaluate_model_in_environment(
        model,
        spec,
        config,
        seed=seed + 99_999,
        include_hidden_states=store_hidden_states,
        include_tuning_details=True,
        include_sr_error=True,
    )
    final_metrics["trained_steps"] = trained_steps
    final_metrics["training_time_seconds"] = float(time.perf_counter() - training_start)

    checkpoint_path = None
    if config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_dir / f"{spec.env_id}_seed{seed}.pt")
        model.save_checkpoint(checkpoint_path)

    steps_to_criterion = _history_steps_to_criterion(
        history if history else [final_metrics],
        rolling_window_size=config.rolling_window_size,
    )
    final_metrics["steps_to_criterion"] = steps_to_criterion

    runtime_summary = {
        "training_time_seconds": float(time.perf_counter() - training_start),
        "peak_gpu_memory_allocated_mb": float(peak_memory_allocated_mb),
        "final_gpu_memory": model.memory_stats(),
        "adaptive_sequence_length": adaptive_sequence_length,
        "adaptive_rollout_batch_size": adaptive_rollout_batch_size,
        "store_hidden_states": bool(store_hidden_states),
    }

    return SingleEnvironmentResult(
        env_id=spec.env_id,
        seed=seed,
        history=history,
        final_metrics=final_metrics,
        steps_to_criterion=steps_to_criterion,
        model=model,
        checkpoint_path=checkpoint_path,
        runtime_summary=runtime_summary,
        resource_events=resource_events,
    )


def run_single_environment_suite(
    specs: Sequence[EnvironmentSpec2D] | None = None,
    config: SingleEnvironmentConfig | None = None,
) -> list[SingleEnvironmentResult]:
    """Run baseline training over a suite of environments and seeds."""
    if specs is None:
        specs = list(build_suite_2d().values())
    if config is None:
        config = SingleEnvironmentConfig()

    results: list[SingleEnvironmentResult] = []

    if config.validate_lshape_first:
        # The L-shape baseline acts as a scientific gate before broader transfer claims.
        l_shape = next((spec for spec in specs if spec.env_id == "B1_l_shape"), None)
        if l_shape is None:
            raise ValueError("validate_lshape_first=True but B1_l_shape is not in the suite.")
        l_shape_results = [
            train_single_environment(l_shape, config, seed=seed)
            for seed in config.seeds
        ]
        mean_lshape_srsa = float(np.mean([result.final_metrics["sRSA"] for result in l_shape_results]))
        if mean_lshape_srsa < 0.4:
            raise RuntimeError(
                f"L-shape checkpoint failed: mean final sRSA={mean_lshape_srsa:.3f} < 0.4."
            )
        results.extend(l_shape_results)

    for spec in specs:
        if config.validate_lshape_first and spec.env_id == "B1_l_shape":
            continue
        for seed in config.seeds:
            results.append(train_single_environment(spec, config, seed=seed))
    return results


def summarize_baseline_results(results: Sequence[SingleEnvironmentResult]) -> dict[str, dict[str, float]]:
    """Aggregate baseline results across seeds for each environment."""
    grouped: dict[str, list[SingleEnvironmentResult]] = {}
    for result in results:
        grouped.setdefault(result.env_id, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for env_id, env_results in grouped.items():
        summary[env_id] = {
            "mean_sRSA": float(np.mean([result.final_metrics["sRSA"] for result in env_results])),
            "mean_fraction_spatially_tuned": float(
                np.mean([result.final_metrics["fraction_spatially_tuned"] for result in env_results])
            ),
            "mean_participation_ratio": float(
                np.mean([result.final_metrics["participation_ratio"] for result in env_results])
            ),
            "mean_replay_quality": float(np.mean([result.final_metrics["replay_quality"] for result in env_results])),
            "mean_sr_error": float(np.mean([result.final_metrics.get("sr_error", np.nan) for result in env_results])),
            "mean_steps_to_criterion": float(np.mean([result.steps_to_criterion for result in env_results])),
            "mean_training_time_seconds": float(
                np.mean([result.runtime_summary.get("training_time_seconds", np.nan) for result in env_results])
            ),
            "peak_gpu_memory_allocated_mb": float(
                np.max([result.runtime_summary.get("peak_gpu_memory_allocated_mb", 0.0) for result in env_results])
            ),
        }
    return summary


__all__ = [
    "SingleEnvironmentConfig",
    "SingleEnvironmentResult",
    "evaluate_model_in_environment",
    "run_single_environment_suite",
    "summarize_baseline_results",
    "train_single_environment",
]
