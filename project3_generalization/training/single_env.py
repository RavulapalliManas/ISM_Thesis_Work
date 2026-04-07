from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from project3_generalization.environments.suite_2d import EnvironmentSpec2D, build_suite_2d, collect_rollout_2d
from project3_generalization.evaluation.metrics import (
    BG1_trials_to_criterion,
    RG1_sRSA,
    fraction_spatially_tuned,
    participation_ratio,
    replay_quality,
)
from project3_generalization.models.hippocampal_module import HippocampalConfig, HippocampalPredictiveRNN


@dataclass
class SingleEnvironmentConfig:
    total_steps: int = 200_000
    sequence_length: int = 256
    evaluation_interval: int = 10_000
    evaluation_rollout_steps: int = 4_000
    replay_rollout_steps: int = 500
    replay_noise_std: float = 0.03
    spatial_information_threshold: float = 0.1
    tuning_grid_size: int = 20
    seeds: tuple[int, ...] = (0, 1, 2)
    checkpoint_dir: str | Path | None = None
    validate_lshape_first: bool = False
    model_config: HippocampalConfig = field(default_factory=HippocampalConfig)


@dataclass
class SingleEnvironmentResult:
    env_id: str
    seed: int
    history: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    steps_to_criterion: float
    model: HippocampalPredictiveRNN
    checkpoint_path: str | None = None


def _rollout_to_batch_tensors(rollout: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(rollout.observations[None, ...], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions[None, ...], dtype=torch.float32, device=device)
    return observations, actions


def _history_steps_to_criterion(history: Sequence[Mapping[str, Any]], criterion: float = 0.9) -> float:
    curve = np.asarray([entry["sRSA"] for entry in history], dtype=float)
    checkpoint_steps = np.asarray([entry["trained_steps"] for entry in history], dtype=float)
    asymptote = np.nanmax(curve)
    threshold = criterion * asymptote
    reached = np.flatnonzero(curve >= threshold)
    if reached.size == 0:
        return float(np.inf)
    return float(checkpoint_steps[reached[0]])


def evaluate_model_in_environment(
    model: HippocampalPredictiveRNN,
    spec: EnvironmentSpec2D,
    config: SingleEnvironmentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
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

    return {
        "sRSA": RG1_sRSA(hidden_np, positions),
        "fraction_spatially_tuned": tuning["fraction"],
        "participation_ratio": participation_ratio(hidden_np),
        "replay_quality": replay_quality(hidden_np, positions, spontaneous_hidden_np, extent=extent),
        "peak_bins": tuning["peak_bins"],
        "tuned_mask": tuning["tuned_mask"],
        "spatial_information": tuning["spatial_information"],
        "tuning_curves": tuning["tuning_curves"],
        "occupancy": tuning["occupancy"],
        "hidden_states": hidden_np,
        "positions": positions,
    }


def train_single_environment(
    spec: EnvironmentSpec2D,
    config: SingleEnvironmentConfig,
    *,
    seed: int,
    model: HippocampalPredictiveRNN | None = None,
    ewc_regularizer: Any | None = None,
    frozen_readout_only: bool = False,
) -> SingleEnvironmentResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model is None:
        model = HippocampalPredictiveRNN(config.model_config)
    if frozen_readout_only:
        model.freeze_all_but_readout()
    else:
        model.unfreeze_all()

    history: list[dict[str, Any]] = []
    trained_steps = 0
    batch_index = 0

    while trained_steps < config.total_steps:
        current_length = min(config.sequence_length, config.total_steps - trained_steps)
        rollout = collect_rollout_2d(spec, current_length, seed=seed + batch_index)
        obs, act = _rollout_to_batch_tensors(rollout, model.device)
        ewc_penalty = None if ewc_regularizer is None else ewc_regularizer.penalty(model)
        batch_metrics = model.train_on_batch(obs, act, ewc_penalty=ewc_penalty)
        trained_steps += current_length
        batch_index += 1

        if trained_steps % config.evaluation_interval == 0 or trained_steps >= config.total_steps:
            eval_metrics = evaluate_model_in_environment(model, spec, config, seed=seed + 10_000 + batch_index)
            eval_metrics.update(batch_metrics)
            eval_metrics["trained_steps"] = trained_steps
            history.append(eval_metrics)

    if not history:
        final_metrics = evaluate_model_in_environment(model, spec, config, seed=seed + 99_999)
        final_metrics["trained_steps"] = trained_steps
        history.append(final_metrics)

    checkpoint_path = None
    if config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_dir / f"{spec.env_id}_seed{seed}.pt")
        model.save_checkpoint(checkpoint_path)

    steps_to_criterion = _history_steps_to_criterion(history)
    history[-1]["steps_to_criterion"] = steps_to_criterion

    return SingleEnvironmentResult(
        env_id=spec.env_id,
        seed=seed,
        history=history,
        final_metrics=history[-1],
        steps_to_criterion=steps_to_criterion,
        model=model,
        checkpoint_path=checkpoint_path,
    )


def run_single_environment_suite(
    specs: Sequence[EnvironmentSpec2D] | None = None,
    config: SingleEnvironmentConfig | None = None,
) -> list[SingleEnvironmentResult]:
    if specs is None:
        specs = list(build_suite_2d().values())
    if config is None:
        config = SingleEnvironmentConfig()

    results: list[SingleEnvironmentResult] = []

    if config.validate_lshape_first:
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
            "mean_steps_to_criterion": float(np.mean([result.steps_to_criterion for result in env_results])),
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
