from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from project3_generalization.environments.suite_2d import EnvironmentSpec2D, collect_rollout_2d
from project3_generalization.models.cortical_module import CorticalModuleConfig, CorticalRNNPrior
from project3_generalization.models.hippocampal_module import HippocampalConfig, HippocampalPredictiveRNN
from project3_generalization.training.single_env import (
    SingleEnvironmentConfig,
    evaluate_model_in_environment,
    train_single_environment,
)


@dataclass
class CurriculumConfig:
    steps_per_environment: int = 50_000
    sequence_length: int = 64
    min_sequence_length: int = 32
    rollout_batch_size: int = 2
    min_rollout_batch_size: int = 1
    evaluation_interval: int = 5_000
    evaluation_rollout_steps: int = 2_000
    replay_rollout_steps: int = 400
    replay_noise_std: float = 0.03
    spatial_information_threshold: float = 0.1
    tuning_grid_size: int = 20
    ewc_lambda: float = 10.0
    ewc_fisher_batches: int = 4
    reexposure_steps: int = 1_000
    slow_batch_threshold_seconds: float = 8.0
    rolling_window_size: int = 3
    store_hidden_states: bool = False
    checkpoint_dir: str | None = None
    model_config: HippocampalConfig = field(default_factory=HippocampalConfig)
    cortical_config: CorticalModuleConfig = field(default_factory=CorticalModuleConfig)

    def as_single_environment_config(self) -> SingleEnvironmentConfig:
        return SingleEnvironmentConfig(
            total_steps=self.steps_per_environment,
            sequence_length=self.sequence_length,
            min_sequence_length=self.min_sequence_length,
            rollout_batch_size=self.rollout_batch_size,
            min_rollout_batch_size=self.min_rollout_batch_size,
            rollout_workers=1,
            evaluation_interval=self.evaluation_interval,
            evaluation_rollout_steps=self.evaluation_rollout_steps,
            replay_rollout_steps=self.replay_rollout_steps,
            replay_noise_std=self.replay_noise_std,
            spatial_information_threshold=self.spatial_information_threshold,
            tuning_grid_size=self.tuning_grid_size,
            slow_batch_threshold_seconds=self.slow_batch_threshold_seconds,
            rolling_window_size=self.rolling_window_size,
            store_hidden_states=self.store_hidden_states,
            checkpoint_dir=self.checkpoint_dir,
            model_config=self.model_config,
        )


@dataclass
class EWCState:
    fisher: dict[str, torch.Tensor]
    reference: dict[str, torch.Tensor]


class EWCRegularizer:
    def __init__(self, lambda_: float = 0.0):
        self.lambda_ = float(lambda_)
        self.tasks: list[EWCState] = []

    def penalty(self, model: HippocampalPredictiveRNN) -> torch.Tensor | None:
        if self.lambda_ <= 0.0 or not self.tasks:
            return None
        named_parameters = dict(model.named_parameters())
        penalty = torch.zeros((), device=model.device)
        for task in self.tasks:
            for name, fisher in task.fisher.items():
                parameter = named_parameters[name]
                penalty = penalty + torch.sum(fisher.to(model.device) * (parameter - task.reference[name].to(model.device)) ** 2)
        return 0.5 * self.lambda_ * penalty

    def consolidate(
        self,
        model: HippocampalPredictiveRNN,
        spec: EnvironmentSpec2D,
        config: CurriculumConfig,
        *,
        seed: int,
    ) -> None:
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=model.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for batch_idx in range(config.ewc_fisher_batches):
            rollout_seed = seed + 200_000 + batch_idx
            rollout = collect_rollout_2d(spec, config.sequence_length, seed=rollout_seed)
            observations = torch.as_tensor(
                rollout.observations[np.newaxis, ...],
                dtype=torch.float32,
                device=model.device,
            )
            actions = torch.as_tensor(
                rollout.actions[np.newaxis, ...],
                dtype=torch.float32,
                device=model.device,
            )
            model.zero_grad()
            model.backward_on_batch(observations, actions)
            if model._amp_enabled:
                model.grad_scaler.unscale_(model.optimizer)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) / config.ewc_fisher_batches
        self.tasks.append(
            EWCState(
                fisher={name: value.detach().cpu() for name, value in fisher.items()},
                reference={name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad},
            )
        )


def greedy_similarity_order(
    env_ids: Sequence[str],
    sim_matrix: np.ndarray,
    *,
    start_env: str | None = None,
) -> list[str]:
    remaining = list(env_ids)
    order: list[str] = []
    if start_env is None:
        start_env = remaining[0]
    current = start_env
    order.append(current)
    remaining.remove(current)
    id_to_index = {env_id: idx for idx, env_id in enumerate(env_ids)}
    while remaining:
        current_idx = id_to_index[current]
        next_env = max(remaining, key=lambda env_id: sim_matrix[current_idx, id_to_index[env_id]])
        order.append(next_env)
        remaining.remove(next_env)
        current = next_env
    return order


def random_curriculum_order(env_ids: Sequence[str], *, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return list(rng.permutation(env_ids))


def _population_overlap_index(initial: Mapping[str, Any], final: Mapping[str, Any]) -> float:
    initial_mask = np.asarray(initial["tuned_mask"], dtype=bool)
    final_mask = np.asarray(final["tuned_mask"], dtype=bool)
    shared = initial_mask & final_mask
    if not np.any(shared):
        return 0.0
    initial_peaks = np.asarray(initial["peak_bins"])
    final_peaks = np.asarray(final["peak_bins"])
    same_peak = np.all(initial_peaks[shared] == final_peaks[shared], axis=1)
    return float(np.mean(same_peak))


def run_curriculum(
    specs: Sequence[EnvironmentSpec2D],
    *,
    curriculum_order: Sequence[str],
    config: CurriculumConfig | None = None,
    scratch_reference: Mapping[str, Mapping[str, float]] | None = None,
    seed: int = 0,
    use_ewc: bool = False,
    frozen_readout_only: bool = False,
    use_cortical_prior: bool = False,
) -> dict[str, Any]:
    if config is None:
        config = CurriculumConfig()
    spec_lookup = {spec.env_id: spec for spec in specs}
    single_env_config = config.as_single_environment_config()
    if use_cortical_prior:
        single_env_config.store_hidden_states = True

    model = HippocampalPredictiveRNN(config.model_config)
    regularizer = EWCRegularizer(config.ewc_lambda if use_ewc else 0.0)
    cortical_model = CorticalRNNPrior(config.cortical_config).to(model.device) if use_cortical_prior else None

    env_ids = list(curriculum_order)
    id_to_index = {env_id: idx for idx, env_id in enumerate(env_ids)}
    transfer_matrix = np.full((len(env_ids), len(env_ids)), np.nan, dtype=float)
    env_results: list[dict[str, Any]] = []

    first_env_reference: dict[str, Any] | None = None
    first_spec = spec_lookup[env_ids[0]]

    for idx, env_id in enumerate(env_ids):
        spec = spec_lookup[env_id]
        if cortical_model is not None and idx > 0:
            cortical_model.initialize_hippocampus(model)

        zero_shot = evaluate_model_in_environment(model, spec, single_env_config, seed=seed + idx + 50_000)
        result = train_single_environment(
            spec,
            single_env_config,
            seed=seed + idx * 1_000,
            model=model,
            ewc_regularizer=regularizer if use_ewc else None,
            frozen_readout_only=(frozen_readout_only and idx > 0),
        )
        model = result.model

        if idx == 0:
            first_env_reference = {
                "peak_bins": result.final_metrics["peak_bins"],
                "tuned_mask": result.final_metrics["tuned_mask"],
                "baseline_sRSA": result.final_metrics["sRSA"],
            }

        if idx > 0:
            prev_id = env_ids[idx - 1]
            scratch_steps = None
            if scratch_reference is not None and env_id in scratch_reference:
                scratch_steps = scratch_reference[env_id].get("mean_steps_to_criterion")
            transfer_efficiency = zero_shot["sRSA"]
            if scratch_steps is not None and np.isfinite(scratch_steps):
                transfer_efficiency = 1.0 - (result.steps_to_criterion / max(float(scratch_steps), 1.0))
            transfer_matrix[id_to_index[prev_id], id_to_index[env_id]] = transfer_efficiency

        if cortical_model is not None:
            cortical_hidden = torch.as_tensor(result.final_metrics["hidden_states"], dtype=torch.float32, device=model.device)
            cortical_model.train_on_hidden_sequence(cortical_hidden)

        if use_ewc:
            regularizer.consolidate(model, spec, config, seed=seed + idx * 10_000)

        env_results.append(
            {
                "env_id": env_id,
                "zero_shot_sRSA": zero_shot["sRSA"],
                "result": result,
            }
        )

    pre_reexposure = evaluate_model_in_environment(model, first_spec, single_env_config, seed=seed + 999_000)
    population_overlap = 0.0
    if first_env_reference is not None:
        population_overlap = _population_overlap_index(first_env_reference, pre_reexposure)

    reexposure_config = config.as_single_environment_config()
    reexposure_config.total_steps = config.reexposure_steps
    reexposure_config.evaluation_interval = config.reexposure_steps
    reexposure_result = train_single_environment(first_spec, reexposure_config, seed=seed + 999_100, model=model)
    post_reexposure = reexposure_result.final_metrics

    baseline_srsa = first_env_reference["baseline_sRSA"] if first_env_reference is not None else pre_reexposure["sRSA"]
    recovery_rate = (post_reexposure["sRSA"] - pre_reexposure["sRSA"]) / (
        max(baseline_srsa - pre_reexposure["sRSA"], 1e-12)
    )

    return {
        "order": list(curriculum_order),
        "environment_results": env_results,
        "transfer_efficiency_matrix": transfer_matrix,
        "reexposure": {
            "pre_sRSA": pre_reexposure["sRSA"],
            "post_sRSA": post_reexposure["sRSA"],
            "recovery_rate": float(recovery_rate),
        },
        "population_overlap_index": float(population_overlap),
        "final_first_environment_metrics": pre_reexposure,
        "model": model,
        "cortical_model": cortical_model,
    }


__all__ = [
    "CurriculumConfig",
    "EWCRegularizer",
    "greedy_similarity_order",
    "random_curriculum_order",
    "run_curriculum",
]
