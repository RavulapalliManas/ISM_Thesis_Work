from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from project3_generalization.environments.suite_2d import EnvironmentSpec2D
from project3_generalization.models.hippocampal_module import HippocampalConfig
from project3_generalization.training.curriculum import CurriculumConfig, run_curriculum
from project3_generalization.training.single_env import SingleEnvironmentConfig, train_single_environment


@dataclass
class AblationConfig:
    recurrence_scales: tuple[float, ...] = (0.3, 0.7, 1.0, 1.5)
    seeds: tuple[int, ...] = (0,)
    mode: str = "single_env"
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)
    single_env_config: SingleEnvironmentConfig = field(default_factory=SingleEnvironmentConfig)
    target_env_id: str | None = None


def run_recurrence_ablation(
    specs: Sequence[EnvironmentSpec2D],
    *,
    config: AblationConfig | None = None,
    curriculum_order: Sequence[str] | None = None,
    scratch_reference: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[float, list[dict[str, Any]]]:
    if config is None:
        config = AblationConfig()
    spec_lookup = {spec.env_id: spec for spec in specs}
    results: dict[float, list[dict[str, Any]]] = {}

    for scale in config.recurrence_scales:
        scale_results: list[dict[str, Any]] = []
        for seed in config.seeds:
            scaled_model_config = HippocampalConfig(**vars(config.curriculum_config.model_config))
            scaled_model_config.recurrence_scale = scale

            if config.mode == "curriculum":
                if curriculum_order is None:
                    raise ValueError("curriculum_order must be provided for curriculum-mode recurrence ablations.")
                curriculum_config = CurriculumConfig(**vars(config.curriculum_config))
                curriculum_config.model_config = scaled_model_config
                scale_results.append(
                    run_curriculum(
                        specs,
                        curriculum_order=curriculum_order,
                        config=curriculum_config,
                        scratch_reference=scratch_reference,
                        seed=seed,
                    )
                )
            elif config.mode == "single_env":
                if config.target_env_id is None:
                    raise ValueError("target_env_id must be set for single_env recurrence ablations.")
                single_env_config = SingleEnvironmentConfig(**vars(config.single_env_config))
                single_env_config.model_config = scaled_model_config
                scale_results.append(
                    {
                        "result": train_single_environment(
                            spec_lookup[config.target_env_id],
                            single_env_config,
                            seed=seed,
                        )
                    }
                )
            else:
                raise ValueError(f"Unknown ablation mode `{config.mode}`.")
        results[scale] = scale_results
    return results


__all__ = [
    "AblationConfig",
    "run_recurrence_ablation",
]
