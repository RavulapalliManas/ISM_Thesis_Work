from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from project3_generalization.environments.similarity import SimilarityConfig, compute_similarity_matrix
from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.hardware import (
    PhaseLogger,
    gpu_memory_snapshot,
    load_hardware_config,
    make_output_directory,
    write_json,
)
from project3_generalization.models.cortical_module import CorticalModuleConfig
from project3_generalization.models.hippocampal_module import HippocampalConfig
from project3_generalization.training.curriculum import CurriculumConfig, greedy_similarity_order, run_curriculum
from project3_generalization.training.single_env import SingleEnvironmentConfig, train_single_environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one hardware-constrained Project 3 experiment.")
    parser.add_argument("--mode", choices=["baseline", "curriculum"], default="baseline")
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ordering", choices=["similarity", "fixed"], default="similarity")
    parser.add_argument("--start-env", type=str, default=None)
    parser.add_argument("--hardware-config", type=str, default="config_hardware.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _default_envs(mode: str) -> list[str]:
    if mode == "curriculum":
        return ["A1_square", "B1_l_shape", "C3_barrier_gap"]
    return ["B1_l_shape"]


def _build_model_config(hardware_config, *, sequence_length: int, hidden_size: int) -> HippocampalConfig:
    return HippocampalConfig(
        hidden_size=hidden_size,
        truncation=sequence_length,
        chunk_length=sequence_length,
        use_amp=hardware_config.execution.amp_enabled,
        amp_dtype=hardware_config.execution.amp_dtype,
        gradient_checkpointing=hardware_config.execution.gradient_checkpointing,
    )


def main() -> None:
    args = _parse_args()
    hardware_config = load_hardware_config(args.hardware_config)
    suite = build_suite_2d()
    env_ids = _default_envs(args.mode) if args.envs is None else list(args.envs)
    specs = [suite[env_id] for env_id in env_ids]

    if args.mode == "curriculum" and len(specs) > hardware_config.curriculum.max_environments:
        raise ValueError(
            f"Hardware-constrained curriculum runs support at most {hardware_config.curriculum.max_environments} environments."
        )

    output_dir = Path(args.output_dir) if args.output_dir is not None else make_output_directory(
        hardware_config.output_root,
        mode=args.mode,
        env_ids=env_ids,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    phase_logger = PhaseLogger(device=device)

    with phase_logger.phase("setup"):
        baseline_model_config = _build_model_config(
            hardware_config,
            sequence_length=hardware_config.baseline.sequence_length,
            hidden_size=hardware_config.baseline.hidden_size,
        )
        baseline_config = SingleEnvironmentConfig(
            total_steps=hardware_config.baseline.total_steps,
            sequence_length=hardware_config.baseline.sequence_length,
            min_sequence_length=hardware_config.baseline.min_sequence_length,
            rollout_batch_size=hardware_config.execution.rollout_batch_size,
            min_rollout_batch_size=hardware_config.execution.min_rollout_batch_size,
            rollout_workers=hardware_config.execution.max_workers,
            evaluation_interval=hardware_config.baseline.evaluation_interval,
            evaluation_rollout_steps=hardware_config.baseline.evaluation_rollout_steps,
            replay_rollout_steps=hardware_config.baseline.replay_rollout_steps,
            hidden_state_sample_limit=hardware_config.baseline.hidden_state_sample_limit,
            srsa_max_samples=hardware_config.metrics.srsa_max_samples,
            cka_batch_size=hardware_config.metrics.cka_batch_size,
            betti_max_points=hardware_config.metrics.betti_max_points,
            sr_grid_size=hardware_config.metrics.sr_grid_size,
            sr_num_steps=hardware_config.similarity.num_steps,
            slow_batch_threshold_seconds=hardware_config.execution.slow_batch_threshold_seconds,
            seeds=(args.seed,),
            checkpoint_dir=output_dir / "checkpoints",
            store_hidden_states=hardware_config.baseline.store_hidden_states,
            model_config=baseline_model_config,
        )

    if args.mode == "baseline":
        with phase_logger.phase("training"):
            result = train_single_environment(specs[0], baseline_config, seed=args.seed)
        payload = {
            "mode": "baseline",
            "env_id": specs[0].env_id,
            "seed": args.seed,
            "sRSA": result.final_metrics["sRSA"],
            "sr_error": result.final_metrics.get("sr_error"),
            "training_time_seconds": result.runtime_summary.get("training_time_seconds"),
            "memory_usage": result.runtime_summary,
            "resource_events": result.resource_events,
            "checkpoint_path": result.checkpoint_path,
            "phase_runtime": phase_logger.records,
            "gpu_memory": gpu_memory_snapshot(device),
            "output_dir": str(output_dir),
            "device": str(device),
        }
    else:
        with phase_logger.phase("similarity"):
            similarity_config = SimilarityConfig(
                num_steps=hardware_config.similarity.num_steps,
                grid_size=hardware_config.similarity.grid_size,
                gamma=hardware_config.similarity.gamma,
                temporal_horizon=hardware_config.similarity.temporal_horizon,
                cg_tolerance=hardware_config.similarity.cg_tolerance,
                cg_max_iter=hardware_config.similarity.cg_max_iter,
                num_workers=hardware_config.execution.max_workers,
                use_memmap=hardware_config.similarity.use_memmap,
                memmap_dir=output_dir / "sr_memmap",
            )
            sim_matrix, sim_ids = compute_similarity_matrix(specs, config=similarity_config, seed=args.seed)
            order = sim_ids if args.ordering == "fixed" else greedy_similarity_order(sim_ids, sim_matrix, start_env=args.start_env)

        with phase_logger.phase("training"):
            curriculum_config = CurriculumConfig(
                steps_per_environment=hardware_config.curriculum.steps_per_environment,
                sequence_length=hardware_config.baseline.sequence_length,
                min_sequence_length=hardware_config.baseline.min_sequence_length,
                rollout_batch_size=hardware_config.execution.rollout_batch_size,
                min_rollout_batch_size=hardware_config.execution.min_rollout_batch_size,
                evaluation_interval=hardware_config.curriculum.evaluation_interval,
                evaluation_rollout_steps=hardware_config.baseline.evaluation_rollout_steps,
                replay_rollout_steps=hardware_config.baseline.replay_rollout_steps,
                reexposure_steps=hardware_config.curriculum.reexposure_steps,
                slow_batch_threshold_seconds=hardware_config.execution.slow_batch_threshold_seconds,
                rolling_window_size=hardware_config.curriculum.rolling_window_size,
                checkpoint_dir=str(output_dir / "checkpoints"),
                store_hidden_states=False,
                model_config=baseline_model_config,
                cortical_config=CorticalModuleConfig(
                    hippocampal_hidden_size=hardware_config.baseline.hidden_size,
                    cortical_hidden_size=hardware_config.two_module.cortical_hidden_size,
                    update_every_n_calls=hardware_config.two_module.update_every_n_calls,
                ),
            )
            results = run_curriculum(
                specs,
                curriculum_order=order,
                config=curriculum_config,
                seed=args.seed,
            )

        final_env_results = []
        training_time_seconds = 0.0
        peak_gpu_memory_allocated_mb = 0.0
        for entry in results["environment_results"]:
            result = entry["result"]
            final_env_results.append(
                {
                    "env_id": entry["env_id"],
                    "zero_shot_sRSA": entry["zero_shot_sRSA"],
                    "final_sRSA": result.final_metrics["sRSA"],
                    "sr_error": result.final_metrics.get("sr_error"),
                    "steps_to_criterion": result.steps_to_criterion,
                    "training_time_seconds": result.runtime_summary.get("training_time_seconds"),
                    "peak_gpu_memory_allocated_mb": result.runtime_summary.get("peak_gpu_memory_allocated_mb"),
                }
            )
            training_time_seconds += float(result.runtime_summary.get("training_time_seconds", 0.0))
            peak_gpu_memory_allocated_mb = max(
                peak_gpu_memory_allocated_mb,
                float(result.runtime_summary.get("peak_gpu_memory_allocated_mb", 0.0)),
            )

        transfer_matrix = results["transfer_efficiency_matrix"]
        finite_transfer = transfer_matrix[np.isfinite(transfer_matrix)] if hasattr(transfer_matrix, "__array__") else []
        payload = {
            "mode": "curriculum",
            "order": results["order"],
            "seed": args.seed,
            "environment_results": final_env_results,
            "transfer_efficiency_matrix": results["transfer_efficiency_matrix"].tolist(),
            "mean_transfer_efficiency": float(finite_transfer.mean()) if len(finite_transfer) else float("nan"),
            "reexposure": results["reexposure"],
            "training_time_seconds": training_time_seconds,
            "memory_usage": {
                "peak_gpu_memory_allocated_mb": peak_gpu_memory_allocated_mb,
                "gpu_memory": gpu_memory_snapshot(device),
            },
            "phase_runtime": phase_logger.records,
            "output_dir": str(output_dir),
        }

    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "phase_runtime.json", phase_logger.records)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
