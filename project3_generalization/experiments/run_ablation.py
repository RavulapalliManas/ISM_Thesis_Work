"""
File: project3_generalization/experiments/run_ablation.py

Description:
CLI entry point for recurrence-strength ablation experiments.

Role in system:
Converts command-line options into an `AblationConfig`, optionally computes a
curriculum order, and then reuses the shared ablation harness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from project3_generalization.environments.similarity import SimilarityConfig, compute_similarity_matrix
from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.hardware import gpu_memory_snapshot
from project3_generalization.training.ablations import AblationConfig, run_recurrence_ablation
from project3_generalization.training.curriculum import CurriculumConfig, greedy_similarity_order
from project3_generalization.training.single_env import SingleEnvironmentConfig


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for recurrence ablations."""
    parser = argparse.ArgumentParser(description="Run recurrence-strength ablations.")
    parser.add_argument("--mode", choices=["curriculum", "single_env"], default="single_env")
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--target-env", type=str, default="B1_l_shape")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--scales", nargs="*", type=float, default=[0.3, 0.7, 1.0, 1.5])
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Execute the ablation sweep and serialize a summary keyed by recurrence scale."""
    start = time.perf_counter()
    args = _parse_args()
    suite = build_suite_2d()
    specs = list(suite.values()) if args.envs is None else [suite[env_id] for env_id in args.envs]
    curriculum_order = None
    if args.mode == "curriculum":
        sim_matrix, sim_ids = compute_similarity_matrix(specs, config=SimilarityConfig(num_steps=20_000, grid_size=30), seed=args.seed)
        curriculum_order = greedy_similarity_order(sim_ids, sim_matrix)

    config = AblationConfig(
        recurrence_scales=tuple(args.scales),
        mode=args.mode,
        curriculum_config=CurriculumConfig(steps_per_environment=args.steps, sequence_length=args.sequence_length),
        single_env_config=SingleEnvironmentConfig(total_steps=args.steps, sequence_length=args.sequence_length),
        target_env_id=args.target_env,
    )
    results = run_recurrence_ablation(specs, config=config, curriculum_order=curriculum_order)
    serializable = {}
    for scale, scale_results in results.items():
        serializable[str(scale)] = [
            {
                "final_sRSA": (
                    entry["result"].final_metrics["sRSA"]
                    if "result" in entry
                    else entry["environment_results"][-1]["result"].final_metrics["sRSA"]
                )
            }
            for entry in scale_results
        ]
    serializable["runtime_seconds"] = float(time.perf_counter() - start)
    serializable["gpu_memory"] = gpu_memory_snapshot()
    print(json.dumps(serializable, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
