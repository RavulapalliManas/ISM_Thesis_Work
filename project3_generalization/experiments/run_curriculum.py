from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from project3_generalization.environments.similarity import SimilarityConfig, compute_similarity_matrix
from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.hardware import gpu_memory_snapshot
from project3_generalization.training.curriculum import (
    CurriculumConfig,
    greedy_similarity_order,
    random_curriculum_order,
    run_curriculum,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 3 curriculum training.")
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--ordering", choices=["similarity", "random"], default="similarity")
    parser.add_argument("--start-env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-environment", type=int, default=50_000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--evaluation-interval", type=int, default=5_000)
    parser.add_argument("--similarity-steps", type=int, default=25_000)
    parser.add_argument("--similarity-grid", type=int, default=30)
    parser.add_argument("--use-ewc", action="store_true")
    parser.add_argument("--frozen-readout-only", action="store_true")
    parser.add_argument("--baseline-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def _load_baseline_summary(path: str | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def main() -> None:
    start = time.perf_counter()
    args = _parse_args()
    suite = build_suite_2d()
    default_envs = ["A1_square", "B1_l_shape", "C3_barrier_gap"]
    specs = [suite[env_id] for env_id in (default_envs if args.envs is None else args.envs)]
    env_ids = [spec.env_id for spec in specs]
    sim_matrix, sim_ids = compute_similarity_matrix(
        specs,
        config=SimilarityConfig(num_steps=args.similarity_steps, grid_size=args.similarity_grid),
        seed=args.seed,
    )
    if args.ordering == "similarity":
        order = greedy_similarity_order(sim_ids, sim_matrix, start_env=args.start_env)
    else:
        order = random_curriculum_order(sim_ids, seed=args.seed)

    config = CurriculumConfig(
        steps_per_environment=args.steps_per_environment,
        sequence_length=args.sequence_length,
        evaluation_interval=args.evaluation_interval,
    )
    results = run_curriculum(
        specs,
        curriculum_order=order,
        config=config,
        scratch_reference=_load_baseline_summary(args.baseline_json),
        seed=args.seed,
        use_ewc=args.use_ewc,
        frozen_readout_only=args.frozen_readout_only,
    )
    serializable = {
        "order": results["order"],
        "transfer_efficiency_matrix": results["transfer_efficiency_matrix"].tolist(),
        "reexposure": results["reexposure"],
        "population_overlap_index": results["population_overlap_index"],
        "environment_results": [
            {
                "env_id": entry["env_id"],
                "zero_shot_sRSA": entry["zero_shot_sRSA"],
                "final_sRSA": entry["result"].final_metrics["sRSA"],
                "steps_to_criterion": entry["result"].steps_to_criterion,
            }
            for entry in results["environment_results"]
        ],
        "runtime_seconds": float(time.perf_counter() - start),
        "gpu_memory": gpu_memory_snapshot(),
    }
    print(json.dumps(serializable, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
