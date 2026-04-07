from __future__ import annotations

import argparse
import json
from pathlib import Path

from project3_generalization.environments.similarity import SimilarityConfig, compute_similarity_matrix
from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.training.curriculum import CurriculumConfig, greedy_similarity_order, run_curriculum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-module cortical+hippocampal curriculum.")
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-environment", type=int, default=50_000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--evaluation-interval", type=int, default=5_000)
    parser.add_argument("--similarity-steps", type=int, default=100_000)
    parser.add_argument("--similarity-grid", type=int, default=50)
    parser.add_argument("--baseline-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    suite = build_suite_2d()
    specs = list(suite.values()) if args.envs is None else [suite[env_id] for env_id in args.envs]
    sim_matrix, sim_ids = compute_similarity_matrix(
        specs,
        config=SimilarityConfig(num_steps=args.similarity_steps, grid_size=args.similarity_grid),
        seed=args.seed,
    )
    order = greedy_similarity_order(sim_ids, sim_matrix)
    scratch_reference = None
    if args.baseline_json is not None:
        scratch_reference = json.loads(Path(args.baseline_json).read_text())

    config = CurriculumConfig(
        steps_per_environment=args.steps_per_environment,
        sequence_length=args.sequence_length,
        evaluation_interval=args.evaluation_interval,
    )
    results = run_curriculum(
        specs,
        curriculum_order=order,
        config=config,
        scratch_reference=scratch_reference,
        seed=args.seed,
        use_cortical_prior=True,
    )
    serializable = {
        "order": results["order"],
        "reexposure": results["reexposure"],
        "population_overlap_index": results["population_overlap_index"],
        "transfer_efficiency_matrix": results["transfer_efficiency_matrix"].tolist(),
        "environment_results": [
            {
                "env_id": entry["env_id"],
                "zero_shot_sRSA": entry["zero_shot_sRSA"],
                "final_sRSA": entry["result"].final_metrics["sRSA"],
                "steps_to_criterion": entry["result"].steps_to_criterion,
            }
            for entry in results["environment_results"]
        ],
    }
    print(json.dumps(serializable, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
