from __future__ import annotations

import argparse
import json
from pathlib import Path

from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.training.single_env import (
    SingleEnvironmentConfig,
    run_single_environment_suite,
    summarize_baseline_results,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-environment baselines for Project 3.")
    parser.add_argument("--envs", nargs="*", default=None, help="Optional subset of environment ids.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--validate-lshape-first", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    suite = build_suite_2d()
    specs = list(suite.values()) if args.envs is None else [suite[env_id] for env_id in args.envs]
    config = SingleEnvironmentConfig(
        total_steps=args.total_steps,
        sequence_length=args.sequence_length,
        evaluation_interval=args.evaluation_interval,
        seeds=tuple(args.seeds),
        validate_lshape_first=args.validate_lshape_first,
    )
    results = run_single_environment_suite(specs, config)
    summary = summarize_baseline_results(results)

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
