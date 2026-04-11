"""
File: project3_generalization/experiments/run_baselines.py

Description:
CLI entry point for single-environment baseline experiments.

Role in system:
Thin orchestration layer that turns command-line arguments into a baseline
training configuration, executes the shared training loop, and emits a compact
JSON summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.hardware import gpu_memory_snapshot
from project3_generalization.training.single_env import (
    SingleEnvironmentConfig,
    run_single_environment_suite,
    summarize_baseline_results,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline runner."""
    parser = argparse.ArgumentParser(description="Run single-environment baselines for Project 3.")
    parser.add_argument("--envs", nargs="*", default=None, help="Optional subset of environment ids.")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--validate-lshape-first", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the baseline experiment suite and optionally save a JSON summary."""
    start = time.perf_counter()
    args = _parse_args()
    suite = build_suite_2d()
    default_envs = ["B1_l_shape"]
    specs = [suite[env_id] for env_id in (default_envs if args.envs is None else args.envs)]
    config = SingleEnvironmentConfig(
        total_steps=args.total_steps,
        sequence_length=args.sequence_length,
        evaluation_interval=args.evaluation_interval,
        seeds=tuple(args.seeds),
        validate_lshape_first=args.validate_lshape_first,
    )
    results = run_single_environment_suite(specs, config)
    summary = summarize_baseline_results(results)
    payload = {
        "summary": summary,
        "runtime_seconds": float(time.perf_counter() - start),
        "gpu_memory": gpu_memory_snapshot(),
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
