"""
Convenience launcher for the fast project5_symmetry training path.

Examples
--------
Dry-run the default RTX-oriented setup:
    PYTHONPATH=. python -m project5_symmetry.run_fast --dry-run

Run the Phase 0 gate with the balanced fast preset:
    PYTHONPATH=. python -m project5_symmetry.run_fast --phase 0

Run Phase 1 with larger per-seed batches:
    PYTHONPATH=. python -m project5_symmetry.run_fast --phase 1 --batch 32
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass


DEFAULT_OUT = "project5_symmetry/results_fast"


@dataclass(frozen=True)
class LaunchPreset:
    parallel_seeds: int
    batch: int
    anchor_subsample: int
    defer_srsa: bool
    num_workers: int


PRESETS: dict[str, LaunchPreset] = {
    "conservative": LaunchPreset(
        parallel_seeds=2,
        batch=16,
        anchor_subsample=32,
        defer_srsa=True,
        num_workers=0,
    ),
    "balanced": LaunchPreset(
        parallel_seeds=4,
        batch=16,
        anchor_subsample=32,
        defer_srsa=True,
        num_workers=0,
    ),
    "max": LaunchPreset(
        parallel_seeds=4,
        batch=32,
        anchor_subsample=32,
        defer_srsa=True,
        num_workers=0,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the fast GPU-oriented project5_symmetry trainer."
    )
    parser.add_argument(
        "--phase",
        default="0",
        choices=["0", "1", "2a", "2b", "4a", "4b", "all"],
        help="Experiment phase to run.",
    )
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=sorted(PRESETS),
        help="Starting point for fast-trainer settings.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output directory root for checkpoints, logs, and metrics.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA device id for CUDA_VISIBLE_DEVICES, e.g. 0.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Per-seed batch size override. Good first values: 16 or 32.",
    )
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=None,
        help="Number of seeds to train together in one process.",
    )
    parser.add_argument(
        "--anchor-subsample",
        type=int,
        default=None,
        help="Number of rollout anchors sampled per trajectory. Use 0 for full anchors.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override PRNN_NUM_WORKERS. Fast path usually wants 0.",
    )
    parser.add_argument(
        "--trainer",
        default="fast",
        choices=["fast", "legacy"],
        help="Trainer backend. Keep 'fast' for the new GPU path.",
    )
    parser.add_argument(
        "--live-srsa",
        action="store_true",
        help="Compute sRSA during training instead of deferring it until the end.",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip the Phase 0 gate when running later phases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the effective launch config and exit without training.",
    )
    return parser


def _effective_config(args: argparse.Namespace) -> dict[str, object]:
    preset = PRESETS[args.preset]
    return {
        "phase": args.phase,
        "preset": args.preset,
        "out": args.out,
        "gpu": args.gpu,
        "trainer": args.trainer,
        "parallel_seeds": args.parallel_seeds if args.parallel_seeds is not None else preset.parallel_seeds,
        "batch": args.batch if args.batch is not None else preset.batch,
        "anchor_subsample": (
            args.anchor_subsample if args.anchor_subsample is not None else preset.anchor_subsample
        ),
        "defer_srsa": (not args.live_srsa) if args.live_srsa else preset.defer_srsa,
        "num_workers": args.num_workers if args.num_workers is not None else preset.num_workers,
        "no_gate": args.no_gate,
    }


def _apply_env(config: dict[str, object]):
    if config["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    os.environ["PRNN_TRAINER"] = str(config["trainer"])
    os.environ["PRNN_PARALLEL_SEEDS"] = str(config["parallel_seeds"])
    os.environ["PRNN_PER_SEED_BATCH"] = str(config["batch"])
    os.environ["PRNN_ANCHOR_SUBSAMPLE"] = str(config["anchor_subsample"])
    os.environ["PRNN_DEFER_SRSA"] = "1" if config["defer_srsa"] else "0"
    os.environ["PRNN_NUM_WORKERS"] = str(config["num_workers"])


def _run_phase(config: dict[str, object]):
    from project5_symmetry.experiments.sweep import (
        ALL_CONDITIONS,
        PHASE1,
        PHASE2A,
        PHASE2B,
        PHASE4A,
        PHASE4B,
        run_phase0,
        run_sweep,
    )

    phase = str(config["phase"])
    out = str(config["out"])

    if phase == "0":
        run_phase0(out)
        return

    if not bool(config["no_gate"]) and not run_phase0(out):
        return

    phase_map = {
        "1": (PHASE1, "_phase1"),
        "2a": (PHASE2A, "_phase2a"),
        "2b": (PHASE2B, "_phase2b"),
        "4a": (PHASE4A, "_phase4a"),
        "4b": (PHASE4B, "_phase4b"),
        "all": (ALL_CONDITIONS, "_all"),
    }
    conditions, label = phase_map[phase]
    run_sweep(conditions, out, label=label)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    config = _effective_config(args)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    _apply_env(config)

    print("Launching project5_symmetry fast runner with config:")
    print(json.dumps(config, indent=2))
    _run_phase(config)


if __name__ == "__main__":
    main()
