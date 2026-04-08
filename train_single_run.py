from __future__ import annotations

import argparse
import json
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one visual predictive-RNN experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/visual_single_run.example.json",
        help="Path to the JSON config for the run.",
    )
    parser.add_argument("--env-id", type=str, default=None, help="Optional environment override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional output root override.")
    return parser.parse_args()


def main() -> None:
    try:
        from project3_generalization.visual_rnn.train import ExperimentConfig, run_single_experiment
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        if missing in {"torch", "shapely", "sklearn", "matplotlib", "ratinabox"}:
            print(
                "Missing Python dependency detected while starting the visual predictive-RNN run.\n"
                f"Missing module: {missing}\n\n"
                "Install the visual-run dependencies first:\n"
                "  py -3.11 -m pip install -r requirements_visual_rnn.txt\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        raise

    args = _parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.env_id is not None:
        config.env_id = args.env_id
    if args.seed is not None:
        config.seed = args.seed
    if args.output_root is not None:
        config.output_root = args.output_root

    result = run_single_experiment(config)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "best_checkpoint": result.best_checkpoint,
                "summary": result.summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
