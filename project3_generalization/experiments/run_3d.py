from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from project3_generalization.environments.suite_3d import (
    PlaceCells3D,
    SurfaceNavigator3D,
    VolumetricNavigator3D,
    build_suite_3d,
    simulate_navigator_3d,
)
from project3_generalization.evaluation.metrics import GG2_field_size_anisotropy
from project3_generalization.hardware import gpu_memory_snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight 3D environment simulations.")
    parser.add_argument("--env-id", type=str, default="3D_2_volumetric_room")
    parser.add_argument("--navigator", choices=["surface", "volume"], default="volume")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = _parse_args()
    suite = build_suite_3d()
    env = suite[args.env_id]
    navigator_cls = SurfaceNavigator3D if args.navigator == "surface" else VolumetricNavigator3D
    navigator_kwargs = {"alpha": args.alpha} if args.navigator == "surface" else {}
    positions, _ = simulate_navigator_3d(
        env,
        navigator_cls,
        args.steps,
        seed=args.seed,
        **navigator_kwargs,
    )

    place_cells = PlaceCells3D(env, n=32, alpha=args.alpha, seed=args.seed)
    rate_samples = place_cells.get_state(positions)
    grid_size = 10
    bounds = np.asarray(env.bounds)
    binned = np.zeros((rate_samples.shape[1], grid_size, grid_size, grid_size), dtype=float)
    xs = np.clip((positions[:, 0] / max(bounds[0], 1e-9) * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip((positions[:, 1] / max(bounds[1], 1e-9) * grid_size).astype(int), 0, grid_size - 1)
    zs = np.clip((positions[:, 2] / max(bounds[2], 1e-9) * grid_size).astype(int), 0, grid_size - 1)
    for t in range(len(positions)):
        binned[:, xs[t], ys[t], zs[t]] += rate_samples[t]

    anisotropy = GG2_field_size_anisotropy(binned, alpha_motion=args.alpha)
    result = {
        "env_id": env.env_id,
        "navigator": args.navigator,
        "alpha": args.alpha,
        "positions_mean": positions.mean(axis=0).tolist(),
        "field_size_anisotropy": anisotropy,
        "runtime_seconds": float(time.perf_counter() - start),
        "gpu_memory": gpu_memory_snapshot(),
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
