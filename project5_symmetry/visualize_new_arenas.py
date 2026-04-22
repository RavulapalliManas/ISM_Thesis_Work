"""
Visualize the newer symmetry-condition arenas in ``project5_symmetry``.

This script renders the square 18x18 arena under each symmetry landmark layout:

- ``s4``: four-way rotational symmetry
- ``s2``: 180-degree rotational symmetry
- ``s1``: no rotational symmetry / four distinct quadrants

It writes one PNG per condition plus a combined overview figure.

Examples
--------
    PYTHONPATH=. python project5_symmetry/visualize_new_arenas.py
    PYTHONPATH=. python project5_symmetry/visualize_new_arenas.py --out outputs/new_arenas
    PYTHONPATH=. python project5_symmetry/visualize_new_arenas.py --conditions s4 s1 --tile-size 16
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from project5_symmetry.environments.arena import PixelObsWrapper, SymmetryArena, compute_H2
except ImportError as exc:
    raise ImportError(
        "Failed to import project5_symmetry arena dependencies. "
        "This codepath expects `gymnasium` and `minigrid` in the same Python "
        "environment you use to run the script. Having `gym` installed is not "
        "enough because `project5_symmetry/environments/arena.py` imports "
        "`gymnasium` directly.\n\n"
        "Try:\n"
        "  python3 -c \"import sys, gymnasium, minigrid; print(sys.executable); "
        "print(gymnasium.__file__); print(minigrid.__file__)\""
    ) from exc


DEFAULT_OUT = "project5_symmetry/env_viz/new_arenas"
DEFAULT_CONDITIONS = ("s4", "s2", "s1")
CONDITION_TITLES = {
    "s4": "S4: C4 rotational symmetry",
    "s2": "S2: 180-degree symmetry",
    "s1": "S1: asymmetric quadrants",
}


def build_env(
    condition: str,
    *,
    size: int = 18,
    fov: int = 7,
    seed: int = 0,
) -> PixelObsWrapper:
    """Construct a wrapped symmetry arena for one landmark condition."""
    return PixelObsWrapper(
        SymmetryArena(
            shape="square",
            size=size,
            U=0,
            F=fov,
            seed=seed,
            use_landmarks=True,
            symmetry_condition=condition,
        ),
        tile_size=1,
    )


def render_topdown(env: PixelObsWrapper, tile_size: int) -> np.ndarray:
    """Return a top-down RGB render of the full arena."""
    inner = env.unwrapped
    inner.reset()
    return inner.get_frame(highlight=False, tile_size=tile_size)


def _save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_single_condition(condition: str, out_dir: str, tile_size: int, seed: int, fov: int) -> dict[str, float]:
    """Render and save one condition image, returning summary metadata."""
    env = build_env(condition, size=18, fov=fov, seed=seed)
    img = render_topdown(env, tile_size=tile_size)
    h2 = compute_H2(env)
    n_landmark_tiles = len(env.unwrapped._get_landmark_tiles())

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(CONDITION_TITLES.get(condition, condition), fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(
        f"landmark tiles={n_landmark_tiles} | passable={len(env.unwrapped.passable_positions)} | "
        f"H2 mean={h2['mean']:.2f}",
        fontsize=9,
    )
    _save_figure(fig, os.path.join(out_dir, f"{condition}.png"))

    return {
        "condition": condition,
        "n_landmark_tiles": float(n_landmark_tiles),
        "n_passable": float(len(env.unwrapped.passable_positions)),
        "h2_mean": float(h2["mean"]),
    }


def save_overview(conditions: tuple[str, ...], out_dir: str, tile_size: int, seed: int, fov: int) -> None:
    """Save a combined overview panel for all requested conditions."""
    n_cols = min(3, len(conditions))
    n_rows = int(np.ceil(len(conditions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, condition in zip(axes, conditions):
        env = build_env(condition, size=18, fov=fov, seed=seed)
        img = render_topdown(env, tile_size=tile_size)
        h2 = compute_H2(env)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(CONDITION_TITLES.get(condition, condition), fontsize=12, fontweight="bold", pad=10)
        ax.text(
            0.5,
            -0.06,
            f"H2 mean={h2['mean']:.2f} | landmark tiles={len(env.unwrapped._get_landmark_tiles())}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    for ax in axes[len(conditions) :]:
        ax.axis("off")

    fig.suptitle("project5_symmetry: new arena layouts", fontsize=15, fontweight="bold", y=0.98)
    _save_figure(fig, os.path.join(out_dir, "overview.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output directory for PNG files. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        choices=sorted(CONDITION_TITLES),
        help="Symmetry conditions to render.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Tile size in rendered top-down images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed passed into SymmetryArena.",
    )
    parser.add_argument(
        "--fov",
        type=int,
        default=7,
        help="Agent field-of-view size used when computing observations/H2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = tuple(args.conditions)
    os.makedirs(args.out, exist_ok=True)

    summaries = []
    for condition in conditions:
        summaries.append(
            save_single_condition(
                condition=condition,
                out_dir=args.out,
                tile_size=args.tile_size,
                seed=args.seed,
                fov=args.fov,
            )
        )

    save_overview(
        conditions=conditions,
        out_dir=args.out,
        tile_size=args.tile_size,
        seed=args.seed,
        fov=args.fov,
    )

    print(f"Saved {len(conditions)} arena image(s) and an overview to {args.out}")
    for summary in summaries:
        print(
            f"  {summary['condition']}: landmark_tiles={int(summary['n_landmark_tiles'])}, "
            f"passable={int(summary['n_passable'])}, H2={summary['h2_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
