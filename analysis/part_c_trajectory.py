import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
TRAJ_DIR = BASE / "project5_symmetry" / "results" / "symmetry_sweep" / "s4" / "trajectories"
METRICS_DIR = BASE / "results" / "metrics"
FIG_DIR = BASE / "results" / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def coord_bounds(positions):
    return (
        int(np.min(positions[:, 0])),
        int(np.max(positions[:, 0])),
        int(np.min(positions[:, 1])),
        int(np.max(positions[:, 1])),
    )


def rot(coord, bounds, k=1):
    r, c = map(int, coord)
    min_r, max_r, min_c, max_c = bounds
    out = (r, c)
    for _ in range(k % 4):
        rr, cc = out
        out = (min_r + (cc - min_c), max_c - (rr - min_r))
    return out


def canonical_group(coord, bounds):
    members = [rot(coord, bounds, k) for k in range(4)]
    return tuple(sorted(members))


def circular_stats(headings, n_bins=12):
    if len(headings) == 0:
        return float("nan"), float("nan")
    theta = 2 * np.pi * np.asarray(headings, dtype=float) / n_bins
    z = np.mean(np.exp(1j * theta))
    mean = float(np.angle(z) % (2 * np.pi))
    var = float(1.0 - np.abs(z))
    return mean, var


def load_trajectories(n=500, batch=50):
    paths = sorted(TRAJ_DIR.glob("traj_*.npz"))[:n]
    if len(paths) < n:
        raise FileNotFoundError(f"Requested {n} trajectory files, found {len(paths)}")
    coverage_positions = []
    headings = []
    print("C1/C2 loading trajectory batches")
    for start in range(0, n, batch):
        batch_paths = paths[start : start + batch]
        print(f"  batch {start:03d}-{start + len(batch_paths) - 1:03d}")
        for path in batch_paths:
            with np.load(path) as z:
                pos = z["pos"]
                head = z["heading"]
            m = min(len(pos), len(head))
            coverage_positions.append(pos[:m])
            headings.append(head[:m])
    return paths, np.vstack(coverage_positions), np.concatenate(headings)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    paths, pos_all, heading_all = load_trajectories()
    bounds = coord_bounds(pos_all)
    min_r, max_r, min_c, max_c = bounds
    n_rows = max_r - min_r + 1
    n_cols = max_c - min_c + 1
    if (n_rows, n_cols) != (18, 18):
        print(f"UNEXPECTED RESULT - trajectory coordinate bounds produce grid {n_rows}x{n_cols}: {bounds}")

    coverage = np.zeros((n_rows, n_cols), dtype=int)
    heading_by_coord = {(r, c): [] for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)}
    for (r, c), h in zip(pos_all, heading_all):
        rr, cc = int(r) - min_r, int(c) - min_c
        coverage[rr, cc] += 1
        heading_by_coord[(int(r), int(c))].append(int(h))

    np.save(METRICS_DIR / "s4_coverage_map.npy", coverage)
    coverage_std = float(np.std(coverage) / np.mean(coverage))

    groups = sorted({canonical_group(coord, bounds) for coord in heading_by_coord})
    group_cvs = []
    group_vars = []
    group_summaries = []
    for group in groups:
        counts = np.asarray([coverage[r - min_r, c - min_c] for r, c in group], dtype=float)
        cv = float(np.std(counts) / np.mean(counts)) if np.mean(counts) else float("nan")
        group_cvs.append(cv)
        vars_ = []
        means_ = []
        for coord in group:
            mean, var = circular_stats(heading_by_coord[coord])
            means_.append(mean)
            vars_.append(var)
        group_vars.extend(vars_)
        group_summaries.append(
            {
                "group": [list(x) for x in group],
                "counts": counts.astype(int).tolist(),
                "coverage_cv": cv,
                "circular_means_rad": means_,
                "circular_variances": vars_,
            }
        )

    high_cv = [g for g in group_summaries if g["coverage_cv"] > 0.3]
    summary = {
        "n_trajectories": len(paths),
        "n_visits": int(np.sum(coverage)),
        "coordinate_bounds": {"min_r": min_r, "max_r": max_r, "min_c": min_c, "max_c": max_c},
        "coverage_std_over_mean": coverage_std,
        "mean_c4_group_coverage_cv": float(np.nanmean(group_cvs)),
        "median_c4_group_coverage_cv": float(np.nanmedian(group_cvs)),
        "n_groups_cv_gt_0_3": int(len(high_cv)),
        "mean_circular_variance": float(np.nanmean(group_vars)),
        "median_circular_variance": float(np.nanmedian(group_vars)),
        "example_groups": group_summaries[:5],
    }
    with open(METRICS_DIR / "s4_trajectory_c1_c2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nC1 behavioral coverage")
    print(f"Total trajectory files analyzed: {len(paths)}")
    print(f"Total visits: {summary['n_visits']}")
    print(f"Coverage std/mean: {coverage_std:.6f}")
    print(f"Mean C4 group coverage CV: {summary['mean_c4_group_coverage_cv']:.6f}")
    print(f"Groups with CV > 0.3: {len(high_cv)} / {len(groups)}")
    if coverage_std > 0.5:
        print("SOFT FLAG - coverage_std > 0.5; undersampling may bias H estimates.")
    if high_cv:
        print("SOFT FLAG - at least one C4 group has coverage CV > 0.3.")

    print("\nC2 heading distribution")
    print(f"Mean circular variance: {summary['mean_circular_variance']:.6f}")
    print(f"Median circular variance: {summary['median_circular_variance']:.6f}")

    make_figure(coverage, heading_by_coord, group_summaries, bounds)

    print("\nOutput files")
    for path in [METRICS_DIR / "s4_coverage_map.npy", METRICS_DIR / "s4_trajectory_c1_c2_summary.json"]:
        print(f"{path.relative_to(BASE)} {path.stat().st_size} bytes")


def make_figure(coverage, heading_by_coord, group_summaries, bounds):
    min_r, max_r, min_c, max_c = bounds
    # Pick a well-sampled non-boundary group for readable polar histograms.
    candidate = max(group_summaries, key=lambda g: min(g["counts"]))
    group = [tuple(x) for x in candidate["group"]]

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.45, hspace=0.55)
    ax_map = fig.add_subplot(gs[:, 0])
    im = ax_map.imshow(coverage, cmap="viridis", interpolation="nearest")
    ax_map.axhline(8.5, color="white", linewidth=0.8)
    ax_map.axvline(8.5, color="white", linewidth=0.8)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("Position visitation frequency\n(500 trajectories)")
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    theta_edges = np.linspace(0, 2 * np.pi, 13)
    for idx, coord in enumerate(group):
        ax = fig.add_subplot(gs[idx // 2, 1 + idx % 2], projection="polar")
        vals = np.asarray(heading_by_coord[coord], dtype=int)
        counts = np.bincount(vals, minlength=12)
        ax.bar(theta_edges[:-1], counts, width=2 * np.pi / 12, align="edge", color="#2166AC", alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{coord}", labelpad=4, fontsize=8)

    # Use the remaining cell for a compact group count balance panel.
    ax_bar = fig.add_subplot(gs[:, 3])
    ax_bar.bar(np.arange(4), candidate["counts"], color="#F4A582", edgecolor="black", linewidth=0.4)
    ax_bar.set_xticks(np.arange(4), [f"P{i+1}" for i in range(4)])
    ax_bar.set_ylabel("Visits")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    paths = []
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"fig_coverage_and_heading.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    for path in paths:
        print(f"{path.relative_to(BASE)} {path.stat().st_size} bytes. Coverage/heading figure generated from 500 trajectories.")


if __name__ == "__main__":
    main()
