#!/usr/bin/env python3
"""Rebuild all manuscript figures and save to figures/final as PDF+PNG."""

import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

from scipy import stats
from scipy.stats import spearmanr, gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.manifold import Isomap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.arena import SymmetryArena

# -------------------------
# Global style
# -------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman", "Times"],
    "font.size": 12.5,
    "axes.labelsize": 13.5,
    "axes.titlesize": 15,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLORS = {"S1": "#2196F3", "S2": "#FF9800", "S4": "#F44336"}
CONDITIONS = ["S1", "S2", "S4"]
COND_DIR = {"S1": "s1", "S2": "s2", "S4": "s4"}

RNG = np.random.default_rng(0)
RESULTS_ROOT = ROOT / "results2" / "symmetry_sweep"
ANALYSIS_DIR = ROOT / "results2" / "analysis_output"
FIG_DIR = ROOT / "figures" / "final"


# -------------------------
# Helpers
# -------------------------


def savefig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=600, facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=13,
    )


def load_eval(seed_dir: Path):
    for name in ["evaluation.pkl", "eval.pkl"]:
        path = seed_dir / name
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def load_training_log(seed_dir: Path):
    path = seed_dir / "training_log.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_condition_evals(condition: str):
    cond_dir = RESULTS_ROOT / COND_DIR[condition]
    evals = []
    for seed_dir in sorted(cond_dir.glob("seed_*")):
        ev = load_eval(seed_dir)
        if ev is None:
            continue
        log = load_training_log(seed_dir)
        evals.append({"seed_dir": seed_dir, "eval": ev, "log": log})
    return evals


def load_master_metrics():
    path = ANALYSIS_DIR / "master_metrics.csv"
    if not path.exists():
        return None
    import pandas as pd

    return pd.read_csv(path)


def load_spectral_gap_data():
    path = ANALYSIS_DIR / "spectral_gap_data.csv"
    if not path.exists():
        return None
    import pandas as pd

    return pd.read_csv(path)


def load_observation_summary(condition: str):
    path = RESULTS_ROOT / COND_DIR[condition] / "observation_summary.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def positions_to_grid(values, positions, grid_size=18):
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    for val, (c, r) in zip(values, positions):
        if 1 <= c <= grid_size and 1 <= r <= grid_size:
            grid[int(r) - 1, int(c) - 1] = val
    return grid


def grid_mask_from_positions(positions, grid_size=18):
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for c, r in positions:
        if 1 <= c <= grid_size and 1 <= r <= grid_size:
            mask[int(r) - 1, int(c) - 1] = True
    return mask


def compute_unit_ra(map_grid, mask):
    rotated = np.rot90(map_grid, k=1)
    valid = mask & np.isfinite(map_grid) & np.isfinite(rotated)
    if valid.sum() < 5:
        return np.nan
    a = map_grid[valid].ravel()
    b = rotated[valid].ravel()
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    r, _ = spearmanr(a, b)
    return float(r)


def quadrant_order_indices(positions, grid_size=18):
    center = grid_size / 2.0
    def quadrant(c, r):
        if r <= center and c <= center:
            return 0
        if r <= center and c > center:
            return 1
        if r > center and c <= center:
            return 2
        return 3
    order = sorted(
        range(len(positions)),
        key=lambda i: (quadrant(positions[i][0], positions[i][1]), positions[i][1], positions[i][0]),
    )
    return np.array(order, dtype=int)


def align_curves(logs, key):
    curves = []
    for log in logs:
        if log is None or key not in log:
            continue
        steps = np.array(log.get("steps", []), dtype=float)
        vals = np.array(log.get(key, []), dtype=float)
        if len(steps) == 0 or len(vals) == 0:
            continue
        n = min(len(steps), len(vals))
        curves.append((steps[:n], vals[:n]))
    if not curves:
        return None
    max_step = max(c[0][-1] for c in curves)
    common = np.linspace(0, max_step, 200)
    interp = np.array([np.interp(common, s, v) for s, v in curves])
    return common, interp


def bootstrap_kde(values, x_grid, n_boot=200):
    if len(values) < 5:
        return None, None
    boot = []
    for _ in range(n_boot):
        sample = RNG.choice(values, size=len(values), replace=True)
        kde = gaussian_kde(sample)
        boot.append(kde(x_grid))
    boot = np.array(boot)
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


# -------------------------
# Figure 1 - Arena layouts
# -------------------------


def render_arena_grid(ax, condition, grid_size=18):
    arena = SymmetryArena(
        shape="square",
        size=grid_size,
        U=0,
        F=7,
        seed=0,
        use_landmarks=True,
        symmetry_condition=condition.lower(),
    )
    tiles = arena._get_landmark_tiles()

    ax.set_xlim(0.5, grid_size + 0.5)
    ax.set_ylim(0.5, grid_size + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticks(np.arange(0.5, grid_size + 1.5), minor=True)
    ax.set_yticks(np.arange(0.5, grid_size + 1.5), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linewidth=0.5)

    xs, ys, colors = [], [], []
    for (r, c), rgb in tiles.items():
        xs.append(c)
        ys.append(r)
        colors.append(rgb)
    if xs:
        ax.scatter(xs, ys, s=36, marker="s", c=colors, edgecolor="none")

    for spine in ax.spines.values():
        spine.set_visible(False)


# -------------------------
# Figure builders
# -------------------------


def build_figure_1():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    titles = ["S4 (C$_4$ symmetry)", "S2 (C$_2$ symmetry)", "S1 (asymmetric)"]
    for ax, cond, title in zip(axes[:3], ["S4", "S2", "S1"], titles):
        render_arena_grid(ax, cond)
        ax.set_title(title)

    ax = axes[3]
    render_arena_grid(ax, "S4")
    grid_size = 18
    center = (grid_size + 1) / 2.0
    ax.plot([0.5, grid_size + 0.5], [center, center], ls="--", lw=1.2, color="#555555")
    ax.plot([center, center], [0.5, grid_size + 0.5], ls="--", lw=1.2, color="#555555")
    ax.plot([0.5, grid_size + 0.5], [0.5, grid_size + 0.5], ls="--", lw=1.2, color="#555555")
    ax.plot([0.5, grid_size + 0.5], [grid_size + 0.5, 0.5], ls="--", lw=1.2, color="#555555")
    ax.set_title("Symmetry axes")

    for label, ax in zip(["A", "B", "C", "D"], axes):
        add_panel_label(ax, label)

    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="Blue staircase", markerfacecolor="#000073", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Red cross", markerfacecolor="#730000", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Yellow castle", markerfacecolor="#737300", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Green chevron", markerfacecolor="#005900", markersize=8),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    savefig(fig, "fig1_arena_layouts")


def build_figure_2():
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.5, 4.4),
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.0, 1.1]},
        layout="constrained",
    )
    ax_a, ax_b1, ax_b2, ax_c = axes

    # Panel A: sRSA pipeline
    mat = np.array([
        [0.0, 0.2, 0.3, 0.4],
        [0.2, 0.0, 0.5, 0.6],
        [0.3, 0.5, 0.0, 0.7],
        [0.4, 0.6, 0.7, 0.0],
    ])
    ax_a.imshow(mat, cmap="Blues", vmin=0, vmax=0.8)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_title("sRSA pipeline", pad=8)
    ax_a.annotate(
        "Upper triangle",
        xy=(1.1, 0.4),
        xytext=(2.5, -0.2),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=10,
    )
    ax_a.annotate(
        "Spearman r\nacross seeds",
        xy=(2.6, 2.6),
        xytext=(2.6, 3.4),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=10,
        ha="center",
    )
    add_panel_label(ax_a, "A")

    # Panel B: seed comparison
    base = np.array([
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
        [4, 3, 2, 1, 0],
    ], dtype=float)
    perm = [0, 2, 4, 1, 3]
    mat1 = base
    mat2 = base[np.ix_(perm, perm)]
    ax_b1.imshow(mat1, cmap="viridis", vmin=0, vmax=4)
    ax_b1.set_xticks([])
    ax_b1.set_yticks([])
    ax_b1.set_title("Seed 1")
    ax_b2.imshow(mat2, cmap="viridis", vmin=0, vmax=4)
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    ax_b2.set_title("Seed 2")
    ax_b2.text(0.5, -0.12, "sRSA = 0.977", transform=ax_b2.transAxes, ha="center", fontsize=10)
    ax_b2.text(
        0.5,
        -0.22,
        "Row permutation - all pairwise distances preserved",
        transform=ax_b2.transAxes,
        ha="center",
        fontsize=9.5,
        color="#B00020",
    )
    add_panel_label(ax_b1, "B")

    # Panel C: RA vs PAA schematics
    ax_c.axis("off")
    ax_ra = ax_c.inset_axes([0.02, 0.45, 0.46, 0.5])
    ax_paa = ax_c.inset_axes([0.52, 0.05, 0.46, 0.5])
    for sub in (ax_ra, ax_paa):
        sub.set_xlim(0, 1)
        sub.set_ylim(0, 1)
        sub.set_axis_off()

    pts = np.array([[0.2, 0.7], [0.35, 0.8], [0.5, 0.7], [0.35, 0.6]])
    rot = np.array([[0.7, 0.8], [0.8, 0.65], [0.7, 0.5], [0.6, 0.65]])
    ax_ra.scatter(pts[:, 0], pts[:, 1], s=26, color="#222222")
    ax_ra.scatter(rot[:, 0], rot[:, 1], s=26, color="#222222")
    ax_ra.annotate("RA", xy=(0.52, 0.68), xytext=(0.05, 0.92),
                   arrowprops=dict(arrowstyle="->", lw=1.1), fontsize=10)
    ax_ra.set_title("Global rotation", fontsize=11)

    cluster_a = np.array([[0.15, 0.25], [0.25, 0.35], [0.3, 0.2]])
    cluster_b = np.array([[0.65, 0.25], [0.75, 0.35], [0.8, 0.2]])
    ax_paa.scatter(cluster_a[:, 0], cluster_a[:, 1], s=26, color="#222222")
    ax_paa.scatter(cluster_b[:, 0], cluster_b[:, 1], s=26, color="#222222")
    ax_paa.annotate("PAA", xy=(0.72, 0.26), xytext=(0.55, 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.1), fontsize=10)
    ax_paa.set_title("Local folding", fontsize=11)
    add_panel_label(ax_c, "C")

    savefig(fig, "fig2_srsa_blindness")


def build_figure_3():
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.2], hspace=0.25)

    # Outcome grid
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Global orientation stable (PAA < 0.05)")
    ax.set_ylabel("Local precision high (RA < 0.15)")

    cells = [
        (0, 0, "Full collapse", "#DDDDDD", "--", None),
        (1, 0, "Partial folding <- observed", "#F8B4B4", "-", None),
        (1, 1, "Full robustness", "#B7E4C7", "--", None),
        (0, 1, "Not possible (degenerate but precise)", "#EEEEEE", "--", "//"),
    ]
    for x, y, label, color, ls, hatch in cells:
        rect = plt.Rectangle(
            (x, y),
            1,
            1,
            facecolor=color,
            edgecolor="#555555",
            lw=1.2,
            linestyle=ls,
            hatch=hatch,
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=9, fontweight="bold" if "observed" in label else "normal")
    ax.set_aspect("equal")
    add_panel_label(ax, "A")

    # Embedding schematics
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    for i, title in enumerate(["Full collapse", "Partial folding", "Full robustness"]):
        sub = ax2.inset_axes([0.02 + i * 0.32, 0.05, 0.28, 0.9])
        sub.set_xticks([])
        sub.set_yticks([])
        sub.set_xlim(-1, 1)
        sub.set_ylim(-1, 1)
        if title == "Full collapse":
            pts = RNG.normal(0, 0.1, size=(20, 2))
        elif title == "Partial folding":
            pts = np.vstack([RNG.normal([-0.5, 0], 0.1, size=(10, 2)), RNG.normal([0.5, 0], 0.1, size=(10, 2))])
        else:
            ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            pts = np.column_stack([np.cos(ang), np.sin(ang)])
        sub.scatter(pts[:, 0], pts[:, 1], s=8, color="#333333")
        sub.set_title(title, fontsize=8)
    savefig(fig, "fig3_outcome_space")


def build_figure_4(metrics_df):
    metrics = {
        "RA (Rotational Autocorrelation)": "ra",
        "SCI (Symmetry Collapse Index)": "sci",
        "C2 Contrast": "c2_contrast",
        "Decode Error": "decode_error",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (label, key) in zip(axes.flat, metrics.items()):
        for i, cond in enumerate(CONDITIONS):
            vals = metrics_df[(metrics_df["experiment"] == "symmetry_sweep") & (metrics_df["condition"] == cond)][key].dropna().values
            if len(vals) == 0:
                continue
            ax.bar(i, np.mean(vals), color=COLORS[cond], alpha=0.7, width=0.6)
            jitter = RNG.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", s=20, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(CONDITIONS)
        ax.set_title(label)
    add_panel_label(axes.flat[0], "A")
    add_panel_label(axes.flat[1], "B")
    add_panel_label(axes.flat[2], "C")
    add_panel_label(axes.flat[3], "D")
    savefig(fig, "fig4_metric_comparison")


def build_figure_5(condition_evals):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    vmin, vmax = 0.0, 1.2

    mean_rdms = {}
    mean_positions = {}

    for ax, cond in zip(axes[:3], CONDITIONS):
        evals = condition_evals[cond]
        rdms = [e["eval"]["rsa_matrix"] for e in evals if "rsa_matrix" in e["eval"]]
        positions = evals[0]["eval"]["position_array"] if evals else None
        if not rdms or positions is None:
            ax.axis("off")
            continue
        order = quadrant_order_indices(positions)
        rdms = [r[np.ix_(order, order)] for r in rdms]
        mean_rdm = np.mean(rdms, axis=0)
        mean_rdms[cond] = mean_rdm
        mean_positions[cond] = positions

        idx = np.triu_indices(mean_rdm.shape[0], k=1)
        rhos = []
        for i in range(len(rdms)):
            for j in range(i + 1, len(rdms)):
                rhos.append(spearmanr(rdms[i][idx], rdms[j][idx]).statistic)
        rho = float(np.mean(rhos)) if rhos else np.nan

        im = ax.imshow(mean_rdm, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(cond)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.95, f"r = {rho:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    # Difference panel S4 - S1
    ax = axes[3]
    if "S4" in mean_rdms and "S1" in mean_rdms:
        diff = mean_rdms["S4"] - mean_rdms["S1"]
        norm = TwoSlopeNorm(vcenter=0.0)
        im = ax.imshow(diff, cmap="coolwarm", norm=norm)
        ax.set_title("S4 - S1")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")

    for label, ax in zip(["A", "B", "C", "D"], axes):
        add_panel_label(ax, label)

    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cax, label="RDM distance")
    savefig(fig, "fig5_cross_seed_rdm")


def build_figure_6a(condition_evals):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    metric_keys = [
        ("srsa_euclid", "sRSA"),
        ("manifold_id", "Manifold ID"),
        ("mean_field_coherence", "Field Coherence"),
        ("loss", "Loss"),
    ]

    for ax, (key, label) in zip(axes.flat, metric_keys):
        for cond in CONDITIONS:
            logs = [e["log"] for e in condition_evals[cond] if e.get("log")]
            aligned = align_curves(logs, key)
            if aligned is None:
                continue
            steps, interp = aligned
            mean = np.nanmean(interp, axis=0)
            sem = np.nanstd(interp, axis=0) / np.sqrt(interp.shape[0])
            x = steps / 1000.0

            if key == "manifold_id":
                ax.plot(x, mean, color=COLORS[cond], alpha=0.2, lw=0.8)
                step_delta = steps[1] - steps[0] if len(steps) > 1 else 1000
                window_steps = max(1, int(round(3000 / step_delta)))
                kernel = np.ones(window_steps) / window_steps
                smooth = np.convolve(mean, kernel, mode="valid")
                ax.plot(x[: len(smooth)], smooth, color=COLORS[cond], lw=2, label=cond)
            else:
                ax.plot(x, mean, color=COLORS[cond], lw=2, label=cond)
                ax.fill_between(x, mean - sem, mean + sem, color=COLORS[cond], alpha=0.15, lw=0)

        ax.set_title(label)
        ax.set_xlabel("Steps (1e3)")
        ax.set_ylabel(label)

    # annotate field coherence spike
    ax_fc = axes.flat[2]
    ax_fc.annotate(
        "Transient reorganization (~5k steps)",
        xy=(5, ax_fc.get_ylim()[1] * 0.7),
        xytext=(8, ax_fc.get_ylim()[1] * 0.9),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
    )

    for label, ax in zip(["A", "B", "C", "D"], axes.flat):
        add_panel_label(ax, label)
    axes.flat[0].legend(frameon=False)

    savefig(fig, "fig6a_learning_trajectories")


def build_figure_6b(condition_evals):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    for cond in CONDITIONS:
        logs = [e["log"] for e in condition_evals[cond] if e.get("log")]
        curves = []
        steps_ref = None
        for log in logs:
            if "srsa_euclid" not in log or "srsa_city" not in log:
                continue
            steps = np.array(log.get("steps", []), dtype=float)
            eu = np.array(log.get("srsa_euclid", []), dtype=float)
            city = np.array(log.get("srsa_city", []), dtype=float)
            n = min(len(steps), len(eu), len(city))
            if n == 0:
                continue
            if steps_ref is None:
                steps_ref = steps[:n]
            curves.append(eu[:n] - city[:n])

        if not curves or steps_ref is None:
            continue

        curves = np.vstack(curves)
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0) / np.sqrt(curves.shape[0])
        x = steps_ref / 1000.0
        ax.plot(x, mean, color=COLORS[cond], lw=2, label=cond)
        ax.fill_between(x, mean - sem, mean + sem, color=COLORS[cond], alpha=0.15, lw=0)

    ax.axhline(0.0, color="#666666", ls="--", lw=1.0)
    ax.set_xlabel("Steps (1e3)")
    ax.set_ylabel("DTG (sRSA_euclid - sRSA_city)")
    ax.set_title("Distance-Topology Gap over training")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend_.remove() if ax.legend_ else None
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    savefig(fig, "fig6b_dtg")


def build_figure_7(condition_evals):
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    shared_im = None

    for row, cond in enumerate(CONDITIONS):
        evals = condition_evals[cond]
        if not evals:
            continue
        ev = evals[0]["eval"]
        H = ev["position_hidden"]
        pos = ev["position_array"]
        grid_size = int(np.max(pos))
        mask = grid_mask_from_positions(pos, grid_size)

        # pick top units by variance
        variances = []
        for u in range(H.shape[1]):
            g = positions_to_grid(H[:, u], pos, grid_size)
            variances.append(np.nanvar(g))
        top_units = np.argsort(variances)[-6:][::-1]

        for col, unit in enumerate(top_units):
            ax = axes[row, col]
            g = positions_to_grid(H[:, unit], pos, grid_size)
            g = gaussian_filter(np.nan_to_num(g, nan=0.0), sigma=0.6)
            im = ax.imshow(g, cmap="magma", origin="lower", interpolation="nearest")
            shared_im = im
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(cond, rotation=0, ha="right", va="center", labelpad=8)

            # RA annotation
            ra = compute_unit_ra(g, mask)
            if np.isfinite(ra):
                ax.text(0.98, 0.98, f"RA={ra:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="white")

            # symmetry axes on S4
            if cond == "S4":
                center = (grid_size - 1) / 2.0
                ax.axhline(center, color="white", lw=0.8, ls="--", alpha=0.8)
                ax.axvline(center, color="white", lw=0.8, ls="--", alpha=0.8)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(shared_im, cax=cax, label="Normalized firing rate")
    add_panel_label(axes[0, 0], "A")
    savefig(fig, "fig7_place_cell_rate_maps")


def build_figure_8(condition_evals):
    fig, ax = plt.subplots(figsize=(6.5, 3))

    for cond in CONDITIONS:
        widths = []
        for item in condition_evals[cond]:
            ev = item["eval"]
            H = ev["position_hidden"]
            pos = ev["position_array"]
            grid_size = int(np.max(pos))
            for u in range(H.shape[1]):
                g = positions_to_grid(H[:, u], pos, grid_size)
                g = gaussian_filter(np.nan_to_num(g, nan=0.0), sigma=0.6)
                peak = np.max(g)
                if peak <= 0:
                    continue
                area = np.sum(g > 0.5 * peak)
                widths.append(np.sqrt(area))
        widths = np.array(widths)
        if len(widths) < 5:
            continue

        kde = gaussian_kde(widths)
        x = np.linspace(widths.min(), widths.max(), 200)
        y = kde(x)
        ci_low, ci_high = bootstrap_kde(widths, x, n_boot=200)

        ax.plot(x, y, color=COLORS[cond], lw=2, label=cond)
        if ci_low is not None:
            ax.fill_between(x, ci_low, ci_high, color=COLORS[cond], alpha=0.15, lw=0)

        mean_val = widths.mean()
        ax.axvline(mean_val, color=COLORS[cond], ls="--", lw=1.2)
        ax.text(mean_val, ax.get_ylim()[1] * 0.9, f"{mean_val:.2f}", color=COLORS[cond], ha="center", fontsize=9)

        # Rug plot
        rug_y = np.zeros_like(widths) + (ax.get_ylim()[0] + 0.02)
        ax.plot(widths, rug_y, "|", color=COLORS[cond], alpha=0.2)

    ax.set_xlabel("Field width (sqrt area)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")
    savefig(fig, "fig8_field_width_distribution")


def build_figure_9(condition_evals):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 1, 1, 0.5]})

    error_maps = {}
    for ax, cond in zip(axes[:3], CONDITIONS):
        ev = condition_evals[cond][0]["eval"] if condition_evals[cond] else None
        if ev is None:
            ax.axis("off")
            continue
        H = ev["position_hidden"]
        pos = ev["position_array"]
        knn = KNeighborsRegressor(n_neighbors=5).fit(H, pos)
        preds = knn.predict(H)
        errs = np.linalg.norm(preds - pos, axis=1)
        grid_size = int(np.max(pos))
        g = positions_to_grid(errs, pos, grid_size)
        error_maps[cond] = g
        im = ax.imshow(g, cmap="hot_r", vmin=0.25, vmax=1.75, origin="lower")
        ax.set_title(cond)
        ax.set_xticks([])
        ax.set_yticks([])
        if cond == "S4":
            center = (grid_size - 1) / 2.0
            ax.axhline(center, color="cyan", lw=1.5, ls="--", alpha=0.8)
            ax.axvline(center, color="cyan", lw=1.5, ls="--", alpha=0.8)
            ax.plot([0, grid_size - 1], [0, grid_size - 1], color="cyan", lw=1.5, ls="--", alpha=0.8)
            ax.plot([0, grid_size - 1], [grid_size - 1, 0], color="cyan", lw=1.5, ls="--", alpha=0.8)

    # Histogram panel
    axh = axes[3]
    if "S1" in error_maps and "S4" in error_maps:
        s1 = error_maps["S1"][np.isfinite(error_maps["S1"])]
        s4 = error_maps["S4"][np.isfinite(error_maps["S4"])]
        axh.hist(s1, bins=30, color=COLORS["S1"], alpha=0.6, density=True, orientation="horizontal", label="S1")
        axh.hist(s4, bins=30, color=COLORS["S4"], alpha=0.6, density=True, orientation="horizontal", label="S4")
        axh.set_xlabel("Density")
        axh.legend(frameon=False)
    else:
        axh.axis("off")

    for label, ax in zip(["A", "B", "C", "D"], axes):
        add_panel_label(ax, label)

    cax = fig.add_axes([0.1, 0.08, 0.65, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Decoding error")
    savefig(fig, "fig9_decoding_error_maps")


def build_figure_10(metrics_df, condition_evals):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0, 1.0]})

    # C2 contrast bar
    ax = axes[0]
    for i, cond in enumerate(CONDITIONS):
        vals = metrics_df[(metrics_df["experiment"] == "symmetry_sweep") & (metrics_df["condition"] == cond)]["c2_contrast"].dropna().values
        if len(vals) == 0:
            continue
        ax.bar(i, np.mean(vals), color=COLORS[cond], alpha=0.7, width=0.6)
        ax.scatter(np.full(len(vals), i) + RNG.uniform(-0.1, 0.1, size=len(vals)), vals, color="black", s=20)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CONDITIONS)
    ax.set_title("C2 Contrast")
    add_panel_label(ax, "A")

    # RDM panels
    for ax, cond, label in zip(axes[1:], CONDITIONS, ["B", "C", "D"]):
        evals = condition_evals[cond]
        if not evals:
            ax.axis("off")
            continue
        ev = evals[0]["eval"]
        rdm = ev["rsa_matrix"]
        positions = ev["position_array"]
        order = quadrant_order_indices(positions)
        rdm = rdm[np.ix_(order, order)]
        im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=1.2)
        ax.set_title(cond)
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_label(ax, label)
        if cond == "S4":
            n = rdm.shape[0]
            for f in [0.25, 0.5, 0.75]:
                ax.axhline(f * n, color="w", lw=0.7, ls="--")
                ax.axvline(f * n, color="w", lw=0.7, ls="--")

    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cax, label="RDM distance")
    savefig(fig, "fig10_c2_contrast_rdms")


def build_figure_11(condition_evals):
    fig = plt.figure(figsize=(11, 3.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.2, 1.0, 1.0, 1.0], wspace=0.4)

    # Left panel: manifold ID over training
    ax = fig.add_subplot(gs[0, 0])
    for cond in CONDITIONS:
        logs = [e["log"] for e in condition_evals[cond] if e.get("log")]
        aligned = align_curves(logs, "manifold_id")
        if aligned is None:
            continue
        steps, interp = aligned
        mean = np.nanmean(interp, axis=0)
        window = 30
        kernel = np.ones(window) / window
        smooth = np.convolve(mean, kernel, mode="valid")
        x = steps / 1000.0
        ax.plot(x, mean, color=COLORS[cond], alpha=0.2, lw=0.8)
        ax.plot(x[: len(smooth)], smooth, color=COLORS[cond], lw=2, label=cond)
    ax.set_xlabel("Steps (1e3)")
    ax.set_ylabel("Manifold ID")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")

    # Right panels: embeddings
    for i, cond in enumerate(CONDITIONS):
        ax = fig.add_subplot(gs[0, i + 1])
        evals = condition_evals[cond]
        if not evals:
            ax.axis("off")
            continue
        ev = evals[0]["eval"]
        H = ev["position_hidden"]
        pos = ev["position_array"]
        try:
            emb = Isomap(n_components=2, n_neighbors=10).fit_transform(H)
        except Exception:
            emb = H[:, :2]
        angles = np.arctan2(pos[:, 1] - pos[:, 1].mean(), pos[:, 0] - pos[:, 0].mean())
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=angles, cmap="hsv", s=10, lw=0)
        ax.set_title(cond)
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_label(ax, chr(ord("B") + i))

    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(sc, cax=cax, label="Heading angle (rad)")
    savefig(fig, "fig11_manifold_geometry")


def build_figure_12(spectral_df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Left: spectral gap strip plot
    ax = axes[0]
    for i, cond in enumerate(CONDITIONS):
        vals = spectral_df[spectral_df["condition"] == cond]["spectral_gap"].values
        if len(vals) == 0:
            continue
        ax.bar(i, np.mean(vals), color=COLORS[cond], alpha=0.7, width=0.6)
        ax.scatter(np.full(len(vals), i) + RNG.uniform(-0.1, 0.1, size=len(vals)), vals, color="black", s=25, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Spectral gap of W_h")
    add_panel_label(ax, "A")

    # Right: sRSA vs spectral gap
    ax = axes[1]
    gaps = {}
    srsas = {}
    for cond in CONDITIONS:
        sub = spectral_df[spectral_df["condition"] == cond]
        gaps[cond] = sub["spectral_gap"].values
        srsas[cond] = sub["srsa"].values
        ax.scatter(gaps[cond], srsas[cond], color=COLORS[cond], s=40, label=cond, zorder=3)
    all_gaps = np.concatenate(list(gaps.values()))
    all_srsas = np.concatenate(list(srsas.values()))
    slope, intercept, r, p, _ = stats.linregress(all_gaps, all_srsas)
    x = np.linspace(all_gaps.min(), all_gaps.max(), 100)
    ax.plot(x, slope * x + intercept, "k--", lw=1.5)
    ax.annotate(f"r = {r:.2f}, p = {p:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
    ax.set_xlabel("Spectral gap")
    ax.set_ylabel("sRSA")
    ax.legend(frameon=False)
    add_panel_label(ax, "B")

    savefig(fig, "fig12_spectral_gap")


def build_figure_13(metrics_df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    abl_order = ["HD_FULL", "HD_DEGRADED", "HD_ABLATED"]
    labels = ["Full", "Degraded", "Ablated"]

    # Bar chart
    ax = axes[0]
    for i, cond in enumerate(abl_order):
        vals = metrics_df[(metrics_df["experiment"] == "hd_ablation") & (metrics_df["condition"] == cond)]["srsa"].dropna().values
        if len(vals) == 0:
            continue
        ax.bar(i, np.mean(vals), color=COLORS["S1"], alpha=0.7, width=0.6)
        ax.scatter(np.full(len(vals), i) + RNG.uniform(-0.1, 0.1, size=len(vals)), vals, color="black", s=25, zorder=3)
    ax.axhline(0.40, color="gray", ls="--", lw=1.5, label="Formation gate (0.40)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("sRSA")
    ax.text(0.02, 0.02, "n=2 seeds; error bars omitted", transform=ax.transAxes, fontsize=8)
    ax.legend(frameon=False)
    add_panel_label(ax, "A")

    # Learning dynamics
    ax = axes[1]
    for cond, label in zip(abl_order, labels):
        cond_dir = ROOT / "results2" / "ablation" / cond.lower().replace("hd_", "")
        if not cond_dir.exists():
            continue
        curves = []
        steps_ref = None
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            log = load_training_log(seed_dir)
            if log is None:
                continue
            steps = np.array(log.get("steps", []), dtype=float)
            srsa = np.array(log.get("srsa_euclid", []), dtype=float)
            n = min(len(steps), len(srsa))
            if n == 0:
                continue
            if steps_ref is None:
                steps_ref = steps[:n]
            curves.append(srsa[:n])
        if not curves or steps_ref is None:
            continue
        curves = np.vstack(curves)
        mean = curves.mean(axis=0)
        x = steps_ref / 1000.0
        ax.plot(x, mean, lw=2, label=label)
    ax.axhline(0.40, color="gray", ls="--", lw=1.5, label="Formation gate (0.40)")
    ax.fill_between([0, ax.get_xlim()[1]], 0, 0.40, color="#EEEEEE", alpha=0.5)
    ax.set_xlabel("Steps (1e3)")
    ax.set_ylabel("sRSA")
    ax.legend(frameon=False)
    add_panel_label(ax, "B")

    savefig(fig, "fig13_hd_ablation")


# -------------------------
# New figures A-D
# -------------------------


def build_new_figure_a(condition_evals):
    fig, ax = plt.subplots(figsize=(6, 4))
    distributions = {}
    for cond in CONDITIONS:
        ras = []
        for item in condition_evals[cond]:
            ev = item["eval"]
            H = ev["position_hidden"]
            pos = ev["position_array"]
            grid_size = int(np.max(pos))
            mask = grid_mask_from_positions(pos, grid_size)
            for u in range(H.shape[1]):
                g = positions_to_grid(H[:, u], pos, grid_size)
                ra = compute_unit_ra(g, mask)
                if np.isfinite(ra):
                    ras.append(ra)
        distributions[cond] = np.array(ras)

    data = [distributions[c] for c in CONDITIONS]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True)
    for body, cond in zip(parts["bodies"], CONDITIONS):
        body.set_facecolor(COLORS[cond])
        body.set_alpha(0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Per-unit Rotational Autocorrelation")
    ax.set_title("RA distribution across units - redistribution hypothesis")
    add_panel_label(ax, "A")
    savefig(fig, "fig_new_a_ra_distribution")


def build_new_figure_b(condition_evals):
    has_ra_curve = False
    for cond in CONDITIONS:
        for item in condition_evals[cond]:
            log = item.get("log")
            if log and "ra" in log:
                has_ra_curve = True
    if not has_ra_curve:
        print("Skipping New Figure B: no RA trajectory available in logs.")
        return


def build_new_figure_c():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")

    within_vals = []
    cross_vals = []
    marginal_vals = []

    for cond in CONDITIONS:
        summary = load_observation_summary(cond)
        if summary is None:
            within_vals.append(np.nan)
            cross_vals.append(np.nan)
            marginal_vals.append(np.nan)
            continue

        positions = np.array(summary.get("positions", []), dtype=float)
        observations = np.array(summary.get("observations", []), dtype=float)
        if positions.size == 0 or observations.size == 0:
            within_vals.append(np.nan)
            cross_vals.append(np.nan)
            marginal_vals.append(np.nan)
            continue

        obs_d = distance.pdist(observations, metric="euclidean")
        spatial_d = distance.pdist(positions, metric="euclidean")

        center_c = 0.5 * (positions[:, 0].min() + positions[:, 0].max())
        center_r = 0.5 * (positions[:, 1].min() + positions[:, 1].max())
        quads = []
        for c, r in positions:
            if r <= center_r and c <= center_c:
                quads.append(0)
            elif r <= center_r and c > center_c:
                quads.append(1)
            elif r > center_r and c <= center_c:
                quads.append(2)
            else:
                quads.append(3)
        quads = np.array(quads)

        n = len(positions)
        mask_within = []
        mask_cross = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if quads[i] == quads[j]:
                    mask_within.append(idx)
                else:
                    mask_cross.append(idx)
                idx += 1
        mask_within = np.array(mask_within, dtype=int)
        mask_cross = np.array(mask_cross, dtype=int)

        within_r = spearmanr(obs_d[mask_within], spatial_d[mask_within]).statistic if len(mask_within) else np.nan
        cross_r = spearmanr(obs_d[mask_cross], spatial_d[mask_cross]).statistic if len(mask_cross) else np.nan
        within_vals.append(within_r)
        cross_vals.append(cross_r)

        marginal = summary.get("odi", {}).get("rho")
        if marginal is None:
            marginal = spearmanr(obs_d, spatial_d).statistic
        marginal_vals.append(marginal)

    if not np.isfinite(within_vals).any() and not np.isfinite(cross_vals).any():
        ax.text(0.5, 0.5, "Observation summaries missing", ha="center", va="center")
        add_panel_label(ax, "A")
        savefig(fig, "fig_new_c_within_cross_odi")
        return

    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, within_vals, width, label="Within-quadrant", color="#4C78A8")
    ax.bar(x + width / 2, cross_vals, width, label="Cross-quadrant", color="#F58518")

    marginal_line = np.nanmean(marginal_vals)
    if np.isfinite(marginal_line):
        ax.axhline(marginal_line, color="gray", ls=":", lw=1.5, label="Marginal ODI (matched)")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Observational Discriminability Index")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))

    finite_vals = np.array(within_vals + cross_vals + marginal_vals, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size:
        pad = 0.01
        ax.set_ylim(finite_vals.min() - pad, finite_vals.max() + pad)

    add_panel_label(ax, "A")
    savefig(fig, "fig_new_c_within_cross_odi")


def build_new_figure_d(condition_evals):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained")

    # Build SR for S4
    grid_size = 18
    arena = SymmetryArena(shape="square", size=grid_size, U=0, F=7, seed=0, use_landmarks=True, symmetry_condition="s4")
    positions = np.array(arena.passable_positions)
    pos_to_idx = {tuple(p): i for i, p in enumerate(positions)}
    n = len(positions)

    T = np.zeros((n, n), dtype=np.float64)
    for i, (c, r) in enumerate(positions):
        neighbors = [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]
        valid = [pos_to_idx[p] for p in neighbors if p in pos_to_idx]
        if valid:
            prob = 1.0 / len(valid)
            for j in valid:
                T[i, j] = prob
        else:
            T[i, i] = 1.0

    gamma = 0.95
    M = np.linalg.inv(np.eye(n) - gamma * T)
    M = M / M.max()
    D_sr = 1.0 - M

    # Observed RDM
    ev = condition_evals["S4"][0]["eval"] if condition_evals["S4"] else None
    if ev is None:
        return
    rdm = ev["rsa_matrix"]
    eval_positions = ev["position_array"]
    map_idx = np.array([pos_to_idx[tuple(p)] for p in eval_positions], dtype=int)
    D_sr = D_sr[np.ix_(map_idx, map_idx)]
    order = quadrant_order_indices(eval_positions)
    D_sr = D_sr[np.ix_(order, order)]
    rdm = rdm[np.ix_(order, order)]

    n_use = rdm.shape[0]
    idx = np.triu_indices(n_use, k=1)
    rho = spearmanr(D_sr[idx], rdm[idx]).statistic

    def rescale_to_target(src, target):
        src_min, src_max = np.quantile(src, 0.02), np.quantile(src, 0.98)
        tgt_min, tgt_max = np.quantile(target, 0.02), np.quantile(target, 0.98)
        src_clip = np.clip(src, src_min, src_max)
        scaled = (src_clip - src_min) / (src_max - src_min + 1e-8)
        return tgt_min + scaled * (tgt_max - tgt_min), tgt_min, tgt_max

    D_sr_scaled, vmin, vmax = rescale_to_target(D_sr, rdm)

    im0 = axes[0].imshow(D_sr_scaled, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted SR geometry (rescaled)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].text(0.98, 0.95, f"r = {rho:.2f}", transform=axes[0].transAxes, va="top", ha="right", fontsize=11)

    im1 = axes[1].imshow(rdm, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Observed neural RDM")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    for label, ax in zip(["A", "B"], axes):
        add_panel_label(ax, label)

    fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02, label="Distance")
    savefig(fig, "fig_new_d_sr_vs_rdm")


# -------------------------
# Main
# -------------------------


def main():
    condition_evals = {cond: load_condition_evals(cond) for cond in CONDITIONS}
    metrics_df = load_master_metrics()
    spectral_df = load_spectral_gap_data()

    build_figure_1()
    build_figure_2()
    build_figure_3()
    if metrics_df is not None:
        build_figure_4(metrics_df)
    build_figure_5(condition_evals)
    build_figure_6a(condition_evals)
    build_figure_6b(condition_evals)
    build_figure_7(condition_evals)
    build_figure_8(condition_evals)
    build_figure_9(condition_evals)
    if metrics_df is not None:
        build_figure_10(metrics_df, condition_evals)
    build_figure_11(condition_evals)
    if spectral_df is not None:
        build_figure_12(spectral_df)
    if metrics_df is not None:
        build_figure_13(metrics_df)

    build_new_figure_a(condition_evals)
    build_new_figure_b(condition_evals)
    build_new_figure_c()
    build_new_figure_d(condition_evals)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
