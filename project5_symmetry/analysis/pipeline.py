#!/usr/bin/env python3
"""
Analysis pipeline for project5_symmetry symmetry-group sweep (s4/s2/s1).

Consumes the aggregated output from run_sweep.py
  (project5_symmetry/results/symmetry_sweep/symmetry_sweep_raw.pkl)
and the per-seed outputs from the phase-based sweep
  (project5_symmetry/results/*/seed_*/metrics.json).

Usage:
    python project5_symmetry/analysis/pipeline.py
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from project5_symmetry.evaluation.metrics import manifold_id, srsa
from project5_symmetry.training.dataset import TrajectoryDataset
from project5_symmetry.training.train import _collect_hidden_states


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SYMMETRY_SWEEP_PATHS = [
    ROOT / "project5_symmetry" / "results" / "symmetry_sweep",
    ROOT / "project5_symmetry" / "results" / "symmetry_sweep_validate",
    ROOT / "outputs" / "symmetry_validate",
]
CONDITION_ORDER = ["s4", "s2", "s1"]
DISPLAY = {"s4": "S4", "s2": "S2", "s1": "S1"}
COLORS = {"s4": "#2166ac", "s2": "#ef8a62", "s1": "#4daf4a"}
N_SHUFFLES = 1000
N_RANDOM_SCI_PAIRS = 10_000
N_DTG_PERMUTATIONS = 10_000


@dataclass
class SeedRecord:
    condition: str
    seed_name: str
    seed_index: int
    condition_dir: Path
    seed_dir: Path


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _sem(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _condition_label(condition: str) -> str:
    return DISPLAY.get(condition, condition.upper())


def _save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def discover_symmetry_sweep_root() -> Path | None:
    """Find the sweep output directory containing symmetry_sweep_raw.pkl."""
    for candidate in SYMMETRY_SWEEP_PATHS:
        if candidate.exists() and (candidate / "symmetry_sweep_raw.pkl").exists():
            return candidate
    return None


def load_symmetry_sweep_results(results_root: Path) -> dict | None:
    """Load the aggregated results from run_sweep.py output."""
    pkl_path = results_root / "symmetry_sweep_raw.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_per_seed_metrics(results_root: Path) -> list[dict]:
    """Load per-seed metrics.json files from condition/seed_* directories."""
    records = []
    for condition in CONDITION_ORDER:
        cond_dir = results_root / condition
        if not cond_dir.is_dir():
            continue
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                data = json.load(f)
            data["condition"] = data.get("condition_id", condition)
            data["seed_index"] = int(seed_dir.name.split("_")[-1])
            records.append(data)
    return records


def run_A1(results_dir: Path) -> pd.DataFrame:
    """Load symmetry sweep results into a DataFrame."""
    records = load_per_seed_metrics(results_dir)
    if not records:
        raw = load_symmetry_sweep_results(results_dir)
        if raw is None:
            print("WARNING: No symmetry sweep results found.")
            return pd.DataFrame()
        rows = []
        for condition, seed_list in raw.items():
            for seed_data in seed_list:
                row = {"condition": condition}
                row.update(seed_data)
                rows.append(row)
        return pd.DataFrame(rows)

    rows = []
    for rec in records:
        rows.append({
            "condition": rec.get("condition", "unknown"),
            "seed": rec.get("seed_index", -1),
            "sRSA_euclid": rec.get("final_srsa_euclid"),
            "sRSA_city": rec.get("final_srsa_city"),
            "SCI": rec.get("final_sci"),
            "manifold_id": rec.get("final_manifold_id"),
        })
    return pd.DataFrame(rows)


def run_A2(a1: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per symmetry condition."""
    if a1.empty:
        return pd.DataFrame()
    stats_rows = []
    for condition in CONDITION_ORDER:
        subset = a1[a1["condition"] == condition]
        vals = subset["sRSA_euclid"].dropna()
        stats_rows.append({
            "condition": condition,
            "mean_sRSA": vals.mean() if len(vals) > 0 else float("nan"),
            "sem_sRSA": _sem(vals) if len(vals) > 1 else float("nan"),
            "n_seeds": len(vals),
        })
    stats_df = pd.DataFrame(stats_rows)
    out_path = RESULTS_DIR / "symmetry_summary_statistics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    print(f"Summary written to {out_path}")
    return stats_df


def run_A3(stats_df: pd.DataFrame) -> None:
    """Generate summary bar plot — sRSA by symmetry condition."""
    if stats_df.empty:
        print("WARNING: No data for A3 plot.")
        _save_placeholder(FIGURES_DIR / "srsa_by_symmetry.png",
                          "sRSA by Symmetry Condition",
                          "No data available to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = stats_df["condition"]
    means = stats_df["mean_sRSA"]
    sems = stats_df["sem_sRSA"]
    colors = [COLORS.get(c, "#333333") for c in conditions]
    ax.bar(range(len(conditions)), means, yerr=sems, color=colors, capsize=5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([DISPLAY.get(c, c) for c in conditions])
    ax.set_ylabel("sRSA (Euclidean)")
    ax.set_title("Representational Similarity by Symmetry Condition")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "srsa_by_symmetry.png", dpi=200)
    plt.close(fig)
    print("Figure saved: figures/srsa_by_symmetry.png")


def run_A4(stats_df: pd.DataFrame) -> None:
    """Generate per-seed scatter plot."""
    pass


def run_A5(a1: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Statistical tests across symmetry conditions."""
    if stats_df.empty or a1.empty:
        return stats_df
    test_rows = []
    for i, c1 in enumerate(CONDITION_ORDER):
        for c2 in CONDITION_ORDER[i + 1:]:
            v1 = a1[a1["condition"] == c1]["sRSA_euclid"].dropna()
            v2 = a1[a1["condition"] == c2]["sRSA_euclid"].dropna()
            if len(v1) > 0 and len(v2) > 0:
                stat, pval = mannwhitneyu(v1, v2, alternative="two-sided")
                test_rows.append({
                    "comparison": f"{c1}_vs_{c2}",
                    "U_statistic": stat,
                    "p_value": pval,
                })
    if test_rows:
        test_df = pd.DataFrame(test_rows)
        out_path = RESULTS_DIR / "symmetry_condition_comparisons.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(out_path, index=False)
        print(f"Comparison stats written to {out_path}")
    return stats_df


def main() -> None:
    ensure_dirs()
    results_root = discover_symmetry_sweep_root()
    if results_root is None:
        print("No symmetry sweep results found. Looked in:")
        for p in SYMMETRY_SWEEP_PATHS:
            print(f"  {p}")
        print("Run project5_symmetry/experiments/run_sweep.py first.")
        return

    print(f"Found results in: {results_root}")
    a1 = run_A1(results_root)
    if a1.empty:
        print("No usable data found.")
        return
    stats_df = run_A2(a1)
    run_A3(stats_df)
    run_A4(stats_df)
    stats_df = run_A5(a1, stats_df)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
