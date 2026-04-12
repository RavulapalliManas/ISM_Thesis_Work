"""Persistence-diagram tracking across training."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from project4_topology_before_geometry.evaluation.topological_metrics import compute_betti_numbers


def compute_persistence_diagram_evolution(trainer, env, model, eval_trials: Iterable[int], seed: int = 42):
    """Collect persistence diagrams from saved checkpoints or in-memory tracker snapshots."""
    results: dict[int, dict[str, object]] = {}
    for trial in eval_trials:
        snapshot = trainer.get_hidden_snapshot(trial, env=env, model=model, seed=seed)
        if snapshot is None:
            continue
        betti = compute_betti_numbers(snapshot, seed=seed)
        results[int(trial)] = {"diagrams": betti.get("diagrams", []), "betti": betti}
    return results


def compute_b1_persistence_vs_radius(inner_radii, results_by_radius):
    """Summarize how the dominant H1 lifetime scales with annulus hole size."""
    radii = np.asarray(inner_radii, dtype=np.float32)
    lifetimes = np.asarray(
        [float(np.max(result.get("dim1_lifetimes", [0.0]))) if len(result.get("dim1_lifetimes", [])) else 0.0 for result in results_by_radius],
        dtype=np.float32,
    )
    if len(radii) < 2:
        return {"r2": float("nan"), "slope": float("nan"), "intercept": float("nan"), "lifetimes": lifetimes}
    coeffs = np.polyfit(radii, lifetimes, deg=1)
    fit = np.polyval(coeffs, radii)
    ss_res = float(np.sum((lifetimes - fit) ** 2))
    ss_tot = float(np.sum((lifetimes - lifetimes.mean()) ** 2) + 1e-9)
    return {
        "r2": float(1.0 - ss_res / ss_tot),
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "lifetimes": lifetimes,
    }

