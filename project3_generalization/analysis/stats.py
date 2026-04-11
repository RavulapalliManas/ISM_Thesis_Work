"""
File: project3_generalization/analysis/stats.py

Description:
Small statistical utility functions used by Project 3 analysis code.

Role in system:
Provides a minimal, dependency-light layer for common effect-size and
multiple-testing calculations that would otherwise be repeated across notebooks
and post-processing scripts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two one-dimensional samples."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    mean_diff = sample_a.mean() - sample_b.mean()
    pooled_var = (
        ((len(sample_a) - 1) * sample_a.var(ddof=1)) + ((len(sample_b) - 1) * sample_b.var(ddof=1))
    ) / max(len(sample_a) + len(sample_b) - 2, 1)
    return float(mean_diff / np.sqrt(pooled_var + 1e-12))


def pearson_r(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return the Pearson correlation coefficient and p-value for two samples."""
    r_value, p_value = pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {"r": float(r_value), "p": float(p_value)}


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray | float]:
    """Apply the Benjamini-Hochberg false-discovery-rate procedure."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = alpha * (np.arange(1, len(ranked) + 1) / max(len(ranked), 1))
    below = ranked <= thresholds
    cutoff = ranked[np.max(np.nonzero(below))] if np.any(below) else 0.0
    return {"rejected": p_values <= cutoff, "cutoff": float(cutoff)}


__all__ = [
    "cohens_d",
    "fdr_bh",
    "pearson_r",
]
