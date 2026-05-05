from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from reasoning_geometry.metrics.srsa import spearman_rsa


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    reduction: Callable[[np.ndarray], float] | None = None,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    reduction = reduction or np.mean
    samples = []
    for _ in range(n_bootstrap):
        draw = np.random.choice(values, size=values.size, replace=True)
        samples.append(reduction(draw))
    return float(np.percentile(samples, 100 * alpha / 2)), float(
        np.percentile(samples, 100 * (1 - alpha / 2))
    )


def symmetric_permutation_test(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    n_permutations: int = 1000,
) -> Dict[str, float | List[float]]:
    observed, _ = spearman_rsa(matrix_a, matrix_b)
    null_values: List[float] = []
    for _ in range(n_permutations):
        perm = np.random.permutation(matrix_b.shape[0])
        permuted = matrix_b[perm][:, perm]
        rho, _ = spearman_rsa(matrix_a, permuted)
        null_values.append(rho)
    null_values_arr = np.asarray(null_values, dtype=np.float64)
    p_value = float(np.mean(null_values_arr >= observed))
    ci_low, ci_high = bootstrap_confidence_interval([observed], n_bootstrap=max(10, n_permutations // 10))
    return {
        "observed": float(observed),
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "null_distribution": null_values_arr.tolist(),
    }


def grassmann_permutation_test(
    factual_states: np.ndarray,
    hallucinated_states: np.ndarray,
    k: int,
    n_permutations: int,
    subspace_fn: Callable[[np.ndarray, int], np.ndarray],
    angle_fn: Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray | float]],
) -> Dict[str, float | List[float]]:
    factual_basis = subspace_fn(factual_states, k)
    hall_basis = subspace_fn(hallucinated_states, k)
    observed_bundle = angle_fn(factual_basis, hall_basis)
    observed = float(observed_bundle["grassmann_distance"])

    all_states = np.concatenate([factual_states, hallucinated_states], axis=0)
    factual_n = factual_states.shape[0]
    null_values: List[float] = []
    for _ in range(n_permutations):
        perm = np.random.permutation(all_states.shape[0])
        perm_f = all_states[perm[:factual_n]]
        perm_h = all_states[perm[factual_n:]]
        d_g = angle_fn(subspace_fn(perm_f, k), subspace_fn(perm_h, k))["grassmann_distance"]
        null_values.append(float(d_g))
    null_arr = np.asarray(null_values, dtype=np.float64)
    p_value = float(np.mean(null_arr >= observed))
    effect = float((observed - null_arr.mean()) / np.clip(null_arr.std(ddof=1), 1e-12, None))
    return {
        "observed": observed,
        "p_value": p_value,
        "effect_size": effect,
        "null_distribution": null_arr.tolist(),
        "null_95": float(np.percentile(null_arr, 95.0)),
    }


def compute_auroc(y_true: Sequence[int], scores: Sequence[float]) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    return {
        "auroc": float(roc_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
        "auprc": float(auc(recall, precision)),
    }


def bootstrap_auroc(y_true: Sequence[int], scores: Sequence[float], n_bootstrap: int = 1000) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(y_true.shape[0], size=y_true.shape[0], replace=True)
        sample_y = y_true[indices]
        sample_s = scores[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        values.append(roc_auc_score(sample_y, sample_s))
    if not values:
        values = [0.5]
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def precision_at_fpr(y_true: Sequence[int], scores: Sequence[float], target_fpr: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    threshold = thresholds[idx]
    preds = scores >= threshold
    true_pos = np.sum((preds == 1) & (y_true == 1))
    pred_pos = np.sum(preds == 1)
    return float(true_pos / pred_pos) if pred_pos > 0 else 0.0


def detection_latency(
    trajectories: Sequence[Sequence[float]],
    labels: Sequence[int],
    threshold: float,
    greater_is_positive: bool = True,
) -> float:
    latencies: List[int] = []
    for signal, label in zip(trajectories, labels):
        if int(label) != 1:
            continue
        array = np.asarray(signal, dtype=np.float64)
        fired = np.where(array >= threshold)[0] if greater_is_positive else np.where(array <= threshold)[0]
        latencies.append(int(fired[0]) if fired.size else len(array))
    return float(np.mean(latencies)) if latencies else 0.0
