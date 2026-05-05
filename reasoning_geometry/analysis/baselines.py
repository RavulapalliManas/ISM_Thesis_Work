from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from reasoning_geometry.common import TrajectoryBundle
from reasoning_geometry.metrics.statistics import (
    bootstrap_auroc,
    compute_auroc,
    detection_latency,
    precision_at_fpr,
)


@dataclass
class DetectorResult:
    name: str
    scores: List[float]
    auroc: float
    ci_low: float
    ci_high: float
    precision_at_10_fpr: float
    detection_latency: float
    curves: Dict[str, object]


def iti_probe_scores(
    train_bundles: Sequence[TrajectoryBundle],
    eval_bundles: Sequence[TrajectoryBundle],
    layer: str = "16",
) -> List[float]:
    x_train: List[np.ndarray] = []
    y_train: List[int] = []
    for bundle in train_bundles:
        states = bundle.layer_hidden_states.get(layer, bundle.hidden_states)
        x_train.append(states.mean(axis=0))
        y_train.append(bundle.label)
    x_eval = [bundle.layer_hidden_states.get(layer, bundle.hidden_states).mean(axis=0) for bundle in eval_bundles]
    clf = LogisticRegression(max_iter=2000, penalty="l2")
    clf.fit(np.stack(x_train), np.asarray(y_train))
    probs = clf.predict_proba(np.stack(x_eval))[:, 1]
    return probs.tolist()


def nearest_neighbour_centroid_scores(
    train_bundles: Sequence[TrajectoryBundle],
    eval_bundles: Sequence[TrajectoryBundle],
) -> List[float]:
    factual_states = np.concatenate(
        [bundle.hidden_states for bundle in train_bundles if bundle.label == 0], axis=0
    )
    centroid = factual_states.mean(axis=0)
    scores = []
    for bundle in eval_bundles:
        distances = np.linalg.norm(bundle.hidden_states - centroid, axis=1)
        scores.append(float(distances.mean()))
    return scores


def aggregate_detector(
    name: str,
    y_true: Sequence[int],
    scores: Sequence[float],
    stepwise_scores: Sequence[Sequence[float]] | None = None,
    greater_is_positive: bool = True,
) -> DetectorResult:
    curves = compute_auroc(y_true, scores)
    ci = bootstrap_auroc(y_true, scores)
    threshold = float(np.median(scores))
    latency = 0.0
    if stepwise_scores is not None:
        latency = detection_latency(stepwise_scores, y_true, threshold, greater_is_positive=greater_is_positive)
    return DetectorResult(
        name=name,
        scores=list(map(float, scores)),
        auroc=float(curves["auroc"]),
        ci_low=ci["ci_low"],
        ci_high=ci["ci_high"],
        precision_at_10_fpr=precision_at_fpr(y_true, scores, target_fpr=0.10),
        detection_latency=latency,
        curves=curves,
    )
