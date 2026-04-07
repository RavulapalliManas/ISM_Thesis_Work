from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from analysis.representationalGeometryAnalysis import representationalGeometryAnalysis as RGA
from project3_generalization.evaluation.topology import compute_betti_numbers


def _match_samples(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)


def participation_ratio(hidden_states: np.ndarray) -> float:
    centered = np.asarray(hidden_states, dtype=float) - np.mean(hidden_states, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), a_min=0.0, a_max=None)
    numerator = np.square(np.sum(eigenvalues))
    denominator = np.sum(np.square(eigenvalues)) + 1e-12
    return float(numerator / denominator)


def compute_tuning_curves(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    grid_size: int = 20,
) -> dict[str, np.ndarray]:
    hidden_states = np.asarray(hidden_states, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n_units = hidden_states.shape[1]
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-9)
    xs = np.clip(((positions[:, 0] - mins[0]) / ranges[0] * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip(((positions[:, 1] - mins[1]) / ranges[1] * grid_size).astype(int), 0, grid_size - 1)

    occupancy = np.zeros((grid_size, grid_size), dtype=float)
    tuning = np.zeros((n_units, grid_size, grid_size), dtype=float)
    np.add.at(occupancy, (ys, xs), 1.0)
    for unit in range(n_units):
        np.add.at(tuning[unit], (ys, xs), hidden_states[:, unit])
    tuning = tuning / np.maximum(occupancy[None, :, :], 1.0)

    occupancy_prob = occupancy / np.maximum(occupancy.sum(), 1.0)
    mean_rate = hidden_states.mean(axis=0)
    ratio = tuning / np.maximum(mean_rate[:, None, None], 1e-9)
    spatial_info = np.sum(
        occupancy_prob[None, :, :] * tuning * np.log2(np.maximum(ratio, 1e-9)),
        axis=(1, 2),
    )
    peak_bins = np.stack(np.unravel_index(np.argmax(tuning.reshape(n_units, -1), axis=1), (grid_size, grid_size)), axis=1)
    return {
        "tuning_curves": tuning,
        "occupancy": occupancy,
        "spatial_information": spatial_info,
        "peak_bins": peak_bins,
    }


def fraction_spatially_tuned(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    grid_size: int = 20,
    spatial_information_threshold: float = 0.1,
) -> dict[str, Any]:
    tuning = compute_tuning_curves(hidden_states, positions, grid_size=grid_size)
    tuned = tuning["spatial_information"] > spatial_information_threshold
    return {
        "fraction": float(np.mean(tuned)),
        "tuned_mask": tuned,
        "spatial_information": tuning["spatial_information"],
        "peak_bins": tuning["peak_bins"],
        "tuning_curves": tuning["tuning_curves"],
        "occupancy": tuning["occupancy"],
    }


def BG1_trials_to_criterion(
    model: Any = None,
    env: Any = None,
    *,
    criterion: float = 0.9,
    curve: Sequence[float] | None = None,
) -> float:
    if curve is None:
        curve = getattr(model, "training_curve", None)
    if curve is None and env is not None:
        curve = getattr(env, "training_curve", None)
    if curve is None:
        raise ValueError("A training curve must be provided or accessible on the model/env.")
    curve_arr = np.asarray(curve, dtype=float)
    asymptote = np.nanmax(curve_arr)
    threshold = criterion * asymptote
    reached = np.flatnonzero(curve_arr >= threshold)
    return float(reached[0]) if reached.size else float(np.inf)


def BG2_zero_shot_accuracy(
    model: Any = None,
    env: Any = None,
    *,
    predicted_sr: np.ndarray | None = None,
    true_sr: np.ndarray | None = None,
) -> float:
    if predicted_sr is None or true_sr is None:
        if model is None or env is None or not hasattr(model, "predict_successor_representation"):
            raise ValueError("Provide predicted_sr and true_sr, or a model with predict_successor_representation().")
        predicted_sr = model.predict_successor_representation(env)
        true_sr = env.successor_representation
    error = np.linalg.norm(predicted_sr - true_sr, ord="fro")
    norm = np.linalg.norm(true_sr, ord="fro") + 1e-12
    return float(1.0 - error / norm)


def RG1_sRSA(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    neural_metric: str = "cosine",
    spatial_metric: str = "euclidean",
) -> float:
    wake = {
        "h": np.asarray(hidden_states, dtype=float),
        "state": {"agent_pos": np.asarray(positions, dtype=float)},
    }
    rsa, _, _, _ = RGA.calculateRSA_space(RGA, wake, metric=neural_metric, spacemetric=spatial_metric)
    return float(rsa[0])


def RG2_CERA(H_A: np.ndarray, H_B: np.ndarray) -> float:
    H_A, H_B = _match_samples(H_A, H_B)
    H_A = H_A - H_A.mean(axis=0, keepdims=True)
    H_B = H_B - H_B.mean(axis=0, keepdims=True)
    rotation, scale = orthogonal_procrustes(H_A, H_B)
    aligned = H_A @ rotation * scale
    return float(np.linalg.norm(aligned - H_B, ord="fro") / (np.linalg.norm(H_B, ord="fro") + 1e-12))


def _center_gram(gram: np.ndarray) -> np.ndarray:
    n = gram.shape[0]
    unit = np.ones((n, n), dtype=float) / n
    return gram - unit @ gram - gram @ unit + unit @ gram @ unit


def _unbiased_hsic(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    n = gram_x.shape[0]
    if n < 4:
        return float("nan")
    gram_x = gram_x.copy()
    gram_y = gram_y.copy()
    np.fill_diagonal(gram_x, 0.0)
    np.fill_diagonal(gram_y, 0.0)
    term1 = np.trace(gram_x @ gram_y)
    term2 = gram_x.sum() * gram_y.sum() / ((n - 1) * (n - 2))
    term3 = 2 * np.sum(gram_x.sum(axis=0) * gram_y.sum(axis=0)) / (n - 2)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def RG3_CKA(H_A: np.ndarray, H_B: np.ndarray) -> float:
    H_A, H_B = _match_samples(H_A, H_B)
    gram_a = _center_gram(H_A @ H_A.T)
    gram_b = _center_gram(H_B @ H_B.T)
    hsic_ab = _unbiased_hsic(gram_a, gram_b)
    hsic_aa = _unbiased_hsic(gram_a, gram_a)
    hsic_bb = _unbiased_hsic(gram_b, gram_b)
    return float(hsic_ab / np.sqrt((hsic_aa * hsic_bb) + 1e-12))


def RG4_betti_numbers(hidden_states: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    return compute_betti_numbers(hidden_states, **kwargs)


def SG1_SR_error(M_hat_B: np.ndarray, M_true_B: np.ndarray) -> float:
    return float(np.linalg.norm(M_hat_B - M_true_B, ord="fro") / (np.linalg.norm(M_true_B, ord="fro") + 1e-12))


def SG2_transfer_vs_similarity(
    transfer_efficiency_matrix: np.ndarray,
    sim_matrix: np.ndarray,
) -> dict[str, float]:
    triu = np.triu_indices_from(sim_matrix, k=1)
    transfer = np.asarray(transfer_efficiency_matrix)[triu]
    sim = np.asarray(sim_matrix)[triu]
    keep = np.isfinite(transfer) & np.isfinite(sim)
    if np.sum(keep) < 2:
        return {"r": float("nan"), "p": float("nan"), "n_pairs": 0}
    r_value, p_value = pearsonr(sim[keep], transfer[keep])
    return {"r": float(r_value), "p": float(p_value), "n_pairs": int(np.sum(keep))}


def SG3_eigenspectrum_overlap(M_A: np.ndarray, M_B: np.ndarray, k: int = 10) -> float:
    eigvals_a, eigvecs_a = np.linalg.eig(M_A)
    eigvals_b, eigvecs_b = np.linalg.eig(M_B)
    order_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
    order_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
    vecs_a = np.real(eigvecs_a[:, order_a])
    vecs_b = np.real(eigvecs_b[:, order_b])
    vecs_a = vecs_a / (np.linalg.norm(vecs_a, axis=0, keepdims=True) + 1e-12)
    vecs_b = vecs_b / (np.linalg.norm(vecs_b, axis=0, keepdims=True) + 1e-12)
    similarity = np.abs(np.sum(vecs_a * vecs_b, axis=0))
    return float(np.mean(similarity))


def GG1_elongation_index(hidden_states: np.ndarray, positions: np.ndarray | None = None) -> float:
    embedding = PCA(n_components=2).fit_transform(np.asarray(hidden_states, dtype=float))
    singular_values = PCA(n_components=2).fit(embedding).singular_values_
    return float(singular_values[0] / (singular_values[1] + 1e-12))


def GG2_field_size_anisotropy(
    tuning_maps_3d: np.ndarray,
    alpha_motion: float | None = None,
) -> dict[str, float]:
    tuning_maps_3d = np.asarray(tuning_maps_3d, dtype=float)
    x = np.arange(tuning_maps_3d.shape[1], dtype=float)
    y = np.arange(tuning_maps_3d.shape[2], dtype=float)
    z = np.arange(tuning_maps_3d.shape[3], dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    horizontal_sigmas: list[float] = []
    vertical_sigmas: list[float] = []
    for field in tuning_maps_3d:
        weights = np.maximum(field, 0.0)
        total = weights.sum() + 1e-12
        mean_x = float((weights * xx).sum() / total)
        mean_y = float((weights * yy).sum() / total)
        mean_z = float((weights * zz).sum() / total)
        sigma_x = float(np.sqrt((weights * (xx - mean_x) ** 2).sum() / total))
        sigma_y = float(np.sqrt((weights * (yy - mean_y) ** 2).sum() / total))
        sigma_z = float(np.sqrt((weights * (zz - mean_z) ** 2).sum() / total))
        horizontal_sigmas.append(0.5 * (sigma_x + sigma_y))
        vertical_sigmas.append(sigma_z)

    far = float(np.mean(horizontal_sigmas) / (np.mean(vertical_sigmas) + 1e-12))
    return {"FAR": far, "alpha_motion": float(alpha_motion) if alpha_motion is not None else float("nan")}


def GG3_topological_remapping_index(
    H_A: np.ndarray,
    H_B: np.ndarray,
    positions: np.ndarray | None = None,
) -> float:
    H_A, H_B = _match_samples(H_A, H_B)
    similarity = cosine_similarity(H_A, H_B)
    return float(np.mean(np.diag(similarity)))


def replay_quality(
    wake_hidden: np.ndarray,
    wake_positions: np.ndarray,
    spontaneous_hidden: np.ndarray,
    *,
    extent: Sequence[float] | None = None,
) -> float:
    from sklearn.neighbors import KNeighborsRegressor

    decoder = KNeighborsRegressor(n_neighbors=5, weights="distance")
    decoder.fit(wake_hidden, wake_positions)
    decoded = decoder.predict(spontaneous_hidden)
    step_lengths = np.linalg.norm(np.diff(decoded, axis=0), axis=1)
    if extent is None:
        scale = np.linalg.norm(wake_positions.max(axis=0) - wake_positions.min(axis=0)) + 1e-12
    else:
        left, right, bottom, top = extent
        scale = np.linalg.norm([right - left, top - bottom]) + 1e-12
    return float(np.exp(-np.mean(step_lengths) / scale))


__all__ = [
    "BG1_trials_to_criterion",
    "BG2_zero_shot_accuracy",
    "GG1_elongation_index",
    "GG2_field_size_anisotropy",
    "GG3_topological_remapping_index",
    "RG1_sRSA",
    "RG2_CERA",
    "RG3_CKA",
    "RG4_betti_numbers",
    "SG1_SR_error",
    "SG2_transfer_vs_similarity",
    "SG3_eigenspectrum_overlap",
    "compute_tuning_curves",
    "fraction_spatially_tuned",
    "participation_ratio",
    "replay_quality",
]
