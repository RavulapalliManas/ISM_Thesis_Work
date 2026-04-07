from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from analysis.representationalGeometryAnalysis import representationalGeometryAnalysis as RGA
from project3_generalization.evaluation.topology import compute_betti_numbers


def _match_samples(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)


def _subsample_indices(n_samples: int, max_samples: int | None, seed: int = 0) -> np.ndarray:
    if max_samples is None or n_samples <= max_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def _subsample_aligned(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_states = np.asarray(hidden_states, dtype=float)
    positions = np.asarray(positions, dtype=float)
    keep = _subsample_indices(min(len(hidden_states), len(positions)), max_samples, seed=seed)
    return hidden_states[keep], positions[keep]


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
    max_samples: int | None = None,
    seed: int = 0,
) -> float:
    hidden_states, positions = _subsample_aligned(hidden_states, positions, max_samples=max_samples, seed=seed)
    spatial_positions = np.asarray(positions, dtype=float)
    if len(spatial_positions) == len(hidden_states):
        spatial_positions = np.vstack([spatial_positions, spatial_positions[-1:]])
    elif len(spatial_positions) > len(hidden_states) + 1:
        spatial_positions = spatial_positions[: len(hidden_states) + 1]
    wake = {
        "h": np.asarray(hidden_states, dtype=float),
        "state": {"agent_pos": spatial_positions},
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


def RG3_CKA(
    H_A: np.ndarray,
    H_B: np.ndarray,
    *,
    batch_size: int | None = None,
) -> float:
    H_A, H_B = _match_samples(H_A, H_B)
    H_A = np.asarray(H_A, dtype=np.float64)
    H_B = np.asarray(H_B, dtype=np.float64)
    H_A = H_A - H_A.mean(axis=0, keepdims=True)
    H_B = H_B - H_B.mean(axis=0, keepdims=True)

    if batch_size is None or batch_size >= len(H_A):
        cross = H_A.T @ H_B
        auto_a = H_A.T @ H_A
        auto_b = H_B.T @ H_B
    else:
        cross = np.zeros((H_A.shape[1], H_B.shape[1]), dtype=np.float64)
        auto_a = np.zeros((H_A.shape[1], H_A.shape[1]), dtype=np.float64)
        auto_b = np.zeros((H_B.shape[1], H_B.shape[1]), dtype=np.float64)
        for start in range(0, len(H_A), batch_size):
            end = min(start + batch_size, len(H_A))
            a_batch = H_A[start:end]
            b_batch = H_B[start:end]
            cross += a_batch.T @ b_batch
            auto_a += a_batch.T @ a_batch
            auto_b += b_batch.T @ b_batch

    numerator = np.sum(cross ** 2)
    denominator = np.sqrt(np.sum(auto_a ** 2) * np.sum(auto_b ** 2)) + 1e-12
    return float(numerator / denominator)


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


def estimate_neural_sr(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    *,
    extent: Sequence[float],
    grid_size: int = 30,
) -> np.ndarray:
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    n_states = grid_size * grid_size
    left, right, bottom, top = extent
    xs = np.clip(((positions[:, 0] - left) / max(right - left, 1e-6) * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip(((positions[:, 1] - bottom) / max(top - bottom, 1e-6) * grid_size).astype(int), 0, grid_size - 1)
    flat = ys * grid_size + xs

    state_sums = np.zeros((n_states, hidden_states.shape[1]), dtype=np.float32)
    counts = np.zeros((n_states,), dtype=np.float32)
    np.add.at(state_sums, flat, hidden_states)
    np.add.at(counts, flat, 1.0)
    state_means = state_sums / np.maximum(counts[:, None], 1.0)
    norms = np.linalg.norm(state_means, axis=1, keepdims=True)
    state_means = state_means / np.maximum(norms, 1e-6)
    neural_sr = state_means @ state_means.T
    neural_sr = np.clip(neural_sr, a_min=0.0, a_max=None)
    np.fill_diagonal(neural_sr, 1.0)
    row_sums = neural_sr.sum(axis=1, keepdims=True)
    neural_sr = neural_sr / np.maximum(row_sums, 1e-6)
    return neural_sr.astype(np.float32)


def current_environment_sr_error(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    true_sr: np.ndarray,
    *,
    extent: Sequence[float],
    grid_size: int = 30,
) -> float:
    neural_sr = estimate_neural_sr(hidden_states, positions, extent=extent, grid_size=grid_size)
    return SG1_SR_error(neural_sr, np.asarray(true_sr, dtype=np.float32))


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
    "current_environment_sr_error",
    "estimate_neural_sr",
    "fraction_spatially_tuned",
    "participation_ratio",
    "replay_quality",
]
