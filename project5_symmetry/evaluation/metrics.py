"""
Evaluation metrics for project5_symmetry.

All functions are stateless and take plain numpy arrays.
Faithful to Levenstein et al. 2024 analysis (Methods pp.17-19).

New metrics (our contributions):
  sci          — Symmetry Collapse Index
  dtg_curve    — Topology-Geometry Gap (ΔTG)
  manifold_id  — TwoNN intrinsic dimensionality

Paper metrics (re-implemented to avoid class dependency):
  srsa                 — Spatial RSA (Spearman r, cosine neural / Euclidean spatial)
  compute_tuning_curves — 2D tuning curves via pynapple
  spatial_information   — Skaggs spatial information per unit
  spatial_evs           — Spatial explained variance per unit
"""

import numpy as np
import torch
import pynapple as nap
from scipy.spatial.distance import pdist, cdist
from scipy.signal import correlate2d
from scipy.stats import spearmanr

MAX_SUBSAMPLE = 5000  # larger sample reduces checkpoint-to-checkpoint variance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(hidden: np.ndarray, positions: np.ndarray, n: int):
    if hidden.shape[0] > n:
        idx = np.random.choice(hidden.shape[0], n, replace=False)
        return hidden[idx], positions[idx]
    return hidden, positions


def _pdist_cosine(X: np.ndarray) -> np.ndarray:
    """Vectorized pairwise cosine distances via matrix multiply — replaces pdist(X, 'cosine')."""
    X_t  = torch.from_numpy(X).float()
    norms = X_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_n  = X_t / norms
    cos_sim = X_n @ X_n.T
    D = torch.clamp(1.0 - cos_sim, min=0.0, max=2.0)
    i, j = torch.triu_indices(len(X_t), len(X_t), offset=1)
    return D[i, j].numpy()


def _pdist_euclidean(X: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Euclidean distances via matrix multiply."""
    X_t = torch.from_numpy(X).float()
    sq  = (X_t ** 2).sum(dim=1, keepdim=True)
    D   = sq + sq.T - 2 * X_t @ X_t.T
    D   = torch.clamp(D, min=0).sqrt()
    i, j = torch.triu_indices(len(X_t), len(X_t), offset=1)
    return D[i, j].numpy()


def _pdist_cityblock(X: np.ndarray) -> np.ndarray:
    """Vectorized pairwise CityBlock distances — chunked to avoid OOM."""
    X_t   = torch.from_numpy(X).float()
    n     = len(X_t)
    chunk = 256
    out   = []
    for i in range(0, n, chunk):
        diff = torch.abs(X_t[i:i + chunk].unsqueeze(1) - X_t.unsqueeze(0))
        out.append(diff.sum(dim=2))
    D    = torch.cat(out, dim=0)
    i, j = torch.triu_indices(n, n, offset=1)
    return D[i, j].numpy()


def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    X_t = torch.from_numpy(X).float()
    norms = X_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_n = X_t / norms
    cos_sim = X_n @ X_n.T
    return torch.clamp(1.0 - cos_sim, min=0.0, max=2.0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Paper metric 1 — sRSA
# ─────────────────────────────────────────────────────────────────────────────

def srsa(
    hidden: np.ndarray,
    positions: np.ndarray,
    neural_metric: str = 'cosine',
    space_metric: str = 'euclidean',
    max_n: int = MAX_SUBSAMPLE,
    return_matrix: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Spatial RSA: Spearman r between cosine neural distances and Euclidean
    spatial distances across all pairs of timepoints (Methods p.19).

    hidden    : (T, H)
    positions : (T, 2)  col/row in MiniGrid coords
    """
    h, p = _subsample(hidden, positions, max_n)
    if neural_metric == 'cosine':
        neural_rsa_matrix = _cosine_distance_matrix(h)
        idx = np.triu_indices(neural_rsa_matrix.shape[0], k=1)
        neural_dists = neural_rsa_matrix[idx]
    else:
        neural_rsa_matrix = cdist(h, h, neural_metric)
        neural_dists = pdist(h, neural_metric)
    if space_metric == 'euclidean':
        spatial_dists = _pdist_euclidean(p)
    elif space_metric == 'cityblock':
        spatial_dists = _pdist_cityblock(p)
    else:
        spatial_dists = pdist(p, space_metric)
    srsa_value = float(spearmanr(neural_dists, spatial_dists).statistic)
    if return_matrix:
        return srsa_value, neural_rsa_matrix
    return srsa_value


def cross_seed_rsa_alignment(rsa_matrices):
    """
    Pairwise Spearman correlation between neural RSA matrices across seeds.

    Parameters
    ----------
    rsa_matrices : list of np.ndarray, each shape (N_pos, N_pos)
        Neural distance matrices from each trained seed, as produced
        by the sRSA pipeline. Pass the raw matrix, not the scalar sRSA.

    Returns
    -------
    dict:
        pairwise_rho : np.ndarray (K, K) symmetric
        mean_rho     : float — mean of upper triangle
        std_rho      : float — std of upper triangle
    """
    K = len(rsa_matrices)
    pairwise_rho = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            idx = np.triu_indices(rsa_matrices[a].shape[0], k=1)
            va = rsa_matrices[a][idx]
            vb = rsa_matrices[b][idx]
            rho, _ = spearmanr(va, vb)
            pairwise_rho[a, b] = rho
            pairwise_rho[b, a] = rho
    upper = pairwise_rho[np.triu_indices(K, k=1)]
    return {
        'pairwise_rho': pairwise_rho,
        'mean_rho': float(np.mean(upper)),
        'std_rho': float(np.std(upper)),
    }


def aggregate_hidden_by_position(
    hidden: np.ndarray,
    positions: np.ndarray,
    passable_positions: list[tuple[int, int]] | None = None,
) -> dict:
    """
    Average hidden states over repeated visits to the same discrete arena position.

    Returns a dict with:
      positions : (N_pos, 2) int array in (col, row) order
      hidden    : (N_pos, H) float32 mean hidden state per position
      counts    : (N_pos,) visit counts used in the mean
    """
    pos_int = np.rint(positions).astype(np.int32)
    if passable_positions is None:
        ordered = sorted({(int(c), int(r)) for c, r in pos_int})
    else:
        ordered = [(int(c), int(r)) for c, r in passable_positions]

    pos_to_idx = {pos: i for i, pos in enumerate(ordered)}
    accum = np.zeros((len(ordered), hidden.shape[1]), dtype=np.float64)
    counts = np.zeros(len(ordered), dtype=np.int32)

    for h_t, (c, r) in zip(hidden, pos_int):
        idx = pos_to_idx.get((int(c), int(r)))
        if idx is None:
            continue
        accum[idx] += h_t
        counts[idx] += 1

    valid = counts > 0
    return {
        'positions': np.array(ordered, dtype=np.int32)[valid],
        'hidden': (accum[valid] / counts[valid, None]).astype(np.float32),
        'counts': counts[valid],
    }


def top_cca_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Top canonical correlation between two position-aligned representation matrices.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2 or Yc.shape[0] < 2:
        return float('nan')
    qx, _ = np.linalg.qr(Xc, mode='reduced')
    qy, _ = np.linalg.qr(Yc, mode='reduced')
    if qx.size == 0 or qy.size == 0:
        return float('nan')
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s[0], -1.0, 1.0))


def cross_seed_cca_alignment(position_hidden_matrices: list[np.ndarray]) -> dict:
    """
    Pairwise top-CCA alignment between seeds using matched position-mean hidden states.
    """
    K = len(position_hidden_matrices)
    pairwise_top_cca = np.eye(K, dtype=np.float64)
    for a in range(K):
        for b in range(a + 1, K):
            corr = top_cca_correlation(position_hidden_matrices[a], position_hidden_matrices[b])
            pairwise_top_cca[a, b] = corr
            pairwise_top_cca[b, a] = corr
    upper = pairwise_top_cca[np.triu_indices(K, k=1)]
    return {
        'pairwise_top_cca': pairwise_top_cca,
        'mean_top_cca': float(np.nanmean(upper)),
        'std_top_cca': float(np.nanstd(upper)),
    }


def _exact_tuning_maps(
    hidden: np.ndarray,
    positions: np.ndarray,
    arena_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos_int = np.rint(positions).astype(np.int32)
    accum = np.zeros((hidden.shape[1], arena_size, arena_size), dtype=np.float64)
    counts = np.zeros((arena_size, arena_size), dtype=np.float64)

    for h_t, (c, r) in zip(hidden, pos_int):
        if 1 <= c <= arena_size and 1 <= r <= arena_size:
            accum[:, r - 1, c - 1] += h_t
            counts[r - 1, c - 1] += 1.0

    maps = np.divide(
        accum,
        counts[None, :, :],
        out=np.zeros_like(accum),
        where=counts[None, :, :] > 0,
    )
    maps[:, counts == 0] = np.nan
    return maps, counts


def _spatial_evs_exact(
    hidden: np.ndarray,
    positions: np.ndarray,
    tuning_maps: np.ndarray,
) -> np.ndarray:
    pos_int = np.rint(positions).astype(np.int32)
    T, H = hidden.shape
    expected = np.zeros((T, H), dtype=np.float64)
    valid = np.zeros(T, dtype=bool)

    for t, (c, r) in enumerate(pos_int):
        if 1 <= c <= tuning_maps.shape[2] and 1 <= r <= tuning_maps.shape[1]:
            expected[t] = tuning_maps[:, r - 1, c - 1]
            valid[t] = True

    evs = np.zeros(H, dtype=np.float64)
    for i in range(H):
        mask = valid & np.isfinite(expected[:, i])
        if mask.sum() < 2:
            evs[i] = np.nan
            continue
        h_i = hidden[mask, i].astype(np.float64)
        var_total = np.var(h_i)
        if var_total <= 0:
            evs[i] = np.nan
            continue
        residual = h_i - expected[mask, i]
        evs[i] = 1.0 - np.var(residual) / var_total
    return evs


def place_field_spatial_coherence(
    hidden: np.ndarray,
    positions: np.ndarray,
    arena_size: int,
    evs_threshold: float = 0.10,
) -> dict:
    """
    Place-field coherence using exact discrete position tuning maps.

    For each sufficiently spatially explained unit, compute:
      autocorr(0,0) / mean autocorr over lags with Euclidean radius 2..5 tiles.
    """
    tuning_maps, occupancy = _exact_tuning_maps(hidden, positions, arena_size=arena_size)
    evs = _spatial_evs_exact(hidden, positions, tuning_maps)

    scores = np.full(hidden.shape[1], np.nan, dtype=np.float64)
    for i in range(hidden.shape[1]):
        if not np.isfinite(evs[i]) or evs[i] < evs_threshold:
            continue
        field = tuning_maps[i]
        if not np.isfinite(field).any():
            continue
        field = np.nan_to_num(field, nan=0.0)
        field = field - field.mean()
        ac = correlate2d(field, field, mode='full', boundary='fill', fillvalue=0.0)
        cy, cx = np.array(ac.shape) // 2
        yy, xx = np.indices(ac.shape)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        annulus = (dist >= 2.0) & (dist <= 5.0)
        annulus_vals = ac[annulus]
        annulus_mean = annulus_vals.mean() if annulus_vals.size else np.nan
        if not np.isfinite(annulus_mean) or annulus_mean <= 0:
            continue
        scores[i] = ac[cy, cx] / (annulus_mean + 1e-8)

    valid_scores = scores[np.isfinite(scores)]
    return {
        'per_unit_score': scores,
        'mean_score': float(np.nanmean(valid_scores)) if valid_scores.size else float('nan'),
        'std_score': float(np.nanstd(valid_scores)) if valid_scores.size else float('nan'),
        'n_valid_units': int(valid_scores.size),
        'evs_threshold': float(evs_threshold),
        'evs': evs,
        'occupancy': occupancy,
    }


def observation_discriminability(
    observations: np.ndarray,
    positions: np.ndarray,
    observation_metric: str = 'euclidean',
    space_metric: str = 'euclidean',
) -> dict:
    """
    Input-space sRSA between mean observations per position and spatial distance.
    """
    obs_dists = pdist(observations, observation_metric)
    if space_metric == 'euclidean':
        spatial_dists = _pdist_euclidean(positions)
    elif space_metric == 'cityblock':
        spatial_dists = _pdist_cityblock(positions)
    else:
        spatial_dists = pdist(positions, space_metric)
    rho = float(spearmanr(obs_dists, spatial_dists).statistic)
    return {
        'rho': rho,
        'observation_metric': observation_metric,
        'space_metric': space_metric,
    }


def _classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = distance_matrix.shape[0]
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * J @ (distance_matrix.astype(np.float64) ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = np.maximum(eigvals[:n_components], 0.0)
    return eigvecs[:, :n_components] * np.sqrt(keep)


def representational_geometry_consistency(
    hidden: np.ndarray,
    neural_metric: str = 'cosine',
) -> dict:
    """
    2D geometry consistency via PCA projection and classical-MDS stress.
    Lower stress means a cleaner 2D map-like representation.
    """
    hidden_centered = hidden - hidden.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(hidden_centered, full_matrices=False)
    if vt.shape[0] >= 2:
        pca_2d = hidden_centered @ vt[:2].T
    elif vt.shape[0] == 1:
        pca_2d = np.column_stack([hidden_centered @ vt[:1].T, np.zeros((hidden.shape[0], 1))])
    else:
        pca_2d = np.zeros((hidden.shape[0], 2), dtype=np.float64)
    pca_var_2d = float(np.sum(s[:2] ** 2) / (np.sum(s ** 2) + 1e-8))

    if neural_metric == 'cosine':
        neural_dist = _cosine_distance_matrix(hidden)
    else:
        neural_dist = cdist(hidden, hidden, neural_metric)
    mds_2d = _classical_mds(neural_dist, n_components=2)

    idx = np.triu_indices(neural_dist.shape[0], k=1)
    d_true = neural_dist[idx]
    d_pca = cdist(pca_2d, pca_2d, 'euclidean')[idx]
    d_mds = cdist(mds_2d, mds_2d, 'euclidean')[idx]

    denom = float(np.sum(d_true ** 2) + 1e-8)
    stress_pca = float(np.sqrt(np.sum((d_true - d_pca) ** 2) / denom))
    stress_mds = float(np.sqrt(np.sum((d_true - d_mds) ** 2) / denom))
    return {
        'stress': stress_mds,
        'stress_pca': stress_pca,
        'pca_var_2d': pca_var_2d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paper metric 2 — Spatial tuning curves (pynapple)
# ─────────────────────────────────────────────────────────────────────────────

def compute_tuning_curves(
    hidden: np.ndarray,
    positions: np.ndarray,
    nb_bins: int = 20,
) -> tuple:
    """
    Compute 2D spatial tuning curves using pynapple.

    Mirrors the approach in utils/predictiveNet.py and the paper (Methods p.18):
    h_i(x) = E[h_i | x] computed via pynapple.compute_2d_tuning_curves_continuous.

    Parameters
    ----------
    hidden    : (T, H) float32 — hidden unit activations
    positions : (T, 2) float32 — (col, row) agent positions
    nb_bins   : spatial bin resolution per axis

    Returns
    -------
    tc  : dict {unit_id: (nb_bins, nb_bins) ndarray}  — tuning curves
    occ : (nb_bins, nb_bins) ndarray                  — occupancy map
    """
    T = hidden.shape[0]
    ts = np.arange(T, dtype=np.float64)

    ep = nap.IntervalSet(start=0.0, end=float(T - 1))

    h_tsd = nap.TsdFrame(t=ts, d=hidden.astype(np.float64),
                          time_support=ep)
    p_tsd = nap.TsdFrame(t=ts, d=positions.astype(np.float64),
                          columns=['x', 'y'], time_support=ep)

    tc = nap.compute_2d_tuning_curves_continuous(
        tsdframe=h_tsd,
        features=p_tsd,
        nb_bins=nb_bins,
        ep=ep,
    )

    # Occupancy: fraction of time in each bin
    x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max(), nb_bins + 1)
    y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max(), nb_bins + 1)
    occ, _, _ = np.histogram2d(positions[:, 0], positions[:, 1],
                                bins=[x_edges, y_edges])
    occ = occ / occ.sum()

    return tc, occ


# ─────────────────────────────────────────────────────────────────────────────
# Paper metric 3 — Spatial information (Skaggs et al. 1993)
# ─────────────────────────────────────────────────────────────────────────────

def spatial_information(tc: dict, occ: np.ndarray) -> np.ndarray:
    """
    Skaggs spatial information per unit (bits per spike / per activation).

    I_i = Σ_x p(x) * (h_i(x) / h̄_i) * log2(h_i(x) / h̄_i)

    Parameters
    ----------
    tc  : tuning curves dict from compute_tuning_curves
    occ : occupancy map (nb_bins, nb_bins), sums to 1

    Returns
    -------
    SI : (H,) array of spatial information values per unit
    """
    occ_flat = occ.ravel()
    valid = occ_flat > 0

    unit_ids = sorted(tc.keys())
    SI = np.zeros(len(unit_ids), dtype=np.float64)

    for i, uid in enumerate(unit_ids):
        tc_flat = tc[uid].ravel()
        mean_rate = np.sum(tc_flat * occ_flat)
        if mean_rate <= 0:
            continue
        ratio = np.zeros_like(tc_flat)
        ratio[valid] = tc_flat[valid] / mean_rate
        log_ratio = np.zeros_like(ratio)
        log_ratio[ratio > 0] = np.log2(ratio[ratio > 0])
        SI[i] = float(np.sum(occ_flat * ratio * log_ratio))

    return SI


# ─────────────────────────────────────────────────────────────────────────────
# Paper metric 4 — Spatial explained variance (%EVS)
# ─────────────────────────────────────────────────────────────────────────────

def spatial_evs(
    hidden: np.ndarray,
    positions: np.ndarray,
    tc: dict,
    nb_bins: int = 20,
) -> np.ndarray:
    """
    Spatial explained variance per unit (Methods p.18):
    EVS_i = 1 - Var[h_i - h_i(x_t)] / Var[h_i]

    Parameters
    ----------
    hidden    : (T, H)
    positions : (T, 2)
    tc        : tuning curves dict from compute_tuning_curves

    Returns
    -------
    evs : (H,) array of EVS values in [0, 1]
    """
    T, H = hidden.shape
    unit_ids = sorted(tc.keys())

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_edges = np.linspace(x_min, x_max, nb_bins + 1)
    y_edges = np.linspace(y_min, y_max, nb_bins + 1)

    # Bin each timepoint
    xi = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, nb_bins - 1)
    yi = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, nb_bins - 1)

    evs = np.zeros(len(unit_ids), dtype=np.float64)
    for i, uid in enumerate(unit_ids):
        tc_arr = tc[uid]                          # (nb_bins, nb_bins)
        expected = tc_arr[xi, yi]                 # (T,) — expected rate at each pos
        h_i = hidden[:, i].astype(np.float64)
        var_total = np.var(h_i)
        if var_total <= 0:
            continue
        residual = h_i - expected
        evs[i] = 1.0 - np.var(residual) / var_total

    return evs


# ─────────────────────────────────────────────────────────────────────────────
# New metric 1 — Symmetry Collapse Index (SCI)
# ─────────────────────────────────────────────────────────────────────────────

def sci(
    hidden: np.ndarray,
    positions: np.ndarray,
    symmetry_pairs: list,
    neural_metric: str = 'cosine',
    n_random_pairs: int = 10000,
) -> float:
    """
    SCI = mean_d_neural(sym-related pairs) / mean_d_neural(random pairs)

    SCI ≈ 1 → fully disambiguated; SCI → 0 → symmetry-related positions collapsed.

    symmetry_pairs : list of ((c1,r1),(c2,r2)) from SymmetryArena.precompute_symmetry_pairs()
    """
    if not symmetry_pairs:
        return float('nan')

    pos_arr = positions.astype(np.float32)
    pair_a  = np.array([list(a) for a, b in symmetry_pairs], dtype=np.float32)
    pair_b  = np.array([list(b) for a, b in symmetry_pairs], dtype=np.float32)

    idx_a = cdist(pair_a, pos_arr, 'cityblock').argmin(axis=1)
    idx_b = cdist(pair_b, pos_arr, 'cityblock').argmin(axis=1)

    h_a, h_b = hidden[idx_a], hidden[idx_b]
    if neural_metric == 'cosine':
        na = np.linalg.norm(h_a, axis=1, keepdims=True) + 1e-8
        nb = np.linalg.norm(h_b, axis=1, keepdims=True) + 1e-8
        sym_dists = 1.0 - np.sum((h_a / na) * (h_b / nb), axis=1)
    else:
        sym_dists = np.abs(h_a - h_b).mean(axis=1)
    mean_sym = float(sym_dists.mean())

    rng = np.random.default_rng(0)
    n   = hidden.shape[0]
    ri  = rng.integers(0, n, size=n_random_pairs)
    rj  = rng.integers(0, n, size=n_random_pairs)
    mask = ri != rj
    ri, rj = ri[mask], rj[mask]
    if neural_metric == 'cosine':
        ha, hb = hidden[ri], hidden[rj]
        na = np.linalg.norm(ha, axis=1, keepdims=True) + 1e-8
        nb = np.linalg.norm(hb, axis=1, keepdims=True) + 1e-8
        rand_dists = 1.0 - np.sum((ha / na) * (hb / nb), axis=1)
    else:
        rand_dists = np.abs(hidden[ri] - hidden[rj]).mean(axis=1)

    return mean_sym / (float(rand_dists.mean()) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# New metric 2 — Topology-Geometry Gap (ΔTG)
# ─────────────────────────────────────────────────────────────────────────────

def dtg_curve(srsa_euclid: list, srsa_city: list) -> np.ndarray:
    """ΔTG(t) = sRSA_Euclidean(t) − sRSA_CityBlock(t), one value per log step."""
    return np.array(srsa_euclid, dtype=np.float64) - np.array(srsa_city, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# New metric 3 — Manifold intrinsic dimensionality (TwoNN)
# ─────────────────────────────────────────────────────────────────────────────

def manifold_id(hidden: np.ndarray, max_n: int = 4000) -> float:
    """
    TwoNN intrinsic dimensionality estimator (Facco et al. 2017).
    ID = 1 / mean(log(r2/r1)), r1/r2 = 1st/2nd NN distances.
    """
    h = hidden[:max_n] if hidden.shape[0] > max_n else hidden
    dists = cdist(h, h, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    top2  = np.partition(dists, kth=2, axis=1)[:, :2]
    r1, r2 = top2[:, 0], top2[:, 1]
    valid  = (r1 > 0) & (r2 > 0)
    if valid.sum() < 10:
        return float('nan')
    return float(1.0 / np.mean(np.log(r2[valid] / r1[valid])))
