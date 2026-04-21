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


# ─────────────────────────────────────────────────────────────────────────────
# Paper metric 1 — sRSA
# ─────────────────────────────────────────────────────────────────────────────

def srsa(
    hidden: np.ndarray,
    positions: np.ndarray,
    neural_metric: str = 'cosine',
    space_metric: str = 'euclidean',
    max_n: int = MAX_SUBSAMPLE,
) -> float:
    """
    Spatial RSA: Spearman r between cosine neural distances and Euclidean
    spatial distances across all pairs of timepoints (Methods p.19).

    hidden    : (T, H)
    positions : (T, 2)  col/row in MiniGrid coords
    """
    h, p = _subsample(hidden, positions, max_n)
    if neural_metric == 'cosine':
        neural_dists = _pdist_cosine(h)
    else:
        neural_dists = pdist(h, neural_metric)
    if space_metric == 'euclidean':
        spatial_dists = _pdist_euclidean(p)
    elif space_metric == 'cityblock':
        spatial_dists = _pdist_cityblock(p)
    else:
        spatial_dists = pdist(p, space_metric)
    return float(spearmanr(neural_dists, spatial_dists).statistic)


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
