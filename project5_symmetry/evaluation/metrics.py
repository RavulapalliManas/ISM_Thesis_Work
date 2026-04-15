import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr

MAX_SUBSAMPLE = 4000  # matches legacy analysis code cap


def _subsample(hidden: np.ndarray, positions: np.ndarray, n: int):
    if hidden.shape[0] > n:
        idx = np.random.choice(hidden.shape[0], n, replace=False)
        return hidden[idx], positions[idx]
    return hidden, positions


def srsa(
    hidden: np.ndarray,
    positions: np.ndarray,
    neural_metric: str = 'cosine',
    space_metric: str = 'euclidean',
    max_n: int = MAX_SUBSAMPLE,
) -> float:
    """
    Spatial Representational Similarity Analysis.

    Parameters
    ----------
    hidden    : (T, H) float array of hidden states
    positions : (T, 2) float array of (row, col) positions
    neural_metric : 'cosine' | 'cityblock'
    space_metric  : 'euclidean' | 'cityblock'

    Returns
    -------
    Spearman r between neural pairwise distances and spatial pairwise distances.
    """
    h, p = _subsample(hidden, positions, max_n)

    if neural_metric == 'cityblock':
        neural_dists = pdist(h, 'cityblock') / h.shape[1]
    else:
        neural_dists = pdist(h, neural_metric)

    spatial_dists = pdist(p, space_metric)
    return float(spearmanr(neural_dists, spatial_dists).statistic)


def sci(
    hidden: np.ndarray,
    positions: np.ndarray,
    symmetry_pairs: list,
    neural_metric: str = 'cosine',
    n_random_pairs: int = 10000,
) -> float:
    """
    Symmetry Collapse Index.

    SCI = mean(d_neural(x, T*x)) / mean(d_neural(x, x'))

    symmetry_pairs : list of ((r1,c1), (r2,c2)) from Arena.precompute_symmetry_pairs()
    Matches each pair to the nearest hidden state by position.

    Returns SCI in [0, 1]; SCI ≈ 1 → fully disambiguated, SCI → 0 → collapsed.
    """
    if not symmetry_pairs:
        return float('nan')

    pos_arr = np.array(positions, dtype=np.float32)  # (T, 2)

    # For each symmetry pair, find closest timestep to each position
    pair_a = np.array([list(a) for a, b in symmetry_pairs], dtype=np.float32)
    pair_b = np.array([list(b) for a, b in symmetry_pairs], dtype=np.float32)

    # Nearest-neighbour lookup: for each pair endpoint, find closest state
    dist_a = cdist(pair_a, pos_arr, metric='cityblock')  # (n_pairs, T)
    dist_b = cdist(pair_b, pos_arr, metric='cityblock')
    idx_a = dist_a.argmin(axis=1)
    idx_b = dist_b.argmin(axis=1)

    # Pairwise neural distances for symmetry-related positions
    h_a = hidden[idx_a]
    h_b = hidden[idx_b]
    if neural_metric == 'cosine':
        # cosine distance between corresponding rows
        norm_a = np.linalg.norm(h_a, axis=1, keepdims=True) + 1e-8
        norm_b = np.linalg.norm(h_b, axis=1, keepdims=True) + 1e-8
        sym_dists = 1.0 - np.sum((h_a / norm_a) * (h_b / norm_b), axis=1)
    else:
        sym_dists = np.abs(h_a - h_b).mean(axis=1)

    mean_sym = float(sym_dists.mean())

    # Random pair baseline
    rng = np.random.default_rng(0)
    n = hidden.shape[0]
    n_rand = min(n_random_pairs, n * (n - 1) // 2)
    ri = rng.integers(0, n, size=n_rand)
    rj = rng.integers(0, n, size=n_rand)
    mask = ri != rj
    ri, rj = ri[mask], rj[mask]
    if neural_metric == 'cosine':
        ha, hb = hidden[ri], hidden[rj]
        na = np.linalg.norm(ha, axis=1, keepdims=True) + 1e-8
        nb = np.linalg.norm(hb, axis=1, keepdims=True) + 1e-8
        rand_dists = 1.0 - np.sum((ha / na) * (hb / nb), axis=1)
    else:
        rand_dists = np.abs(hidden[ri] - hidden[rj]).mean(axis=1)

    mean_rand = float(rand_dists.mean()) + 1e-8

    return mean_sym / mean_rand


def dtg_curve(srsa_euclid: list, srsa_city: list) -> np.ndarray:
    """
    Topology-Geometry Gap: ΔTG(t) = sRSA_Euclidean(t) - sRSA_CityBlock(t).

    Both inputs are lists of floats (one per LOG_INTERVAL checkpoint).
    Returns numpy array of same length.
    """
    return np.array(srsa_euclid, dtype=np.float64) - np.array(srsa_city, dtype=np.float64)


def manifold_id(hidden: np.ndarray, max_n: int = 4000) -> float:
    """
    Intrinsic dimensionality via TwoNN estimator (Facco et al. 2017).

    ID = 1 / mean(log(r2 / r1))
    where r1, r2 are distances to 1st and 2nd nearest neighbours.

    Returns estimated ID (float).
    """
    h = hidden[:max_n] if hidden.shape[0] > max_n else hidden
    n = h.shape[0]

    # Pairwise distances — shape (n, n)
    dists = cdist(h, h, metric='euclidean')
    np.fill_diagonal(dists, np.inf)

    # 1st and 2nd NN distances
    sorted_dists = np.partition(dists, kth=2, axis=1)[:, :3]
    r1 = sorted_dists[:, 0]   # closest (after inf diagonal removed)
    r2 = sorted_dists[:, 1]   # second closest

    # Exclude degenerate rows
    valid = (r1 > 0) & (r2 > 0)
    ratio = r2[valid] / r1[valid]
    if valid.sum() < 10:
        return float('nan')

    return float(1.0 / np.mean(np.log(ratio)))
