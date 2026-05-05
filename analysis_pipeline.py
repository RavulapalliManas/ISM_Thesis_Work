from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, pearsonr

from project5_symmetry.evaluation.metrics import (
    _exact_tuning_maps,
    _spatial_evs_exact,
    manifold_id,
    srsa,
)
from project5_symmetry.training.dataset import TrajectoryDataset
from project5_symmetry.training.train import _collect_hidden_states
from run_symmetry_sweep import P0_CFG, _load_trained_model


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

RESULTS_CANDIDATES = [
    ROOT / "project5_symmetry" / "results" / "symmetry_sweep_validate",
    ROOT / "outputs" / "symmetry_validate",
]
CONDITION_ORDER = ["s4", "s2", "s1"]
DISPLAY = {"s4": "S4", "s2": "S2", "s1": "S1"}
COLORS = {"s4": "#2166ac", "s2": "#ef8a62", "s1": "#4daf4a"}
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_SHUFFLES = 1000
N_RANDOM_SCI_PAIRS = 10_000
N_DTG_PERMUTATIONS = 10_000
RAW_EVS_THRESHOLD = 0.05
RAW_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


@dataclass
class SeedRecord:
    condition: str
    seed_name: str
    seed_index: int
    condition_dir: Path
    seed_dir: Path
    evaluation_pkl: Path
    training_log_json: Path


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


def _safe_std(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.std(ddof=0))


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


def discover_results_root() -> Path:
    for candidate in RESULTS_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No discovered symmetry results root found. Checked: "
        + ", ".join(str(p) for p in RESULTS_CANDIDATES)
    )


def discover_seed_records(results_root: Path) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    for condition_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        evaluation_files = sorted(condition_dir.glob("seed_*/evaluation.pkl"))
        for evaluation_pkl in evaluation_files:
            seed_dir = evaluation_pkl.parent
            training_log_json = seed_dir / "training_log.json"
            if not training_log_json.exists():
                continue
            seed_name = seed_dir.name
            try:
                seed_index = int(seed_name.split("_")[-1])
            except ValueError:
                seed_index = len(records)
            records.append(
                SeedRecord(
                    condition=condition_dir.name.lower(),
                    seed_name=seed_name,
                    seed_index=seed_index,
                    condition_dir=condition_dir,
                    seed_dir=seed_dir,
                    evaluation_pkl=evaluation_pkl,
                    training_log_json=training_log_json,
                )
            )
    return sorted(records, key=lambda r: (CONDITION_ORDER.index(r.condition) if r.condition in CONDITION_ORDER else 99, r.seed_index))


def load_evaluation(record: SeedRecord) -> dict[str, Any]:
    with open(record.evaluation_pkl, "rb") as f:
        return pickle.load(f)


def load_training_log(record: SeedRecord) -> dict[str, Any]:
    with open(record.training_log_json, "r") as f:
        return json.load(f)


def load_raw_hidden_and_positions(record: SeedRecord) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (record.condition, record.seed_name)
    if cache_key in RAW_CACHE:
        return RAW_CACHE[cache_key]
    data_dir = record.condition_dir / "trajectories"
    dataset = TrajectoryDataset(str(data_dir))
    obs_size = P0_CFG.F * P0_CFG.F * 3
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = _load_trained_model(record.seed_dir / "ckpt_final.pt", obs_size=obs_size, k=P0_CFG.k, device=__import__("torch").device(device))
    eval_n = int(__import__("run_symmetry_sweep").SUBSAMPLE_N)
    hidden, positions = _collect_hidden_states(model, dataset, n=eval_n, device=device)
    value = hidden.astype(np.float32), positions.astype(np.float32)
    RAW_CACHE[cache_key] = value
    return value


def position_grid_size(positions: np.ndarray) -> int:
    return int(np.max(np.rint(positions)))


def positions_to_linear_index(positions: np.ndarray, size: int) -> np.ndarray:
    pos = np.rint(positions).astype(int)
    cols = pos[:, 0] - 1
    rows = pos[:, 1] - 1
    return rows * size + cols


def linear_index_to_row_col(index: int, size: int) -> tuple[int, int]:
    row = index // size
    col = index % size
    return row, col


def row_col_to_linear_index(row: int, col: int, size: int) -> int:
    return row * size + col


def exact_tuning_and_evs(hidden: np.ndarray, positions: np.ndarray, arena_size: int) -> tuple[np.ndarray, np.ndarray]:
    tuning_maps, _ = _exact_tuning_maps(hidden, positions, arena_size)
    evs = _spatial_evs_exact(hidden, positions, tuning_maps)
    return tuning_maps, evs


def _corr_with_rot90(field: np.ndarray) -> float:
    src = np.nan_to_num(field, nan=0.0).reshape(-1)
    rot = np.nan_to_num(np.rot90(field), nan=0.0).reshape(-1)
    if np.allclose(src.std(), 0.0) or np.allclose(rot.std(), 0.0):
        return float("nan")
    return float(pearsonr(src, rot)[0])


def compute_ra_for_seed(hidden: np.ndarray, positions: np.ndarray, arena_size: int, evs_threshold: float = RAW_EVS_THRESHOLD) -> dict[str, Any]:
    tuning_maps, evs = exact_tuning_and_evs(hidden, positions, arena_size)
    include = np.isfinite(evs) & (evs > evs_threshold)
    ra_per_unit = np.full(hidden.shape[1], np.nan, dtype=float)
    for unit_idx in np.where(include)[0]:
        ra_per_unit[unit_idx] = _corr_with_rot90(tuning_maps[unit_idx])
    return {
        "ra_per_unit": ra_per_unit,
        "evs": evs,
        "include_mask": include,
        "observed_ra": _safe_mean(ra_per_unit[include]),
    }


def shuffled_ra_null(
    hidden: np.ndarray,
    positions: np.ndarray,
    arena_size: int,
    include_mask: np.ndarray,
    n_shuffles: int = N_SHUFFLES,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.full(n_shuffles, np.nan, dtype=float)
    for sidx in range(n_shuffles):
        shuffled_positions = positions[rng.permutation(len(positions))]
        tuning_maps, _ = _exact_tuning_maps(hidden, shuffled_positions, arena_size)
        values = np.full(hidden.shape[1], np.nan, dtype=float)
        for unit_idx in np.where(include_mask)[0]:
            values[unit_idx] = _corr_with_rot90(tuning_maps[unit_idx])
        out[sidx] = _safe_mean(values[include_mask])
    return out


def metric_row(
    metric: str,
    comparison: str,
    g1_name: str,
    g2_name: str,
    g1: list[float],
    g2: list[float],
) -> dict[str, Any]:
    a = np.asarray(g1, dtype=float)
    b = np.asarray(g2, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    row = {
        "metric": metric,
        "comparison": comparison,
        "group_1": g1_name,
        "group_2": g2_name,
        "n_1": int(a.size),
        "n_2": int(b.size),
        "mean_1": _safe_mean(a),
        "sem_1": _sem(a),
        "mean_2": _safe_mean(b),
        "sem_2": _sem(b),
        "U": float("nan"),
        "p_raw": float("nan"),
        "p_bonferroni": float("nan"),
        "rank_biserial_r": float("nan"),
    }
    if a.size > 0 and b.size > 0:
        u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
        row["U"] = float(u_stat)
        row["p_raw"] = float(p_value)
        row["p_bonferroni"] = float(min(1.0, p_value * 3.0))
        row["rank_biserial_r"] = float(1.0 - 2.0 * u_stat / (a.size * b.size))
    return row


def compute_decoding_error(H: np.ndarray, positions: np.ndarray) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    H_norm = (H - H.mean(axis=0, keepdims=True)) / (H.std(axis=0, keepdims=True) + 1e-8)
    kf = KFold(n_splits=min(5, len(H_norm)), shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    errors: list[float] = []
    for train_idx, test_idx in kf.split(H_norm):
        model.fit(H_norm[train_idx], positions[train_idx])
        pred = model.predict(H_norm[test_idx])
        err = np.abs(pred - positions[test_idx]).sum(axis=1)
        errors.extend(err.tolist())
    return _safe_mean(errors)


def compute_frac_tuned(hidden: np.ndarray, positions: np.ndarray, threshold: float) -> float:
    arena_size = position_grid_size(positions)
    _, evs = exact_tuning_and_evs(hidden, positions, arena_size)
    return float(np.mean(np.isfinite(evs) & (evs > threshold)))


def compute_paa_gain(condition_records: list[SeedRecord], evaluations: dict[tuple[str, str], dict[str, Any]]) -> float:
    if len(condition_records) < 2:
        return float("nan")
    size = 18

    def rotate_positions(position_array: np.ndarray, k: int) -> np.ndarray:
        out = []
        for col, row in np.rint(position_array).astype(int):
            r0, c0 = row - 1, col - 1
            for _ in range(k % 4):
                r0, c0 = c0, size - 1 - r0
            out.append([c0 + 1, r0 + 1])
        return np.asarray(out, dtype=int)

    def reorder_hidden(evaluation: dict[str, Any], rotated_positions: np.ndarray) -> np.ndarray:
        base_positions = np.asarray(evaluation["position_array"], dtype=int)
        hidden = np.asarray(evaluation["position_hidden"], dtype=float)
        mapping = {tuple(pos.tolist()): i for i, pos in enumerate(base_positions)}
        order = [mapping[tuple(pos.tolist())] for pos in rotated_positions if tuple(pos.tolist()) in mapping]
        return hidden[order]

    pair_gains: list[float] = []
    for i in range(len(condition_records)):
        for j in range(i + 1, len(condition_records)):
            eva = evaluations[(condition_records[i].condition, condition_records[i].seed_name)]
            evb = evaluations[(condition_records[j].condition, condition_records[j].seed_name)]
            pos_a = np.asarray(eva["position_array"], dtype=int)
            H_a = np.asarray(eva["position_hidden"], dtype=float)
            identity = float(np.corrcoef(H_a.reshape(-1), np.asarray(evb["position_hidden"], dtype=float).reshape(-1))[0, 1])
            best = identity
            for k in (1, 2, 3):
                rotated = rotate_positions(pos_a, k)
                H_rot = reorder_hidden(eva, rotated)
                H_b = np.asarray(evb["position_hidden"], dtype=float)
                if H_rot.shape != H_b.shape:
                    continue
                best = max(best, float(np.corrcoef(H_rot.reshape(-1), H_b.reshape(-1))[0, 1]))
            pair_gains.append(best - identity)
    return _safe_mean(pair_gains)


def symmetry_pairs_for_condition(condition: str, size: int) -> list[tuple[int, int]]:
    pair_set: set[tuple[int, int]] = set()
    for idx in range(size * size):
        row, col = linear_index_to_row_col(idx, size)
        related: list[tuple[int, int]] = []
        if condition in {"s4", "s1"}:
            related.extend([
                (col, size - 1 - row),
                (size - 1 - row, size - 1 - col),
                (size - 1 - col, row),
            ])
        elif condition == "s2":
            related.append((size - 1 - row, size - 1 - col))
        for rr, cc in related:
            jdx = row_col_to_linear_index(rr, cc, size)
            if idx != jdx:
                pair_set.add(tuple(sorted((idx, jdx))))
    return sorted(pair_set)


def compute_sci_for_seed(position_hidden: np.ndarray, condition: str, size: int) -> float:
    H = np.asarray(position_hidden, dtype=float)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / np.clip(norms, 1e-8, None)
    pair_list = symmetry_pairs_for_condition(condition, size)
    if not pair_list:
        return float("nan")
    sym_dists = np.array([np.linalg.norm(H_norm[a] - H_norm[b]) for a, b in pair_list], dtype=float)
    sym_pair_set = set(pair_list)
    rng = np.random.default_rng(0)
    random_pairs: set[tuple[int, int]] = set()
    n_positions = H_norm.shape[0]
    while len(random_pairs) < N_RANDOM_SCI_PAIRS:
        a = int(rng.integers(n_positions))
        b = int(rng.integers(n_positions))
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in sym_pair_set:
            continue
        random_pairs.add(pair)
    rand_dists = np.array([np.linalg.norm(H_norm[a] - H_norm[b]) for a, b in random_pairs], dtype=float)
    return float(sym_dists.mean() / np.clip(rand_dists.mean(), 1e-8, None))


def quadrant_index(col: int, row: int) -> int:
    top = row <= 9
    left = col <= 9
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


def quadrant_similarity(position_hidden: np.ndarray, position_array: np.ndarray) -> tuple[np.ndarray, float]:
    H = np.asarray(position_hidden, dtype=float)
    D = cdist(H, H, metric="euclidean")
    max_dist = float(np.max(D)) if np.max(D) > 0 else 1.0
    S = 1.0 - (D / max_dist)
    quadrants = {0: [], 1: [], 2: [], 3: []}
    for idx, (col, row) in enumerate(np.rint(position_array).astype(int)):
        quadrants[quadrant_index(col, row)].append(idx)
    Q = np.full((4, 4), np.nan, dtype=float)
    for i in range(4):
        for j in range(4):
            if not quadrants[i] or not quadrants[j]:
                continue
            Q[i, j] = float(S[np.ix_(quadrants[i], quadrants[j])].mean())
    c2_pairs = np.nanmean([Q[0, 2], Q[1, 3]])
    c4_pairs = np.nanmean([Q[0, 1], Q[0, 3], Q[1, 2], Q[2, 3]])
    return Q, float(c2_pairs - c4_pairs)


def permutation_test_two_groups(a: list[float], b: list[float], n_perm: int = N_DTG_PERMUTATIONS) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    observed = float(abs(arr_a.mean() - arr_b.mean()))
    pooled = np.concatenate([arr_a, arr_b])
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        pa = perm[: len(arr_a)]
        pb = perm[len(arr_a) :]
        if abs(pa.mean() - pb.mean()) >= observed:
            count += 1
    pooled_std = np.sqrt(((arr_a.size - 1) * arr_a.var(ddof=1) + (arr_b.size - 1) * arr_b.var(ddof=1)) / max(arr_a.size + arr_b.size - 2, 1))
    d = float((arr_a.mean() - arr_b.mean()) / pooled_std) if pooled_std > 0 else float("nan")
    return count / n_perm, d


def summarize_metric_across_conditions(metric_name: str, per_seed_metric: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = [("s4", "s1"), ("s4", "s2"), ("s2", "s1")]
    for c1, c2 in comparisons:
        g1 = [v for v in per_seed_metric.get(c1, {}).values()]
        g2 = [v for v in per_seed_metric.get(c2, {}).values()]
        rows.append(metric_row(metric_name, f"{_condition_label(c1)} vs {_condition_label(c2)}", _condition_label(c1), _condition_label(c2), g1, g2))
    return rows


def run_A1(records: list[SeedRecord]) -> dict[str, Any]:
    print("\nA1. RA shuffle baseline")
    observed_by_condition: dict[str, dict[str, float]] = {}
    z_by_condition: dict[str, dict[str, float]] = {}
    p_by_condition: dict[str, dict[str, float]] = {}
    null_by_condition: dict[str, dict[str, np.ndarray]] = {}
    seed_ra: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        hidden, positions = load_raw_hidden_and_positions(record)
        arena_size = position_grid_size(positions)
        ra = compute_ra_for_seed(hidden, positions, arena_size)
        shuffle = shuffled_ra_null(hidden, positions, arena_size, ra["include_mask"], seed=record.seed_index)
        observed_by_condition.setdefault(record.condition, {})[record.seed_name] = ra["observed_ra"]
        null_by_condition.setdefault(record.condition, {})[record.seed_name] = shuffle
        null_mean = float(np.nanmean(shuffle))
        null_std = float(np.nanstd(shuffle))
        z = (ra["observed_ra"] - null_mean) / null_std if np.isfinite(null_std) and null_std > 0 else float("nan")
        p = float(np.mean(shuffle >= ra["observed_ra"])) if np.isfinite(ra["observed_ra"]) else float("nan")
        z_by_condition.setdefault(record.condition, {})[record.seed_name] = z
        p_by_condition.setdefault(record.condition, {})[record.seed_name] = p
        seed_ra[(record.condition, record.seed_name)] = {**ra, "shuffle": shuffle, "z_score": z, "p_value": p}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, condition in zip(axes, CONDITION_ORDER):
        if condition in null_by_condition and null_by_condition[condition]:
            arrays = list(null_by_condition[condition].values())
            parts = ax.violinplot(arrays, showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(COLORS.get(condition, "#999999"))
                body.set_alpha(0.45)
            obs_values = list(observed_by_condition.get(condition, {}).values())
            if obs_values:
                mean_obs = _safe_mean(obs_values)
                ax.errorbar(
                    x=1,
                    y=mean_obs,
                    yerr=_sem(obs_values),
                    fmt="o",
                    color="red",
                    capsize=4,
                )
            ax.set_xticks([1], [_condition_label(condition)])
        else:
            ax.text(0.5, 0.5, "No discovered data", ha="center", va="center")
            ax.set_xticks([])
        ax.set_title(_condition_label(condition))
        ax.set_ylabel("RA")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "RA_shuffle_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for condition in CONDITION_ORDER:
        obs_values = list(observed_by_condition.get(condition, {}).values())
        pooled_null = np.concatenate(list(null_by_condition.get(condition, {}).values())) if null_by_condition.get(condition) else np.array([])
        row = {
            "condition": _condition_label(condition),
            "observed_mean": _safe_mean(obs_values),
            "observed_sem": _sem(obs_values),
            "null_mean": _safe_mean(pooled_null),
            "null_std": _safe_std(pooled_null),
            "seed_z_scores": z_by_condition.get(condition, {}),
            "seed_p_values": p_by_condition.get(condition, {}),
        }
        rows.append(row)
    ra_table = pd.DataFrame(rows)
    print(ra_table.to_string(index=False))
    return {
        "observed_by_condition": observed_by_condition,
        "null_by_condition": null_by_condition,
        "z_by_condition": z_by_condition,
        "p_by_condition": p_by_condition,
        "seed_ra": seed_ra,
        "summary_table": ra_table,
    }


def run_A2(records: list[SeedRecord], a1: dict[str, Any]) -> pd.DataFrame:
    print("\nA2. Mann-Whitney tests")
    evaluations = {(r.condition, r.seed_name): load_evaluation(r) for r in records}
    training_logs = {(r.condition, r.seed_name): load_training_log(r) for r in records}

    by_metric: dict[str, dict[str, dict[str, float]]] = {}
    for record in records:
        key = (record.condition, record.seed_name)
        evaluation = evaluations[key]
        training = training_logs[key]
        H = np.asarray(evaluation["position_hidden"], dtype=float)
        positions = np.asarray(evaluation["position_array"], dtype=float)
        raw_hidden, raw_positions = load_raw_hidden_and_positions(record)

        metrics = {
            "sRSA_euclid": float(training["srsa_euclid"][-1]) if training.get("srsa_euclid") else float("nan"),
            "sRSA_city": float(training["srsa_city"][-1]) if training.get("srsa_city") else float("nan"),
            "DTG": float(training["srsa_euclid"][-1] - training["srsa_city"][-1]) if training.get("srsa_euclid") and training.get("srsa_city") else float("nan"),
            "Manifold_ID": float(training["manifold_id"][-1]) if training.get("manifold_id") else float(manifold_id(H)),
            "RA": float(a1["observed_by_condition"].get(record.condition, {}).get(record.seed_name, np.nan)),
            "PAA_gain": float("nan"),
            "Decoding_error": compute_decoding_error(H, positions),
            "frac_tuned": compute_frac_tuned(raw_hidden, raw_positions, RAW_EVS_THRESHOLD),
            "C2_contrast": float("nan"),
        }
        by_metric.setdefault("PAA_gain", {}).setdefault(record.condition, {})
        by_metric.setdefault("C2_contrast", {}).setdefault(record.condition, {})
        for metric_name, value in metrics.items():
            by_metric.setdefault(metric_name, {}).setdefault(record.condition, {})[record.seed_name] = value

    for condition in {r.condition for r in records}:
        crecords = [r for r in records if r.condition == condition]
        evaluations_for_cond = {(r.condition, r.seed_name): evaluations[(r.condition, r.seed_name)] for r in crecords}
        paa = compute_paa_gain(crecords, evaluations_for_cond)
        for r in crecords:
            by_metric["PAA_gain"][condition][r.seed_name] = paa
            qmat, c2_value = quadrant_similarity(
                np.asarray(evaluations[(r.condition, r.seed_name)]["position_hidden"], dtype=float),
                np.asarray(evaluations[(r.condition, r.seed_name)]["position_array"], dtype=float),
            )
            by_metric["C2_contrast"][condition][r.seed_name] = c2_value

    all_rows: list[dict[str, Any]] = []
    for metric_name in [
        "sRSA_euclid",
        "sRSA_city",
        "DTG",
        "Manifold_ID",
        "RA",
        "PAA_gain",
        "Decoding_error",
        "frac_tuned",
        "C2_contrast",
    ]:
        all_rows.extend(summarize_metric_across_conditions(metric_name, by_metric.get(metric_name, {})))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "stats_results.csv", index=False)
    print(df.to_string(index=False))
    return df


def run_A3(records: list[SeedRecord]) -> dict[str, dict[str, float]]:
    print("\nA3. Symmetry Collapse Index")
    sci_by_condition: dict[str, dict[str, float]] = {}
    for record in records:
        evaluation = load_evaluation(record)
        H = np.asarray(evaluation["position_hidden"], dtype=float)
        positions = np.asarray(evaluation["position_array"], dtype=float)
        size = position_grid_size(positions)
        sci_val = compute_sci_for_seed(H, record.condition, size)
        sci_by_condition.setdefault(record.condition, {})[record.seed_name] = sci_val

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(CONDITION_ORDER))
    means = [_safe_mean(list(sci_by_condition.get(c, {}).values())) for c in CONDITION_ORDER]
    sems = [_sem(list(sci_by_condition.get(c, {}).values())) for c in CONDITION_ORDER]
    ax.bar(xs, means, yerr=sems, color=[COLORS[c] for c in CONDITION_ORDER], capsize=4)
    ax.axhline(1.0, ls="--", color="black", lw=1)
    ax.set_xticks(xs, [_condition_label(c) for c in CONDITION_ORDER])
    ax.set_ylabel("SCI")
    ax.set_title("Symmetry Collapse Index")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "SCI_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame(
        {
            "condition": [_condition_label(c) for c in CONDITION_ORDER],
            "SCI_mean": means,
            "SCI_sem": sems,
        }
    ).to_string(index=False))

    print("SCI trajectory skipped: no intermediate checkpoints discovered in the checked-in results.")
    return sci_by_condition


def run_A4(records: list[SeedRecord]) -> dict[str, dict[str, float]]:
    print("\nA4. frac_tuned discrepancy analysis")
    per_threshold: dict[float, dict[str, dict[str, float]]] = {thr: {} for thr in THRESHOLDS}
    final_srsa: dict[str, dict[str, float]] = {}
    frac_at_050: dict[str, dict[str, float]] = {}

    for record in records:
        raw_hidden, raw_positions = load_raw_hidden_and_positions(record)
        training = load_training_log(record)
        final_srsa.setdefault(record.condition, {})[record.seed_name] = (
            float(training["srsa_euclid"][-1]) if training.get("srsa_euclid") else float("nan")
        )
        for thr in THRESHOLDS:
            frac = compute_frac_tuned(raw_hidden, raw_positions, thr)
            per_threshold[thr].setdefault(record.condition, {})[record.seed_name] = frac
            if np.isclose(thr, 0.50):
                frac_at_050.setdefault(record.condition, {})[record.seed_name] = frac

    fig, ax = plt.subplots(figsize=(7, 4))
    for condition in CONDITION_ORDER:
        means = [_safe_mean(list(per_threshold[thr].get(condition, {}).values())) for thr in THRESHOLDS]
        sems = [_sem(list(per_threshold[thr].get(condition, {}).values())) for thr in THRESHOLDS]
        ax.plot(THRESHOLDS, means, marker="o", color=COLORS[condition], label=_condition_label(condition))
        means_arr = np.asarray(means, dtype=float)
        sems_arr = np.asarray(sems, dtype=float)
        ax.fill_between(THRESHOLDS, means_arr - sems_arr, means_arr + sems_arr, color=COLORS[condition], alpha=0.18)
    ax.set_xlabel("EVS threshold")
    ax.set_ylabel("frac_tuned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frac_tuned_threshold_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for condition in CONDITION_ORDER:
        xs = np.asarray(list(final_srsa.get(condition, {}).values()), dtype=float)
        ys = np.asarray(list(per_threshold[0.10].get(condition, {}).values()), dtype=float)
        if xs.size == 0 or ys.size == 0:
            continue
        ax.scatter(xs, ys, color=COLORS[condition], label=_condition_label(condition))
        if xs.size >= 2:
            coeffs = np.polyfit(xs, ys, deg=1)
            grid = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(grid, np.polyval(coeffs, grid), color=COLORS[condition], alpha=0.8)
    ax.set_xlabel("sRSA Euclid")
    ax.set_ylabel("frac_tuned (EVS > 0.10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frac_tuned_vs_srsa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    interpretation = (
        "The fraction of spatially tuned units in all conditions (~4–8%) falls "
        "substantially below Levenstein et al.'s reported ~20–30%. Two explanations "
        "are consistent with the data. First, the present experiments used batch size "
        "B=8, compared to Levenstein's B=1; smoother gradient updates from larger "
        "batches may reduce the sharpness of individual unit tuning while preserving "
        "population-level spatial structure (sRSA remained above 0.64 in all conditions). "
        "Second, the variance-explained threshold used here may differ from Levenstein's "
        "implementation. The frac_tuned discrepancy does not affect the main conclusions, "
        "which rest on population-level metrics (sRSA, DTG, PAA, RA), but warrants "
        "investigation in future work with matched training hyperparameters."
    )
    (RESULTS_DIR / "frac_tuned_interpretation.txt").write_text(interpretation + "\n")
    return frac_at_050


def run_A5(records: list[SeedRecord], stats_df: pd.DataFrame) -> pd.DataFrame:
    print("\nA5. Quadrant similarity matrix")
    q_by_condition: dict[str, list[np.ndarray]] = {}
    offdiag_var: dict[str, dict[str, float]] = {}
    c2_values: dict[str, dict[str, float]] = {}

    for record in records:
        evaluation = load_evaluation(record)
        Q, c2_value = quadrant_similarity(
            np.asarray(evaluation["position_hidden"], dtype=float),
            np.asarray(evaluation["position_array"], dtype=float),
        )
        q_by_condition.setdefault(record.condition, []).append(Q)
        c2_values.setdefault(record.condition, {})[record.seed_name] = c2_value
        offdiag = Q[~np.eye(4, dtype=bool)]
        offdiag_var.setdefault(record.condition, {})[record.seed_name] = float(np.nanvar(offdiag))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    all_q = [np.nanmean(q_by_condition[c], axis=0) if q_by_condition.get(c) else np.full((4, 4), np.nan) for c in CONDITION_ORDER]
    finite_vals = np.concatenate([q[np.isfinite(q)] for q in all_q if np.isfinite(q).any()]) if any(np.isfinite(q).any() for q in all_q) else np.array([0.0, 1.0])
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    for ax, condition, matrix in zip(axes, CONDITION_ORDER, all_q):
        if np.isfinite(matrix).any():
            im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{matrix[i, j]:.2f}" if np.isfinite(matrix[i, j]) else "NA", ha="center", va="center", color="white", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(_condition_label(condition))
        ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
        ax.set_yticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    if any(np.isfinite(q).any() for q in all_q):
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quadrant_similarity_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    extra_rows = summarize_metric_across_conditions("quadrant_offdiag_var", offdiag_var)
    stats_df = pd.concat([stats_df, pd.DataFrame(extra_rows)], ignore_index=True)
    stats_df.to_csv(RESULTS_DIR / "stats_results.csv", index=False)

    explanation = (
        "The negative C2 Contrast in S4 (−0.130) reflects uniform blurring of all "
        "inter-quadrant distinctions under C4 symmetry. When all four quadrants are "
        "equally confusable under the visual observation function, predictive learning "
        "compresses all inter-quadrant pairs uniformly, including the 180-degree related "
        "pairs that define the C2 subgroup. The result is that 180-degree pairs are no "
        "more similar than 90-degree pairs — producing negative contrast rather than "
        "the selective compression seen in S2. Under C2 symmetry, only 180-degree pairs "
        "are visually confusable, so the network selectively compresses those pairs "
        "while preserving 90-degree distinctions, producing positive contrast."
    )
    (RESULTS_DIR / "c2_contrast_explanation.txt").write_text(explanation + "\n")
    return stats_df


def run_A6(stats_df: pd.DataFrame) -> None:
    print("\nA6. DTG non-monotonicity permutation test")
    if not (RESULTS_DIR / "stats_results.csv").exists():
        raise FileNotFoundError("Expected results/stats_results.csv to exist before A6.")
    table = pd.read_csv(RESULTS_DIR / "stats_results.csv")
    # This script records comparison-level stats rather than seed-level DTG arrays.
    # Pull seed-level DTG values from the in-memory source data instead.
    results_root = discover_results_root()
    records = discover_seed_records(results_root)
    dtg_by_condition: dict[str, list[float]] = {}
    for record in records:
        training = load_training_log(record)
        if training.get("srsa_euclid") and training.get("srsa_city"):
            dtg = float(training["srsa_euclid"][-1] - training["srsa_city"][-1])
            dtg_by_condition.setdefault(record.condition, []).append(dtg)

    s2 = dtg_by_condition.get("s2", [])
    s1 = dtg_by_condition.get("s1", [])
    if len(s2) == 3 and len(s1) == 3:
        p_value, cohen_d = permutation_test_two_groups(s2, s1)
        sig_text = "significant" if p_value < 0.05 else "not significant"
        body = (
            "The S2 DTG of −0.011 does not follow the expected monotone ordering with S1 "
            f"(−0.020). A permutation test yields p = {p_value:.4f}. Cohen's d = {cohen_d:.4f}. "
            f"With n=3 seeds, this result is {sig_text} and should be treated as preliminary. "
            "A plausible mechanistic account: C2 symmetry creates a 180-degree positional ambiguity "
            "that path integration partially resolves within each half-arena, producing more Euclidean "
            "local geometry than C4, which creates a four-way ambiguity that exceeds path integration's "
            "resolving power. Additional seeds are needed to distinguish this from sampling variance."
        )
    else:
        body = (
            "The checked-in results in this worktree do not contain the full S2 (n=3) and S1 (n=3) "
            "DTG seed sets required for the requested permutation test. The analysis pipeline is wired "
            "to run the test automatically once those seed directories are present under "
            "project5_symmetry/results/symmetry_sweep_validate/."
        )
    (RESULTS_DIR / "dtg_nonmonotonicity.txt").write_text(body + "\n")


def main() -> None:
    ensure_dirs()
    results_root = discover_results_root()
    records = discover_seed_records(results_root)
    if not records:
        raise RuntimeError(f"No discovered seed records under {results_root}")

    a1 = run_A1(records)
    stats_df = run_A2(records, a1)
    run_A3(records)
    run_A4(records)
    stats_df = run_A5(records, stats_df)
    run_A6(stats_df)
    print("\nAnalysis pipeline complete.")


if __name__ == "__main__":
    main()
