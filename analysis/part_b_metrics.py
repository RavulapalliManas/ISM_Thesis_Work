import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
SWEEP = BASE / "project5_symmetry" / "results" / "symmetry_sweep"
METRICS_DIR = BASE / "results" / "metrics"
TABLES_DIR = BASE / "results" / "tables"
EVS_THRESHOLD = 0.1
CONDITIONS = {"s1": range(3), "s2": range(3), "s4": range(5)}
EXPECTED_RA = {"s1": 0.098, "s2": 0.116, "s4": 0.223}


def ensure_dirs():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_srsa_value(obj):
    if "srsa" in obj:
        return obj["srsa"]
    for key in ("spatial_srsa", "srsa_euclid", "sRSA", "rsa"):
        if key in obj and isinstance(obj[key], (int, float)):
            return obj[key]
    raise KeyError(f"Could not find srsa scalar in evaluation.json keys={list(obj.keys())}")


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else 0.0


def fmt_ms(vals):
    m, s = mean_std(vals)
    return f"{m:.6f} ± {s:.6f}"


def normalize_rows(H):
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return H / norms


def pair_key(i, j):
    return tuple(sorted((int(i), int(j))))


def build_permutation(pos_coords, rotation):
    lookup = {(int(r), int(c)): idx for idx, (r, c) in enumerate(pos_coords)}
    min_r, min_c = np.min(pos_coords, axis=0).astype(int)
    max_r, max_c = np.max(pos_coords, axis=0).astype(int)
    if (max_r - min_r) != (max_c - min_c):
        raise ValueError(f"Rotation requires square coordinates, got rows {min_r}-{max_r}, cols {min_c}-{max_c}")
    perm = np.zeros(len(pos_coords), dtype=int)
    for idx, (r, c) in enumerate(pos_coords):
        r, c = int(r), int(c)
        if rotation == "90":
            nr, nc = min_r + (c - min_c), max_c - (r - min_r)
        elif rotation == "180":
            nr, nc = max_r - (r - min_r), max_c - (c - min_c)
        elif rotation == "270":
            nr, nc = max_r - (c - min_c), min_c + (r - min_r)
        else:
            raise ValueError(rotation)
        perm[idx] = lookup[(nr, nc)]
    return perm


def unique_undirected_pairs(pairs):
    out = sorted({pair_key(i, j) for i, j in pairs if int(i) != int(j)})
    return out


def cosine_pair_mean(H_norm, pairs):
    if not pairs:
        return float("nan")
    ii = np.fromiter((p[0] for p in pairs), dtype=int)
    jj = np.fromiter((p[1] for p in pairs), dtype=int)
    return float(np.mean(1.0 - np.sum(H_norm[ii] * H_norm[jj], axis=1)))


def sample_random_pairs(n_pairs, n_pos, rng):
    all_pairs = list(itertools.combinations(range(n_pos), 2))
    idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
    return [all_pairs[i] for i in idx]


def compute_ra(H, evs, coherence_per_unit):
    # Stored n_valid_units is the finite coherence subset after the EVS gate.
    valid_units = np.where((evs > EVS_THRESHOLD) & np.isfinite(coherence_per_unit))[0]
    ra = np.full(H.shape[1], np.nan, dtype=float)
    for u in valid_units:
        T = H[:, u].reshape(18, 18)
        r = pearsonr(T.ravel(), np.rot90(T, k=1).ravel()).statistic
        ra[u] = r
    return ra, valid_units


def compute_decode_err(H, pos_coords):
    X = StandardScaler().fit_transform(H)
    y = pos_coords.astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errors = []
    for train_idx, test_idx in kf.split(X):
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge.fit(X[train_idx], y[train_idx])
        pred = ridge.predict(X[test_idx])
        errors.append(np.mean((pred - y[test_idx]) ** 2))
    return float(np.mean(errors) / (18**2))


def mann_whitney_with_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    res = mannwhitneyu(x, y, alternative="two-sided", method="exact")
    n1, n2 = len(x), len(y)
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (res.statistic - mu) / sigma if sigma else float("nan")
    r = z / math.sqrt(n1 + n2) if n1 + n2 else float("nan")
    return float(res.statistic), float(res.pvalue), float(r)


def load_all_data():
    all_data = {}
    rows = []
    summaries = {}
    for cond, seeds in CONDITIONS.items():
        summary_path = SWEEP / cond / "condition_summary.pkl"
        if summary_path.exists():
            summaries[cond] = load_pickle(summary_path)
        for seed in seeds:
            seed_name = f"seed_{seed:02d}"
            ev = load_pickle(SWEEP / cond / seed_name / "evaluation.pkl")
            ev_json = load_json(SWEEP / cond / seed_name / "evaluation.json")
            pfc = ev["place_field_coherence"]
            rgc = ev["rgc"]
            key = f"{cond}_{seed_name}"
            all_data[key] = {
                "condition": cond,
                "seed": seed,
                "H": ev["position_hidden"],
                "RSA": ev["rsa_matrix"],
                "spatial_srsa": float(ev["srsa"]),
                "pos_coords": ev["position_array"],
                "pos_counts": ev["position_counts"],
                "evs": pfc["evs"],
                "coherence_per_unit": pfc["per_unit_score"],
                "coherence_mean": float(pfc["mean_score"]),
                "coherence_std": float(pfc["std_score"]),
                "n_valid": int(pfc["n_valid_units"]),
                "occupancy": pfc["occupancy"],
                "rgc_stress": float(rgc["stress"]),
                "rgc_stress_pca": float(rgc["stress_pca"]),
                "pca_var_2d": float(rgc["pca_var_2d"]),
                "json_srsa": float(json_srsa_value(ev_json)),
            }
            rows.append(
                {
                    "condition": cond.upper(),
                    "seed": seed_name,
                    "spatial_sRSA": all_data[key]["spatial_srsa"],
                    "rgc_stress": all_data[key]["rgc_stress"],
                    "coherence_mean": all_data[key]["coherence_mean"],
                    "n_valid": all_data[key]["n_valid"],
                }
            )
    with open(METRICS_DIR / "all_evaluations.pkl", "wb") as f:
        pickle.dump(all_data, f)
    return all_data, summaries, pd.DataFrame(rows)


def build_s4_summary(all_data):
    vals = [all_data[f"s4_seed_{i:02d}"] for i in range(5)]
    srsa = [v["spatial_srsa"] for v in vals]
    rgc = [v["rgc_stress"] for v in vals]
    coh_mean = [v["coherence_mean"] for v in vals]
    coh_std = [v["coherence_std"] for v in vals]
    summary = {
        "srsa_mean": mean_std(srsa)[0],
        "srsa_std": mean_std(srsa)[1],
        "srsa_per_seed": srsa,
        "rgc_stress_mean": mean_std(rgc)[0],
        "rgc_stress_std": mean_std(rgc)[1],
        "place_field_coherence_mean": mean_std(coh_mean)[0],
        "place_field_coherence_std": mean_std(coh_std)[0],
    }
    with open(METRICS_DIR / "s4_condition_summary_built.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def compute_all_metrics(all_data, summaries):
    soft_flags = []
    hard_stops = []
    rows = []
    ra_rows = []
    coherence_rows = []

    pos_coords = next(iter(all_data.values()))["pos_coords"]
    perm_90 = build_permutation(pos_coords, "90")
    perm_180 = build_permutation(pos_coords, "180")
    perm_270 = build_permutation(pos_coords, "270")
    if not np.array_equal(perm_90[perm_90[perm_90[perm_90]]], np.arange(324)):
        hard_stops.append("B3 perm_90 four-rotation identity check failed.")
        return None, None, soft_flags, hard_stops

    paa_rows = []
    paa_by_condition = {}
    seed_paa = {}
    triu_idx = np.triu_indices(324, k=1)
    for cond, seeds in CONDITIONS.items():
        pairs = list(itertools.combinations(seeds, 2))
        applicable = [perm_180] if cond in ("s1", "s2") else [perm_90, perm_180, perm_270]
        cond_gains = []
        cond_baselines = []
        for si, sj in pairs:
            ki = f"{cond}_seed_{si:02d}"
            kj = f"{cond}_seed_{sj:02d}"
            RSA_i = all_data[ki]["RSA"]
            RSA_j = all_data[kj]["RSA"]
            vec_i = RSA_i[triu_idx]
            vec_j = RSA_j[triu_idx]
            baseline = float(spearmanr(vec_i, vec_j).correlation)
            max_rotated = baseline
            for perm in applicable:
                rotated = float(spearmanr(vec_i, RSA_j[perm, :][:, perm][triu_idx]).correlation)
                max_rotated = max(max_rotated, rotated)
            gain = max_rotated - baseline
            cond_baselines.append(baseline)
            cond_gains.append(gain)
            paa_rows.append(
                {
                    "condition": cond,
                    "pair": f"seed_{si:02d}-seed_{sj:02d}",
                    "baseline_rho": baseline,
                    "max_rotated_rho": max_rotated,
                    "PAA_gain": gain,
                }
            )
        if cond in summaries:
            stored = float(summaries[cond]["alignment"]["mean_rho"])
            computed = float(np.mean(cond_baselines))
            if abs(computed - stored) > 0.005:
                hard_stops.append(
                    f"B3 baseline validation failed for {cond.upper()}: computed={computed:.6f}, stored={stored:.6f}"
                )
        paa_by_condition[cond] = {
            "mean": float(np.mean(cond_gains)),
            "std": float(np.std(cond_gains, ddof=1)) if len(cond_gains) > 1 else 0.0,
            "pairs": [r for r in paa_rows if r["condition"] == cond],
        }
        if paa_by_condition[cond]["mean"] > 0.05:
            hard_stops.append(
                f"B3 mean PAA_gain > 0.05 for {cond.upper()}: {paa_by_condition[cond]['mean']:.6f}"
            )
        for seed in seeds:
            involved = [
                r["PAA_gain"]
                for r in paa_rows
                if r["condition"] == cond and f"seed_{seed:02d}" in r["pair"].split("-")
            ]
            seed_paa[(cond, seed)] = float(np.mean(involved)) if involved else float("nan")

    if hard_stops:
        return None, paa_by_condition, soft_flags, hard_stops

    with open(METRICS_DIR / "PAA_gains.json", "w", encoding="utf-8") as f:
        json.dump(paa_by_condition, f, indent=2)

    for key, data in all_data.items():
        cond = data["condition"]
        seed = data["seed"]
        H = data["H"]
        H_norm = normalize_rows(H)
        evs = data["evs"]

        ra_per_unit, valid_units = compute_ra(H, evs, data["coherence_per_unit"])
        if abs(len(valid_units) - data["n_valid"]) > 5:
            hard_stops.append(
                f"B2 n_valid mismatch for {cond.upper()} seed_{seed:02d}: computed={len(valid_units)}, stored={data['n_valid']}"
            )
        np.save(METRICS_DIR / f"{cond}_seed{seed:02d}_RA_per_unit.npy", ra_per_unit)
        np.save(METRICS_DIR / f"{cond}_seed{seed:02d}_EVS_per_unit.npy", evs)
        ra_mean = float(np.nanmean(ra_per_unit))
        ra_std = float(np.nanstd(ra_per_unit, ddof=1))
        if abs(ra_mean - EXPECTED_RA[cond]) > 0.05:
            soft_flags.append(
                f"RA deviation for {cond.upper()} seed_{seed:02d}: {ra_mean:.6f} vs expected {EXPECTED_RA[cond]:.6f}"
            )
        ra_rows.append(
            {
                "condition": cond,
                "seed": f"seed_{seed:02d}",
                "N_valid": len(valid_units),
                "stored_n_valid": data["n_valid"],
                "mean_RA": ra_mean,
                "std_RA": ra_std,
            }
        )

        if cond == "s4":
            sym_pairs = unique_undirected_pairs(
                [(i, perm_90[i]) for i in range(324)]
                + [(i, perm_180[i]) for i in range(324)]
                + [(i, perm_270[i]) for i in range(324)]
            )
        elif cond == "s2":
            sym_pairs = unique_undirected_pairs([(i, perm_180[i]) for i in range(324)])
        else:
            s4_count = len(
                unique_undirected_pairs(
                    [(i, perm_90[i]) for i in range(324)]
                    + [(i, perm_180[i]) for i in range(324)]
                    + [(i, perm_270[i]) for i in range(324)]
                )
            )
            sym_pairs = sample_random_pairs(s4_count, 324, np.random.default_rng(10_000 + seed))

        rng = np.random.default_rng(42 + seed)
        random_pairs = sample_random_pairs(len(sym_pairs), 324, rng)
        sci = cosine_pair_mean(H_norm, sym_pairs) / cosine_pair_mean(H_norm, random_pairs)

        c2_pairs = [(i, int(perm_180[i])) for i in range(324) if i != int(perm_180[i])]
        c4_pairs = [(i, int(perm_90[i])) for i in range(324) if i != int(perm_90[i])]
        c4_pairs += [(i, int(perm_270[i])) for i in range(324) if i != int(perm_270[i])]
        c2_set = {pair_key(i, j) for i, j in c2_pairs}
        c4_only = [(i, j) for i, j in c4_pairs if pair_key(i, j) not in c2_set]
        c2_contrast = cosine_pair_mean(H_norm, c2_pairs) - cosine_pair_mean(H_norm, c4_only)

        pairwise_neural = pdist(H_norm, "cosine")
        pairwise_euclid = pdist(data["pos_coords"].astype(float), "euclidean")
        pairwise_city = pdist(data["pos_coords"].astype(float), "cityblock")
        srsa_euclid = float(spearmanr(pairwise_neural, pairwise_euclid).correlation)
        srsa_city = float(spearmanr(pairwise_neural, pairwise_city).correlation)
        dtg = srsa_euclid - srsa_city

        decode_err = compute_decode_err(H, data["pos_coords"])
        valid_coh = data["coherence_per_unit"][(evs > EVS_THRESHOLD) & np.isfinite(data["coherence_per_unit"])]
        coh_median = float(np.nanmedian(valid_coh))
        coh_mean_valid = float(np.nanmean(valid_coh))
        frac_gt_10 = float(np.nanmean(valid_coh > 10.0))
        frac_gt_50 = float(np.nanmean(valid_coh > 50.0))
        frac_gt_100 = float(np.nanmean(valid_coh > 100.0))
        if frac_gt_50 > 0.01 or data["coherence_std"] > data["coherence_mean"] * 2:
            soft_flags.append(
                f"Coherence skew for {cond.upper()} seed_{seed:02d}: median={coh_median:.3f}, mean={data['coherence_mean']:.3f}, std={data['coherence_std']:.3f}"
            )

        if abs(srsa_euclid - data["spatial_srsa"]) > 0.02:
            data["srsa_mismatch"] = True
        else:
            data["srsa_mismatch"] = False

        rows.append(
            {
                "condition": cond.upper(),
                "seed": f"seed_{seed:02d}",
                "spatial_sRSA": data["spatial_srsa"],
                "sRSA_euclid_computed": srsa_euclid,
                "sRSA_city_computed": srsa_city,
                "json_srsa": data["json_srsa"],
                "RA": ra_mean,
                "PAA_gain": seed_paa[(cond, seed)],
                "SCI": sci,
                "C2_Contrast": c2_contrast,
                "DTG": dtg,
                "DecodeErr": decode_err,
                "coherence_median": coh_median,
                "coherence_mean": data["coherence_mean"],
                "coherence_mean_valid": coh_mean_valid,
                "coherence_frac_gt_10": frac_gt_10,
                "coherence_frac_gt_50": frac_gt_50,
                "coherence_frac_gt_100": frac_gt_100,
                "rgc_stress": data["rgc_stress"],
                "rgc_stress_pca": data["rgc_stress_pca"],
                "pca_var_2d": data["pca_var_2d"],
                "n_valid_units": len(valid_units),
                "stored_n_valid_units": data["n_valid"],
            }
        )
        coherence_rows.append(rows[-1])

    if hard_stops:
        return None, paa_by_condition, soft_flags, hard_stops

    result_df = pd.DataFrame(rows)
    srsa_mismatch_count = int(np.sum(np.abs(result_df["sRSA_euclid_computed"] - result_df["spatial_sRSA"]) > 0.02))
    if srsa_mismatch_count > len(result_df) / 2:
        hard_stops.append(f"B6 sRSA mismatch >0.02 for majority of seeds: {srsa_mismatch_count}/{len(result_df)}")
        return result_df, paa_by_condition, soft_flags, hard_stops
    elif srsa_mismatch_count:
        soft_flags.append(f"sRSA mismatch >0.02 for {srsa_mismatch_count}/{len(result_df)} seeds.")

    pd.DataFrame(paa_rows).to_csv(TABLES_DIR / "paa_pair_values.csv", index=False)
    pd.DataFrame(ra_rows).to_csv(TABLES_DIR / "ra_values.csv", index=False)
    pd.DataFrame(coherence_rows).to_csv(TABLES_DIR / "coherence_distribution_summary.csv", index=False)

    qdm_summary = compute_qdm(all_data, result_df, perm_180)
    with open(METRICS_DIR / "quadrant_distance_matrices.json", "w", encoding="utf-8") as f:
        json.dump(qdm_summary, f, indent=2)
    if not qdm_summary["s2_pattern_confirmed"]:
        soft_flags.append("S2 C2 QDM pattern not detected.")

    parse_training_logs(soft_flags)
    stats_df = compute_stats(result_df)
    stats_df.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)
    result_df.to_csv(TABLES_DIR / "master_metrics.csv", index=False)

    return result_df, paa_by_condition, soft_flags, hard_stops


def compute_qdm(all_data, result_df, perm_180):
    qdm_by_condition = {}
    for cond, seeds in CONDITIONS.items():
        matrices = []
        for seed in seeds:
            data = all_data[f"{cond}_seed_{seed:02d}"]
            H_norm = normalize_rows(data["H"])
            pos = data["pos_coords"]
            labels = np.zeros(324, dtype=int)
            for idx, (r, c) in enumerate(pos):
                if r <= 9 and c <= 9:
                    labels[idx] = 0
                elif r <= 9 and c > 9:
                    labels[idx] = 1
                elif r > 9 and c <= 9:
                    labels[idx] = 2
                else:
                    labels[idx] = 3
            qdm = np.zeros((4, 4), dtype=float)
            for i in range(4):
                idx_i = np.where(labels == i)[0]
                for j in range(4):
                    idx_j = np.where(labels == j)[0]
                    pairs = [(a, b) for a in idx_i for b in idx_j if a != b]
                    qdm[i, j] = cosine_pair_mean(H_norm, pairs)
            matrices.append(qdm)
        qdm_by_condition[cond] = np.mean(matrices, axis=0).tolist()
    s2 = np.asarray(qdm_by_condition["s2"])
    return {
        "matrices": qdm_by_condition,
        "s2_pattern_confirmed": bool(s2[0, 2] < s2[0, 1] and s2[1, 3] < s2[1, 2]),
        "s2_checks": {
            "Q1_Q3": float(s2[0, 2]),
            "Q1_Q2": float(s2[0, 1]),
            "Q2_Q4": float(s2[1, 3]),
            "Q2_Q3": float(s2[1, 2]),
        },
    }


def parse_training_logs(soft_flags):
    for cond, seeds in CONDITIONS.items():
        for seed in seeds:
            seed_name = f"seed_{seed:02d}"
            log = load_json(SWEEP / cond / seed_name / "training_log.json")
            df = pd.DataFrame(
                {
                    "condition": cond.upper(),
                    "seed": seed_name,
                    "steps": log["steps"],
                    "srsa_euclid": log["srsa_euclid"],
                    "srsa_city": log["srsa_city"],
                    "loss": log["loss"],
                    "manifold_id": log["manifold_id"],
                    "pca_variance_2d": log["pca_variance_2d"],
                    "mds_stress": log["mds_stress"],
                    "mean_field_coherence": log["mean_field_coherence"],
                }
            )
            df["DTG_series"] = df["srsa_euclid"] - df["srsa_city"]
            obs = log["observation_discriminability"]
            df["observation_discriminability"] = obs
            if isinstance(obs, list) and np.nanstd(obs) > 1e-9:
                soft_flags.append(f"observation_discriminability varies for {cond.upper()} {seed_name}")
            df.to_csv(TABLES_DIR / f"training_log_{cond}_{seed_name}.csv", index=False)


def compute_stats(result_df):
    metrics = [
        "spatial_sRSA",
        "RA",
        "PAA_gain",
        "SCI",
        "C2_Contrast",
        "DTG",
        "DecodeErr",
        "coherence_median",
        "rgc_stress",
    ]
    comparisons = [("S4", "S1"), ("S4", "S2"), ("S2", "S1")]
    rows = []
    alpha = 0.05 / (len(metrics) * len(comparisons))
    for metric in metrics:
        vals = {cond: result_df[result_df["condition"] == cond][metric].to_numpy(dtype=float) for cond in ("S4", "S2", "S1")}
        for a, b in comparisons:
            U, p, r = mann_whitney_with_r(vals[a], vals[b])
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{a}_vs_{b}",
                    "U": U,
                    "p": p,
                    "effect_r": r,
                    "bonferroni_alpha": alpha,
                    "survives_bonferroni": bool(p < alpha),
                    "S4_mean": mean_std(vals["S4"])[0],
                    "S4_std": mean_std(vals["S4"])[1],
                    "S2_mean": mean_std(vals["S2"])[0],
                    "S2_std": mean_std(vals["S2"])[1],
                    "S1_mean": mean_std(vals["S1"])[0],
                    "S1_std": mean_std(vals["S1"])[1],
                }
            )
    return pd.DataFrame(rows)


def print_outputs(initial_df, result_df, paa_by_condition, summaries, soft_flags, hard_stops):
    print("\nB1 initial load table")
    print(initial_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if hard_stops:
        print("\nHARD STOPS")
        for flag in hard_stops:
            print(f"- {flag}")
        return

    print("\nB2 RA table")
    print(pd.read_csv(TABLES_DIR / "ra_values.csv").to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nB3 PAA pair table")
    print(pd.read_csv(TABLES_DIR / "paa_pair_values.csv").to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nB3 PAA condition means")
    for cond in ("s4", "s2", "s1"):
        print(f"{cond.upper()}: {paa_by_condition[cond]['mean']:.6f} ± {paa_by_condition[cond]['std']:.6f}")

    print("\nB12 condition summary")
    metrics = [
        "spatial_sRSA",
        "RA",
        "PAA_gain",
        "SCI",
        "C2_Contrast",
        "DTG",
        "DecodeErr",
        "coherence_median",
        "rgc_stress",
    ]
    summary_rows = []
    for metric in metrics:
        summary_rows.append(
            {
                "Metric": metric,
                "S4 (n=5)": fmt_ms(result_df[result_df["condition"] == "S4"][metric]),
                "S2 (n=3)": fmt_ms(result_df[result_df["condition"] == "S2"][metric]),
                "S1 (n=3)": fmt_ms(result_df[result_df["condition"] == "S1"][metric]),
            }
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nB11 statistical tests")
    print(pd.read_csv(TABLES_DIR / "statistical_tests.csv").to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\nODI validation")
    for cond in ("s1", "s2"):
        odi = summaries[cond]["odi"]["rho"]
        status = "OK" if abs(float(odi) - 0.336) <= 0.01 else "FLAG"
        print(f"{cond.upper()} ODI rho={float(odi):.6f} [{status}]")
        if status == "FLAG":
            soft_flags.append(f"{cond.upper()} ODI differs from 0.336 by >0.01: {float(odi):.6f}")

    print("\nOutput files")
    for path in sorted([*METRICS_DIR.glob("*"), *TABLES_DIR.glob("*")]):
        if path.is_file() and (
            path.name in {
                "all_evaluations.pkl",
                "s4_condition_summary_built.json",
                "PAA_gains.json",
                "quadrant_distance_matrices.json",
                "master_metrics.csv",
                "statistical_tests.csv",
                "paa_pair_values.csv",
                "ra_values.csv",
                "coherence_distribution_summary.csv",
            }
            or path.name.startswith(("s1_seed", "s2_seed", "s4_seed", "training_log_"))
        ):
            print(f"{path.relative_to(BASE)}\t{path.stat().st_size} bytes")

    print("\nSoft flags")
    if soft_flags:
        for flag in soft_flags:
            print(f"- {flag}")
    else:
        print("None")

    print("\nB1-B12 COMPLETE")


def main():
    ensure_dirs()
    initial_df = None
    result_df = None
    paa_by_condition = None
    soft_flags = []
    hard_stops = []
    try:
        all_data, summaries, initial_df = load_all_data()
        build_s4_summary(all_data)
        result_df, paa_by_condition, soft_flags, hard_stops = compute_all_metrics(all_data, summaries)
        print_outputs(initial_df, result_df, paa_by_condition, summaries, soft_flags, hard_stops)
        if hard_stops:
            raise SystemExit(2)
    except Exception as exc:
        print("\nUNHANDLED ERROR")
        print(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
