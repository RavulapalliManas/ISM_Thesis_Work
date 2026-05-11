#!/usr/bin/env python3
"""Part 1: Data loading, metric computation, statistics. Saves master_metrics.csv and stats."""

import subprocess, sys, pickle, json, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, spearmanr
from scipy.ndimage import rotate as ndimage_rotate
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

BASE = Path('results2')
RESULTS_DIR = Path('results2/analysis_output')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical positions for cases where evaluation.pkl is missing
CANONICAL_POSITIONS = None
try:
    with open('results2/symmetry_sweep/s1/seed_00/evaluation.pkl', 'rb') as f:
        CANONICAL_POSITIONS = pickle.load(f)['position_array']
except:
    pass

# ── Data Loading ──────────────────────────────────────────────────────────

def load_eval(seed_dir):
    for name in ['evaluation.pkl', 'eval.pkl']:
        p = seed_dir / name
        if p.exists():
            with open(p, 'rb') as f:
                return pickle.load(f)
    return None

def load_training_log(seed_dir):
    for name in ['training_log.json', 'training_curve.json', 'metrics.json']:
        p = seed_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None

def extract_srsa(ev):
    for key in ['srsa', 'spatial_rsa', 'sRSA']:
        if key in ev:
            val = float(ev[key]) if not isinstance(ev[key], (list, np.ndarray)) else float(np.mean(ev[key]))
            if 0.05 <= val <= 0.98:
                return val
    return None

def extract_H(ev):
    for key in ['position_hidden', 'hidden_states', 'H']:
        if key in ev:
            H = np.array(ev[key])
            if H.ndim == 2 and H.shape[0] >= 100:
                return H
    return None

def extract_positions(ev):
    for key in ['position_array', 'positions', 'pos', 'xy']:
        if key in ev:
            pos = np.array(ev[key], dtype=float)
            if pos.ndim == 2 and pos.shape[1] == 2:
                return pos
    return None

def load_condition(base_dir, cond_name, seed_range):
    candidates = [base_dir / cond_name, base_dir / cond_name.lower(),
                  base_dir / f'condition_{cond_name}']
    cond_dir = next((d for d in candidates if d.exists()), None)
    if cond_dir is None:
        print(f"  WARNING: Cannot find {cond_name} in {base_dir}")
        return []
    records = []
    for sid in seed_range:
        sd = cond_dir / f'seed_{sid:02d}'
        if not sd.exists():
            continue
        ev = load_eval(sd)
        if ev is None:
            # Try loading final H matrix for ablation data
            h_files = sorted(sd.glob('H_*.npy'))
            if h_files:
                H_final = np.load(h_files[-1])
                log = load_training_log(sd)
                records.append({
                    'seed_id': sid, 'seed_dir': sd, 'eval': {},
                    'training_log': log, 'srsa': None, 'H': H_final,
                    'positions': None, 'condition': cond_name,
                    'has_eval': False
                })
                print(f"  Loaded {cond_name}/seed_{sid:02d}: H from {h_files[-1].name} (no eval)")
            else:
                print(f"  SKIP: {cond_name}/seed_{sid:02d} — no eval.pkl, no H")
            continue
        srsa = extract_srsa(ev)
        H = extract_H(ev)
        pos = extract_positions(ev)
        log = load_training_log(sd)
        srsa_str = f"sRSA={srsa:.3f}" if srsa else "sRSA=MISSING"
        print(f"  Loaded {cond_name}/seed_{sid:02d}: {srsa_str}")
        records.append({
            'seed_id': sid, 'seed_dir': sd, 'eval': ev,
            'training_log': log, 'srsa': srsa, 'H': H,
            'positions': pos, 'condition': cond_name, 'has_eval': True
        })
    return records

# ── Metric Computation ────────────────────────────────────────────────────

def compute_ra(H, positions, symmetry_order=4, grid_size=18):
    if H is None or positions is None: return np.nan
    pos_min, pos_max = positions.min(0), positions.max(0)
    pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
    xi = (pos_norm[:, 0] * (grid_size-1)).astype(int).clip(0, grid_size-1)
    yi = (pos_norm[:, 1] * (grid_size-1)).astype(int).clip(0, grid_size-1)
    angle_deg = 360.0 / symmetry_order
    ra_vals = []
    for u in range(min(H.shape[1], 200)):  # cap at 200 units for speed
        rmap = np.zeros((grid_size, grid_size))
        cmap = np.zeros((grid_size, grid_size))
        for i in range(len(positions)):
            rmap[yi[i], xi[i]] += H[i, u]
            cmap[yi[i], xi[i]] += 1
        mask = cmap > 0
        rmap[mask] /= cmap[mask]
        rotated = ndimage_rotate(rmap, angle_deg, reshape=False, mode='constant', cval=0.0)
        flat, flat_rot = rmap[mask].flatten(), rotated[mask].flatten()
        if flat.std() > 1e-6 and flat_rot.std() > 1e-6:
            r, _ = spearmanr(flat, flat_rot)
            ra_vals.append(r)
    return float(np.nanmean(ra_vals)) if ra_vals else np.nan

def compute_sci(H, positions, symmetry_order=4):
    if H is None or positions is None: return None
    try:
        Hc = H - H.mean(0)
        D = squareform(pdist(Hc, 'cosine'))
        angle = 2 * np.pi / symmetry_order
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pos_c = positions - positions.mean(0)
        rotated = pos_c @ R.T
        thresh = np.median(np.linalg.norm(pos_c, axis=1)) * 0.2
        partner_d = []
        for i in range(len(positions)):
            diffs = np.linalg.norm(pos_c - rotated[i], axis=1)
            j = np.argmin(diffs)
            if diffs[j] < thresh and i != j:
                partner_d.append(D[i, j])
        if not partner_d: return None
        all_mean = np.mean(D[np.triu_indices_from(D, k=1)])
        return float(1.0 - (np.mean(partner_d) / (all_mean + 1e-8)))
    except: return None

def compute_c2_contrast(H, positions):
    if H is None or positions is None: return None
    try:
        Hc = H - H.mean(0)
        D = squareform(pdist(Hc, 'euclidean'))
        pos_c = positions - positions.mean(0)
        scale = np.median(np.linalg.norm(pos_c, axis=1))
        thresh = scale * 0.25
        def rot_pairs(angle):
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rp = pos_c @ R.T
            pairs = []
            for i in range(len(pos_c)):
                d = np.linalg.norm(pos_c - rp[i], axis=1)
                j = np.argmin(d)
                if d[j] < thresh and j > i: pairs.append((i, j))
            return pairs
        c2 = rot_pairs(np.pi)
        c4 = rot_pairs(np.pi/2)
        if not c2 or not c4: return None
        return float(np.mean([D[i,j] for i,j in c2]) - np.mean([D[i,j] for i,j in c4]))
    except: return None

def compute_decode_error(H, positions, k=5):
    if H is None or positions is None or H.shape[0] < 20: return None
    try:
        knn = KNeighborsRegressor(n_neighbors=k)
        errors = []
        for i in range(len(positions)):
            mask = np.ones(len(positions), dtype=bool); mask[i] = False
            knn.fit(H[mask], positions[mask])
            pred = knn.predict(H[i:i+1])
            errors.append(np.linalg.norm(pred - positions[i]))
        return float(np.mean(errors))
    except: return None

def compute_srsa_manual(H, positions):
    if H is None or positions is None: return None
    try:
        # Neural distance matrix
        D_neural = pdist(H - H.mean(0), 'euclidean')
        # Spatial distance matrix
        D_spatial = pdist(positions, 'euclidean')
        # Spearman correlation
        r, _ = spearmanr(D_neural, D_spatial)
        return float(r)
    except: return None

def compute_frac_tuned(H, positions, threshold=0.1):
    if H is None or positions is None: return None
    try:
        ridge = Ridge(alpha=1.0)
        tuned = 0
        for u in range(H.shape[1]):
            ridge.fit(positions, H[:, u])
            evs = explained_variance_score(H[:, u], ridge.predict(positions))
            if evs > threshold: tuned += 1
        return float(tuned / H.shape[1])
    except: return None

def extract_frac_tuned(ev, threshold=0.1):
    pfc = ev.get('place_field_coherence', {})
    evs_arr = pfc.get('evs')
    if evs_arr is not None:
        evs_arr = np.array(evs_arr)
        return float((evs_arr > threshold).mean())
    return None

def get_all_metrics(rec, sym_order=1):
    ev = rec.get('eval', {})
    H, pos = rec.get('H'), rec.get('positions')
    if pos is None: pos = CANONICAL_POSITIONS
    
    # Truncate to match lengths if using canonical positions or if inconsistent
    if H is not None and pos is not None:
        min_len = min(len(H), len(pos))
        H, pos = H[:min_len], pos[:min_len]
        
    srsa = rec.get('srsa')
    if srsa is None and H is not None and pos is not None:
        srsa = compute_srsa_manual(H, pos)
        
    pfc = ev.get('place_field_coherence', {})

    # Extract or compute each metric
    frac_tuned = extract_frac_tuned(ev)
    if frac_tuned is None and H is not None and pos is not None:
        frac_tuned = compute_frac_tuned(H, pos)

    ra = compute_ra(H, pos, symmetry_order=sym_order) if H is not None and pos is not None else np.nan
    sci = compute_sci(H, pos, symmetry_order=sym_order)
    c2c = compute_c2_contrast(H, pos)
    dec_err = compute_decode_error(H, pos)

    coherence = pfc.get('mean_score')
    rgc = ev.get('rgc', {})

    return {
        'srsa': srsa, 'ra': ra, 'sci': sci, 'c2_contrast': c2c,
        'decode_error': dec_err, 'frac_tuned': frac_tuned,
        'field_coherence': coherence,
        'mds_stress': rgc.get('stress'), 'pca_var_2d': rgc.get('pca_var_2d'),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("LOADING SYMMETRY SWEEP DATA")
print("=" * 60)
sym_base = BASE / 'symmetry_sweep'
s1_seeds = load_condition(sym_base, 's1', range(5))
s2_seeds = load_condition(sym_base, 's2', range(5))
s4_seeds = load_condition(sym_base, 's4', range(9))
conditions = {'S1': s1_seeds, 'S2': s2_seeds, 'S4': s4_seeds}
sym_order = {'S1': 1, 'S2': 2, 'S4': 4}
print(f"\nLoaded: S1={len(s1_seeds)}, S2={len(s2_seeds)}, S4={len(s4_seeds)} seeds")

print("\n" + "=" * 60)
print("LOADING HD ABLATION DATA")
print("=" * 60)
abl_base = BASE / 'ablation'
abl_full = load_condition(abl_base, 'full', [0, 1])
abl_ablated = load_condition(abl_base, 'ablated', [0, 1])
abl_degraded = load_condition(abl_base, 'degraded', [0, 1])
ablation_conds = {'HD_FULL': abl_full, 'HD_ABLATED': abl_ablated, 'HD_DEGRADED': abl_degraded}

print("\n" + "=" * 60)
print("LOADING EPSILON SWEEP DATA")
print("=" * 60)
eps_base = BASE / 'epsilon_sweep'
eps_levels_found = sorted([float(d.name.replace('eps_','')) for d in eps_base.iterdir() if d.is_dir()])
eps_data = {}
for eps in eps_levels_found:
    eps_str = f'eps_{eps:.1f}'
    seeds = load_condition(eps_base, eps_str, range(3))
    eps_data[eps] = seeds
    print(f"  eps={eps:.1f}: {len(seeds)} seeds (H-only, no eval)")

# ── Build Master DataFrame ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPUTING METRICS (this may take a few minutes)")
print("=" * 60)

rows = []
for cond, records in conditions.items():
    for rec in records:
        print(f"  Computing metrics for {cond}/seed_{rec['seed_id']:02d}...")
        metrics = get_all_metrics(rec, sym_order[cond])
        rows.append({'experiment': 'symmetry_sweep', 'condition': cond,
                     'seed_id': rec['seed_id'], **metrics})

for cond, records in ablation_conds.items():
    for rec in records:
        print(f"  Computing metrics for {cond}/seed_{rec['seed_id']:02d}...")
        metrics = get_all_metrics(rec, sym_order=1) # assume S1-like canonical
        rows.append({'experiment': 'hd_ablation', 'condition': cond,
                     'seed_id': rec['seed_id'], **metrics})

# Epsilon seeds have H matrices but no eval - compute all possible metrics
for eps, records in eps_data.items():
    for rec in records:
        print(f"  Computing metrics for eps_{eps:.1f}/seed_{rec['seed_id']:02d}...")
        metrics = get_all_metrics(rec, sym_order=1) # assume S1-like
        rows.append({'experiment': 'epsilon_sweep', 'condition': f'eps_{eps:.1f}',
                     'epsilon': eps, 'seed_id': rec['seed_id'], **metrics})

df = pd.DataFrame(rows)
print("\nMASTER DATAFRAME:")
print(df.to_string(max_rows=40))
df.to_csv(RESULTS_DIR / 'master_metrics.csv', index=False)
print(f"\nSaved: {RESULTS_DIR / 'master_metrics.csv'}")

# ── Statistical Tests ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RUNNING STATISTICAL TESTS")
print("=" * 60)

sym_df = df[df['experiment'] == 'symmetry_sweep']
metrics_to_test = ['srsa', 'ra', 'sci', 'c2_contrast', 'decode_error', 'frac_tuned']
comparisons = [('S1','S2'), ('S2','S4'), ('S1','S4')]

stat_rows = []
for metric in metrics_to_test:
    vals = {c: sym_df[sym_df['condition']==c][metric].dropna().tolist() for c in ['S1','S2','S4']}
    for ca, cb in comparisons:
        a = [x for x in vals[ca] if not np.isnan(x)]
        b = [x for x in vals[cb] if not np.isnan(x)]
        if len(a) < 2 or len(b) < 2:
            stat_rows.append({'metric': metric, 'comparison': f'{ca}_vs_{cb}',
                              'U': np.nan, 'p': np.nan, 'r': np.nan,
                              'mean_a': np.mean(a) if a else np.nan,
                              'mean_b': np.mean(b) if b else np.nan,
                              'n_a': len(a), 'n_b': len(b)})
            continue
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        r = U / (len(a) * len(b))
        stat_rows.append({'metric': metric, 'comparison': f'{ca}_vs_{cb}',
                          'U': U, 'p': p, 'r': r,
                          'mean_a': np.mean(a), 'mean_b': np.mean(b),
                          'n_a': len(a), 'n_b': len(b)})

stat_df = pd.DataFrame(stat_rows)
p_vals = stat_df['p'].dropna().values
if len(p_vals) > 0:
    reject, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='bonferroni')
    stat_df.loc[stat_df['p'].notna(), 'p_bonferroni'] = p_corr
    stat_df.loc[stat_df['p'].notna(), 'significant'] = reject

print("\nSTATISTICS TABLE:")
print(stat_df.to_string())
stat_df.to_csv(RESULTS_DIR / 'statistics_table.csv', index=False)

# Save for Part 2
df.to_pickle(RESULTS_DIR / '_df.pkl')
stat_df.to_pickle(RESULTS_DIR / '_stat_df.pkl')

# Save all condition data for Part 2 figures
import pickle as pkl
with open(RESULTS_DIR / '_conditions.pkl', 'wb') as f:
    # Save minimal data needed for figures (eval dicts are large)
    save_data = {}
    for cond, records in conditions.items():
        save_data[cond] = []
        for rec in records:
            save_data[cond].append({
                'seed_id': rec['seed_id'],
                'srsa': rec['srsa'],
                'H': rec['H'],
                'positions': rec['positions'],
                'training_log': rec['training_log'],
                'eval': rec['eval'],
            })
    pkl.dump(save_data, f)

with open(RESULTS_DIR / '_ablation.pkl', 'wb') as f:
    save_abl = {}
    for cond, records in ablation_conds.items():
        save_abl[cond] = [{'seed_id': r['seed_id'], 'training_log': r['training_log'],
                           'H': r['H']} for r in records]
    pkl.dump(save_abl, f)

print("\n" + "=" * 60)
print("FULL STATISTICS SUMMARY")
print("=" * 60)

print("\n--- SYMMETRY SWEEP ---")
for metric in metrics_to_test:
    print(f"\n  {metric.upper()}:")
    for cond in ['S1', 'S2', 'S4']:
        vals = sym_df[sym_df['condition']==cond][metric].dropna().values
        if len(vals) > 0:
            print(f"    {cond}: {np.mean(vals):.4f} ± {np.std(vals):.4f} "
                  f"(n={len(vals)}, SEM={np.std(vals)/np.sqrt(len(vals)):.4f})")

print("\n--- HD ABLATION ---")
abl_df = df[df['experiment'] == 'hd_ablation']
for cond in ['HD_FULL', 'HD_ABLATED', 'HD_DEGRADED']:
    vals = abl_df[abl_df['condition']==cond]['srsa'].dropna().values
    if len(vals) > 0:
        print(f"  {cond} sRSA: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Paper statistics file
with open(RESULTS_DIR / 'paper_statistics_updates.txt', 'w') as f:
    f.write("# AUTO-GENERATED STATISTICS FOR PAPER UPDATE\n\n")
    for cond in ['S1','S2','S4']:
        f.write(f"\n## Condition {cond}\n")
        for m in metrics_to_test:
            vals = sym_df[sym_df['condition']==cond][m].dropna().values
            if len(vals) > 0:
                f.write(f"  {cond}_{m}: {np.mean(vals):.3f} (±{np.std(vals)/np.sqrt(len(vals)):.3f} SEM, n={len(vals)})\n")
            else:
                f.write(f"  {cond}_{m}: N/A\n")
    f.write("\n## Statistical Tests (Bonferroni)\n")
    for _, row in stat_df.iterrows():
        p_val = row.get('p_bonferroni', row['p'])
        sig = '(sig)' if row.get('significant', False) else '(n.s.)'
        f.write(f"  {row['metric']} {row['comparison']}: p={p_val:.4f} {sig}\n")
    f.write("\n## HD Ablation\n")
    for cond in ['HD_FULL','HD_ABLATED','HD_DEGRADED']:
        vals = abl_df[abl_df['condition']==cond]['srsa'].dropna().values
        s = f"{np.mean(vals):.3f}" if len(vals) > 0 else "N/A"
        f.write(f"  {cond}_sRSA: {s}\n")

print(f"\nSaved: {RESULTS_DIR / 'paper_statistics_updates.txt'}")
print("\nPART 1 COMPLETE. Run full_analysis_part2.py for figures.")
