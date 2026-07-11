"""Did a map form, and how good is it? -- reported ALONGSIDE phase decoding, never instead.

Two different questions, two different instruments:

  phase decoding   did the code fold onto the quotient X/G?
  map quality      did a spatial map form at all, and is it reproducible across seeds?

**A folded code can score high on every map-quality measure.** If two seeds both fold onto
X/C2, their representational geometries agree beautifully -- the folding is reproducible. So
cross-seed RSA must never be read as evidence against folding. The test suite pins this.

What is computed, all on the project's own definitions (evaluation/metrics.py):

  srsa_e, srsa_c   Spatial RSA: Spearman(cosine neural distance, Euclidean / cityblock
                   spatial distance) over pairs of timepoints. NOTE: this is neural-vs-space,
                   not seed-to-seed. The Introduction's "row permutations preserve pairwise
                   distances" argument describes cross-seed RSA, a different quantity.
                   Spatial RSA *is* sensitive to folding -- a folded pair has zero neural
                   distance at large spatial distance -- so it drops when the code folds.
  cross_seed_rho   Pairwise Spearman between seeds' position-conditioned neural RDMs. This is
                   the permutation-blind one, and the one a folded code scores high on.
  spatial_info     Skaggs bits per activation, median over the 100 most tuned units.
  field_sharpness  autocorr(0,0) / mean autocorr at radius 2-5. A peak-to-annulus RATIO, not a
                   correlation: routinely > 1, bigger means sharper, more localised fields.
  n_place_units    units whose tuning map passes the 10% spatial-EVS threshold.

The HD-ablation number in the original report (sRSA 0.615 -> 0.260) was never recomputed on
the clean checkpoints. It is recomputed here.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import (collect, identify,  # noqa: E402
                                                     model_from_checkpoint)
from project5_symmetry.evaluation.metrics import (aggregate_hidden_by_position,  # noqa: E402
                                                  compute_tuning_curves,
                                                  cross_seed_rsa_alignment,
                                                  place_field_spatial_coherence,
                                                  spatial_information, srsa)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def position_rdm(hidden, pos, arena=18):
    """Cosine RDM over the position-conditioned mean population vectors."""
    agg = aggregate_hidden_by_position(hidden, pos)
    M = agg['hidden']
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Mn @ Mn.T


def map_quality(hidden, pos, arena=18):
    out = {}
    n = min(len(hidden), 5000)
    idx = np.random.default_rng(0).choice(len(hidden), n, replace=False)
    h, p = hidden[idx], pos[idx].astype(float)
    out['srsa_e'] = float(srsa(h, p, space_metric='euclidean', max_n=n))
    out['srsa_c'] = float(srsa(h, p, space_metric='cityblock', max_n=n))
    try:
        # metrics.compute_tuning_curves' docstring says it returns (dict, occ), but the
        # installed pynapple's compute_2d_tuning_curves_continuous returns (tc_dict, bins),
        # so the real shape is ((tc_dict, bins), occ). Accept either.
        tc, occ = compute_tuning_curves(hidden, pos, nb_bins=arena)
        tc_dict = tc[0] if isinstance(tc, tuple) else tc
        si = np.asarray(spatial_information(tc_dict, occ), dtype=float)
        top = np.argsort(hidden.var(0))[-100:]
        si_top = si[top][np.isfinite(si[top])]
        out['spatial_info'] = float(np.median(si_top)) if si_top.size else float('nan')
        out['n_informative'] = int(np.nansum(si > 0.2))
    except Exception as e:                              # report, never silently zero
        out['spatial_info'] = float('nan'); out['n_informative'] = -1
        print(f'    spatial_information failed: {type(e).__name__}: {e}', flush=True)
    try:
        # NOTE: `mean_score` is autocorr(0,0) / mean autocorr over radius 2..5 -- a
        # peak-to-annulus ratio, i.e. FIELD SHARPNESS. It is not a correlation and is
        # routinely > 1. Bigger = sharper, more localised fields.
        coh = place_field_spatial_coherence(hidden, pos, arena_size=arena)
        out['field_sharpness'] = float(coh['mean_score'])
        out['n_place_units'] = int(coh['n_valid_units'])
    except Exception as e:
        out['field_sharpness'] = float('nan'); out['n_place_units'] = -1
        print(f'    coherence failed: {type(e).__name__}: {e}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds, rows, rdms = {}, [], defaultdict(list)
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(p, ck['meta'])
        if cond not in ds:
            ds[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        hidden, pos = collect(model, ds[cond], hd, a.n_states, dev)
        q = map_quality(hidden, pos, a.arena)
        rdms[(cond, hd)].append(position_rdm(hidden, pos, a.arena))
        rows.append({'condition': cond, 'hd_mode': hd, 'seed': seed, **q})
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/seed{seed}  srsa_e={q["srsa_e"]:+.4f} '
              f'srsa_c={q["srsa_c"]:+.4f} SI={q["spatial_info"]:.3f} '
              f'sharp={q["field_sharpness"]:.2f} nplace={q["n_place_units"]}', flush=True)

    # cross-seed RSA per (condition, hd): the permutation-blind measure a folded code passes
    cs = {}
    for key, mats in rdms.items():
        if len(mats) >= 2:
            cs[key] = cross_seed_rsa_alignment(mats)['mean_rho']
    for r in rows:
        r['cross_seed_rho'] = cs.get((r['condition'], r['hd_mode']), float('nan'))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f'\n{"cond/hd":<14s} {"srsa_e":>8s} {"srsa_c":>8s} {"cross-seed":>11s} '
          f'{"SI":>7s} {"sharp":>7s} {"n_place":>8s}')
    for key in sorted(rdms):
        sub = [r for r in rows if (r['condition'], r['hd_mode']) == key]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f'{key[0]+"/"+key[1]:<14s} {m("srsa_e"):>8.4f} {m("srsa_c"):>8.4f} '
              f'{cs.get(key, float("nan")):>11.4f} {m("spatial_info"):>7.3f} '
              f'{m("field_sharpness"):>7.2f} {m("n_place_units"):>8.0f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
