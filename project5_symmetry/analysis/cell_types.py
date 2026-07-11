"""Cell-type characterisation of the learned units, per checkpoint.

The predictive network is never told about place, head-direction, or border cells; it only
predicts its next view. This script asks what the entorhinal-hippocampal cell zoo looks like in
the trained population, using the standard read-outs neuroscientists apply to real recordings:

    spatial_info   Skaggs spatial information (bits/spike-equivalent) -- place-cell signature
    hd_rayleigh    resultant-vector length of mean activity across the 4 headings -- HD tuning
    border_frac    fraction of above-threshold rate-map mass in the outer 2-cell ring
    n_fields       connected supra-half-peak regions -- multi-field (folding) signature

A unit is labelled a place cell (spatial_info >= si_thresh), an HD cell (hd_rayleigh >= hd_thresh),
or a border cell (border_frac >= border_thresh); labels are not exclusive. The point is twofold:
(1) the model develops a biologically realistic mix without being asked to, and (2) the fold
shows up as increased place-field repetition (n_fields) with the cell-type composition otherwise
preserved -- folding rearranges where fields sit, it does not destroy the code.

    python3 cell_types.py --runs <ckpt_root> --data-root <traj_root> \
        --conds s1 s2 s4 --out <cell_types.csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.field_stats import count_fields  # noqa: E402
from project5_symmetry.analysis.rate_maps import mean_rate_maps, spatial_information  # noqa: E402
from project5_symmetry.analysis.run_spectrum import identify, model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.hd_encodings import apply_hd  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

HD_MODES = ('full', 'parity', 'axis', 'const')
_HEADING_ANGLE = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])   # MiniGrid 0=E 1=S 2=W 3=N


@torch.no_grad()
def collect_hd(model, dataset, hd_mode, n_states, device, seed=0):
    """Hidden states, positions, and per-step headings, with the model's own HD encoding."""
    hs, ps, hd, total = [], [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, heading = dataset[int(idx)]
        act = apply_hd(act, hd_mode)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0).cpu().numpy()
        take = min(h.shape[0], n_states - total)
        hs.append(h[:take])
        ps.append(pos.numpy()[:h.shape[0]][:take])
        # heading is length T (one per action step); align to the T states after the first
        head = heading.numpy().astype(np.int64)
        hd.append(head[:h.shape[0]][:take])
        total += take
        if total >= n_states:
            break
    hidden = np.concatenate(hs, 0)
    positions = np.concatenate(ps, 0)
    headings = np.concatenate(hd, 0)
    assert np.all(positions == np.floor(positions)), 'positions are not integral'
    return hidden, positions.astype(np.int64), headings


def hd_rayleigh(hidden, headings):
    """Resultant-vector length of each unit's mean activity across the 4 headings.

    Activity is shifted to be non-negative per unit (subtract the min over headings) so the
    resultant is a proper directional-tuning index in [0, 1]; 1 = fires at one heading only.
    """
    U = hidden.shape[1]
    means = np.zeros((U, 4))
    for h in range(4):
        m = headings == h
        if m.any():
            means[:, h] = hidden[m].mean(0)
    means -= means.min(1, keepdims=True)                # non-negative directional tuning curve
    denom = means.sum(1)
    vec = means @ np.exp(1j * _HEADING_ANGLE)
    R = np.abs(vec) / np.where(denom > 1e-9, denom, 1.0)
    R[denom <= 1e-9] = 0.0
    return R


def border_frac(maps, occ, ring=2):
    """Fraction of each unit's positive rate-map mass in the outermost `ring` cells."""
    mask = occ == 0
    U = maps.shape[0]
    out = np.zeros(U)
    for u in range(U):
        m = np.clip(maps[u], 0, None).copy()
        m[mask] = 0.0
        tot = m.sum()
        if tot <= 1e-9:
            continue
        interior = m[ring:-ring, ring:-ring].sum()
        out[u] = 1.0 - interior / tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--si-thresh', type=float, default=0.3)
    ap.add_argument('--hd-thresh', type=float, default=0.3)
    ap.add_argument('--border-thresh', type=float, default=0.5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    ckpts = [p for p in ckpts if identify(p, torch.load(p, map_location='cpu',
             weights_only=False)['meta'])[0] in a.conds]
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt for {a.conds} under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    datasets, rows = {}, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(p, ck['meta'])
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        hidden, pos, headings = collect_hd(model, datasets[cond], hd, a.n_states, dev)
        maps, occ = mean_rate_maps(hidden, pos, a.arena)
        mask = occ == 0
        si = spatial_information(maps, occ)
        R = hd_rayleigh(hidden, headings)
        bf = border_frac(maps, occ)
        nf = np.array([count_fields(maps[u], mask)[0] for u in range(maps.shape[0])])

        is_place = si >= a.si_thresh
        is_hd = R >= a.hd_thresh
        is_border = bf >= a.border_thresh
        U = maps.shape[0]
        rows.append({
            'condition': cond, 'hd_mode': hd, 'seed': seed, 'n_units': U,
            'frac_place': float(is_place.mean()),
            'frac_hd': float(is_hd.mean()),
            'frac_border': float(is_border.mean()),
            'mean_spatial_info': float(np.mean(si)),
            'mean_hd_rayleigh': float(np.mean(R)),
            'mean_border_frac': float(np.mean(bf)),
            'mean_fields_place': float(nf[is_place].mean()) if is_place.any() else float('nan'),
        })
        r = rows[-1]
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/seed{seed}  place={r["frac_place"]:.2f} '
              f'hd={r["frac_hd"]:.2f} border={r["frac_border"]:.2f} '
              f'fields/place={r["mean_fields_place"]:.2f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"cond/hd":<12s} {"place":>6s} {"hd":>6s} {"border":>7s} {"fields/place":>13s}')
    for cond in a.conds:
        for hd in HD_MODES:
            sub = [r for r in rows if r['condition'] == cond and r['hd_mode'] == hd]
            if not sub:
                continue
            m = lambda k: float(np.nanmean([r[k] for r in sub]))
            print(f'{cond+"/"+hd:<12s} {m("frac_place"):>6.2f} {m("frac_hd"):>6.2f} '
                  f'{m("frac_border"):>7.2f} {m("mean_fields_place"):>13.2f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
