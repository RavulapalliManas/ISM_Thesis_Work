"""How the recurrent matrix moves during training, from every initialisation.

`pRNN_th` starts at W = gain*R + (1 - 1/tau)*I. The original authors tried Kaiming He,
found it "blows up", and replaced it with a uniform draw they labelled "goodbug". The
spectral radii explain the choice at step 0 -- Kaiming lands at rho = 1.94 once the leak
term is added, the uniform draw at 1.08 -- but they say nothing about where training goes.

This script reads every checkpoint of a run and reports, per (variant, seed, step):

    rho        spectral radius of W          (does training pull it to a common value?)
    fro        Frobenius norm of W           (how far the matrix has travelled)
    diag       mean of the diagonal          (does the leak term survive training?)
    off_std    std of the off-diagonal       (random part's scale)
    finite     0 if any weight is non-finite (a diverged run, reported not hidden)

The question this settles: does the initialisation leave a persistent trace in the trained
recurrent matrix, or does training forget where it started? If the latter, differences in
learning speed and map quality across inits are transient-dynamics effects, not endpoints.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def stats(W: np.ndarray) -> dict:
    finite = bool(np.all(np.isfinite(W)))
    if not finite:
        return {'rho': float('nan'), 'fro': float('nan'), 'diag': float('nan'),
                'off_std': float('nan'), 'finite': 0}
    off = W[~np.eye(W.shape[0], dtype=bool)]
    return {'rho': float(np.abs(np.linalg.eigvals(W)).max()),
            'fro': float(np.linalg.norm(W)),
            'diag': float(np.diagonal(W).mean()),
            'off_std': float(off.std()),
            'finite': 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    ckpts = sorted(Path(a.runs).rglob('ckpt_*.pt'))
    if not ckpts:
        raise SystemExit(f'no checkpoints under {a.runs}')
    rows = []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        meta = ck['meta']
        if str(ck['step']) == 'final':
            continue                       # duplicate of the last numbered checkpoint
        W = ck['model']['W'].numpy()
        r = {'variant': meta.get('variant', ''), 'layout': meta.get('layout', ''),
             'seed': meta.get('seed', -1), 'step': int(ck['step']), **stats(W)}
        rows.append(r)
        if i % 40 == 0:
            print(f'  {i}/{len(ckpts)}', flush=True)

    rows.sort(key=lambda r: (r['variant'], r['seed'], r['step']))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    diverged = [r for r in rows if not r['finite']]
    if diverged:
        v = sorted({r['variant'] for r in diverged})
        print(f'\n  DIVERGED (non-finite weights): {len(diverged)} checkpoints, variants {v}')

    print(f'\n{"variant":<14s} {"rho@0":>8s} {"rho@final":>10s} {"diag@0":>8s} {"diag@final":>11s}')
    for v in sorted({r['variant'] for r in rows}):
        sub = [r for r in rows if r['variant'] == v and r['finite']]
        if not sub:
            print(f'{v:<14s} {"all non-finite":>40s}')
            continue
        first = [r for r in sub if r['step'] == min(x['step'] for x in sub)]
        last = [r for r in sub if r['step'] == max(x['step'] for x in sub)]
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        print(f'{v:<14s} {m(first,"rho"):>8.3f} {m(last,"rho"):>10.3f} '
              f'{m(first,"diag"):>8.3f} {m(last,"diag"):>11.3f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
