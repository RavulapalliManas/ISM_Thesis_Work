"""Place-field repetition across FOUR identical compartments: the full Spiers (2015) paradigm.

The two-room readout (`run_compartments.py`) at four rooms. Because the rooms are related by
translation, head direction is invariant across them and a predictive code must fold all four
onto one map:

    room_gen    4-way decoder for WHICH room, whole local cells held out. Chance 0.25.
    room_seen   the same, random split, k-NN -- detects ANY room information. Chance 0.25.
    repetition  median across units of the mean pairwise correlation of the four per-room
                local rate maps. Folding -> 1 (fields repeat in every room).
    within_r2   ridge R^2 for position within a room -- the positive control.

Fold signature: room_gen ~ 0.25, room_seen ~ 0.25, repetition ~ 1, within_r2 high.

    python3 run_compartments4.py --runs <ckpts>/compartment4 --data-root <data>/compartment4 \
        --out <compartments4.csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_compartments import collect  # noqa: E402
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.compartment4 import (N_ROOMS, SIZE,  # noqa: E402
                                                        _room_origin)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def label_rooms(pos):
    """(room, local (i,j)) for states inside a compartment; -1 elsewhere. Translation relates
    the rooms, so local coords are just the offset from each room's origin."""
    room = np.full(len(pos), -1)
    loc = np.full((len(pos), 2), -1)
    r, c = pos[:, 1], pos[:, 0]
    for i in range(N_ROOMS):
        r0, c0 = _room_origin(i)
        inside = (r >= r0) & (r < r0 + SIZE) & (c >= c0) & (c < c0 + SIZE)
        room[inside] = i
        loc[inside] = np.stack([r[inside] - r0, c[inside] - c0], 1)
    return room, loc


def _balance(y, rng):
    classes = np.unique(y)
    n = min((y == cl).sum() for cl in classes)
    keep = np.concatenate([rng.choice(np.where(y == cl)[0], n, replace=False) for cl in classes])
    rng.shuffle(keep)
    return keep


def decode_room(H, room, loc, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    m = room >= 0
    H, y, L = H[m], room[m], loc[m]
    keep = _balance(y, rng)
    H, y, L = H[keep], y[keep], L[keep]
    groups = L[:, 0] * SIZE + L[:, 1]

    gen, r2 = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(H, y, groups=groups):
        sc = StandardScaler().fit(H[tr])
        Xtr, Xte = sc.transform(H[tr]), sc.transform(H[te])
        gen.append(LogisticRegression(max_iter=2000).fit(Xtr, y[tr]).score(Xte, y[te]))
        r2.append(Ridge(alpha=1.0).fit(Xtr, L[tr].astype(float)).score(Xte, L[te].astype(float)))

    seen = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(H):
        sc = StandardScaler().fit(H[tr])
        seen.append(KNeighborsClassifier(n_neighbors=15, weights='distance')
                    .fit(sc.transform(H[tr]), y[tr]).score(sc.transform(H[te]), y[te]))
    return float(np.mean(gen)), float(np.mean(seen)), float(np.mean(r2))


def repetition_index(H, room, loc, min_count=5):
    """Per-unit mean pairwise corr of the four per-room local rate maps. Folding -> 1."""
    U = H.shape[1]
    maps = np.full((N_ROOMS, SIZE, SIZE, U), np.nan)
    for rm in range(N_ROOMS):
        for i in range(SIZE):
            for j in range(SIZE):
                sel = (room == rm) & (loc[:, 0] == i) & (loc[:, 1] == j)
                if sel.sum() >= min_count:
                    maps[rm, i, j] = H[sel].mean(0)
    flat = maps.reshape(N_ROOMS, -1, U)
    ok = np.isfinite(flat).all(0).all(1)               # local cells present in all rooms
    flat = flat[:, ok, :]
    if flat.shape[1] < 8:
        return float('nan'), 0
    top = np.argsort(H.var(0))[-100:]
    rs = []
    for u in top:
        pair = []
        for a in range(N_ROOMS):
            for b in range(a + 1, N_ROOMS):
                x, y = flat[a, :, u], flat[b, :, u]
                if x.std() > 1e-8 and y.std() > 1e-8:
                    pair.append(np.corrcoef(x, y)[0, 1])
        if pair:
            rs.append(np.mean(pair))
    return (float(np.median(rs)) if rs else float('nan')), int(flat.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/compartment4')
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds, rows = None, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        seed = ck['meta']['seed']
        if ds is None:
            ds = TrajectoryDataset(str(Path(a.data_root) / 'parallel'))
        model = model_from_checkpoint(ck, torch.device('cpu'))
        H, P = collect(model, ds, a.n_states, seed)
        room, loc = label_rooms(P)
        gen, seen, within = decode_room(H, room, loc, seed=seed)
        rep, ncells = repetition_index(H, room, loc)
        rows.append({'mode': 'parallel4', 'seed': seed, 'chance': 1.0 / N_ROOMS,
                     'room_gen': gen, 'room_seen': seen, 'repetition': rep,
                     'within_r2': within, 'n_cells': ncells,
                     'n_states_in_rooms': int((room >= 0).sum())})
        print(f'  [{i}/{len(ckpts)}] seed{seed}  room_gen={gen:.4f} room_seen={seen:.4f}  '
              f'repetition={rep:+.4f}  within_r2={within:+.3f}  ({ncells} cells)', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    m = lambda k: float(np.nanmean([r[k] for r in rows]))
    print(f'\nparallel4 (chance {1.0/N_ROOMS:.2f}): room_gen={m("room_gen"):.4f} '
          f'room_seen={m("room_seen"):.4f} repetition={m("repetition"):.4f} '
          f'within_r2={m("within_r2"):.3f}')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
