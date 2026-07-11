"""Place-field repetition across identical compartments: the in-silico Grieves et al. (2016).

Two closed, observationally identical 6x6 compartments, related by a group element. The ONLY
difference between conditions is how that element acts on heading:

    translation (parallel)   (x, h) -> (x + delta, h)     compass invariant
    rotation    (radial)     (x, h) -> (R x,     h + 2)   compass equivariant

Grieves, Jenkins, Harland, Wood & Dudchenko (Hippocampus 2016) recorded the same four
compartments in both arrangements: "place cells often exhibited repeated fields" in parallel,
"significantly less place field repetition was apparent" in radial (all p < 0.0001), surviving
removal of the orientation landmark, and attributed to "directional information derived from the
animal's self-motion".

Readouts, per network:

  room_gen      decoder for WHICH compartment, with whole local cells held out. Detects a
                CONSISTENT room direction, i.e. rate remapping. Chance 0.5.
  room_seen     the same decoder with a random split. Detects ANY room information, including
                global remapping (where room identity is written differently at each cell).
  repetition    median across units of corr(rate map in A, rate map in B) over the 36 local
                cells, with B's cells mapped through the group element. This is the quantity
                Grieves et al. measured. Repetition -> 1 means fields repeat.
  within_r2     ridge R^2 for position WITHIN the compartment. The positive control: a folded
                code still knows where in the room it is, it just does not know which room.

    fold          room_gen ~ 0.5   room_seen ~ 0.5   repetition ~ 1
    global remap  room_gen <= 0.5  room_seen ~ 1.0   repetition ~ 0
    rate remap    room_gen ~ 1.0   room_seen ~ 1.0   repetition ~ 1

Prediction: translation (parallel) folds; rotation (radial) lifts. Both keep within_r2 high, or
the network has simply stopped coding space.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.compartment_arenas import (A0, B0, SIZE,  # noqa: E402
                                                               group, make_compartment_env)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


@torch.no_grad()
def collect(model, ds, n_states, seed=0):
    hs, ps, tot = [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(ds), generator=g):
        obs, act, pos, _ = ds[int(idx)]
        _, h, _ = model(obs.unsqueeze(0), act.unsqueeze(0))
        h = h.squeeze(0).numpy()
        take = min(h.shape[0], n_states - tot)
        hs.append(h[:take]); ps.append(pos.numpy()[:h.shape[0]][:take])
        tot += take
        if tot >= n_states:
            break
    return np.concatenate(hs, 0), np.concatenate(ps, 0).astype(int)


def label_rooms(pos, mode):
    """(room, local_i, local_j) for states inside a compartment; -1 elsewhere.

    `pos` is MiniGrid (col, row). Room B's local coords are pulled back through the group, so
    local (i, j) always names the SAME place within a compartment.
    """
    g, _ = group(mode)
    room = np.full(len(pos), -1)
    loc = np.full((len(pos), 2), -1)
    r, c = pos[:, 1], pos[:, 0]
    inA = (r >= A0[0]) & (r < A0[0] + SIZE) & (c >= A0[1]) & (c < A0[1] + SIZE)
    inB = (r >= B0[0]) & (r < B0[0] + SIZE) & (c >= B0[1]) & (c < B0[1] + SIZE)
    room[inA] = 0
    loc[inA] = np.stack([r[inA] - A0[0], c[inA] - A0[1]], 1)
    room[inB] = 1
    bi, bj = r[inB] - B0[0], c[inB] - B0[1]
    gi, gj = np.array([g(int(i), int(j)) for i, j in zip(bi, bj)]).T if inB.any() else (bi, bj)
    loc[inB] = np.stack([gi, gj], 1)
    return room, loc


def _balance(y, rng):
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)])
    rng.shuffle(keep)
    return keep


def decode_room(H, room, loc, n_splits=5, seed=0):
    """Which compartment? TWO decoders, because they detect different things.

    `gen`  holds out whole local cells, so the decoder must find a room-identity direction that
           generalises to places it has never seen. A globally remapped code has none -- room
           identity is written differently at every cell -- so `gen` reads at or below chance.
           What `gen` detects is a CONSISTENT room signal, i.e. rate remapping.

    `seen` splits at random, so the same local cells appear in train and test, and uses a
           NEAREST-NEIGHBOUR classifier. A linear one is the wrong instrument here: a globally
           remapped code writes room identity differently at every cell, so no single hyperplane
           separates the rooms (a logistic decoder tops out near 0.75 on a synthetic global
           remap). k-NN detects ANY room information, however it is written.

    fold          gen ~ 0.5   seen ~ 0.5   (the code cannot tell the rooms apart at all)
    global remap  gen <= 0.5  seen ~ 1.0   (room is decodable, but only where you have been)
    rate remap    gen ~ 1.0   seen ~ 1.0   (a consistent gain direction, decodable anywhere)

    Also returns the within-room position R^2 -- the positive control.
    """
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier
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
    """Per-unit corr(map in A, map in B) over the 36 local cells. Folding -> 1."""
    maps = np.full((2, SIZE, SIZE, H.shape[1]), np.nan)
    for rm in (0, 1):
        for i in range(SIZE):
            for j in range(SIZE):
                sel = (room == rm) & (loc[:, 0] == i) & (loc[:, 1] == j)
                if sel.sum() >= min_count:
                    maps[rm, i, j] = H[sel].mean(0)
    A = maps[0].reshape(-1, H.shape[1])
    B = maps[1].reshape(-1, H.shape[1])
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    if A.shape[0] < 8:
        return float('nan'), 0
    var = H.var(0)
    top = np.argsort(var)[-100:]
    rs = []
    for u in top:
        a, b = A[:, u], B[:, u]
        if a.std() > 1e-8 and b.std() > 1e-8:
            rs.append(np.corrcoef(a, b)[0, 1])
    return (float(np.median(rs)) if rs else float('nan')), int(A.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/compartment')
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds, rows = {}, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        mode, seed = ck['meta']['mode'], ck['meta']['seed']
        if mode not in ds:
            ds[mode] = TrajectoryDataset(str(Path(a.data_root) / mode))
        model = model_from_checkpoint(ck, torch.device('cpu'))
        H, P = collect(model, ds[mode], a.n_states, seed)
        room, loc = label_rooms(P, mode)
        gen, seen, within = decode_room(H, room, loc, seed=seed)
        rep, ncells = repetition_index(H, room, loc)
        rows.append({'mode': mode, 'seed': seed, 'room_gen': gen, 'room_seen': seen,
                     'repetition': rep, 'within_r2': within, 'n_cells': ncells,
                     'n_states_in_rooms': int((room >= 0).sum())})
        print(f'  [{i}/{len(ckpts)}] {mode}/seed{seed}  room_gen={gen:.4f} room_seen={seen:.4f}  '
              f'repetition={rep:+.4f}  within_r2={within:+.3f}  ({ncells} cells)', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"mode":<13s} {"room_gen":>9s} {"room_seen":>10s} {"repetition":>11s} {"within_r2":>10s}')
    for mode in sorted({r['mode'] for r in rows}):
        sub = [r for r in rows if r['mode'] == mode]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f'{mode:<13s} {m("room_gen"):>9.4f} {m("room_seen"):>10.4f} '
              f'{m("repetition"):>11.4f} {m("within_r2"):>10.3f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
