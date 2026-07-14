"""Spiers' anisotropic decoding error: the fold, converted into a cost.

Everything else in this paper measures the fold as a property of the code. This measures what the
fold COSTS an animal that has to use it, which is the currency an experimentalist reads in.

Spiers et al. (2015), decoding position from CA1 in four identical compartments in a row, found the
reconstruction error was strongly ANISOTROPIC: 20.18 +/- 0.86 bins along the axis on which the
compartments repeat, against 8.57 +/- 0.51 bins across it (t225 = 11.30). And when they collapsed
the compartments onto a single frame and re-decoded, the anisotropy REVERSED (3.44 vs 8.26). That
pair of facts IS the quotient: the code knows where you are inside a room and does not know which
room, so error piles up along one axis and one axis only.

Our two-compartment maze has the same geometry. Room B sits 13 rows below room A in the same
columns (A0 = (2,9), B0 = (15,9)), so the repetition axis is the ROW axis and the orthogonal axis
is the COLUMN axis. We decode the cell the agent occupies from the population state and split the
error into those two components.

PREDICTIONS, before the run:
    translation (the compass cannot break it, so the map folds):
        error ALONG the repetition axis is LARGE -- around the 13-row room offset, because the
        decoder puts the animal in the wrong room at the right place within it -- and error ACROSS
        it is SMALL. Strongly anisotropic.
    rotation (the compass breaks it, so the map does not fold):
        both errors small. Isotropic.
    collapsed (decode the WITHIN-ROOM position instead):
        the anisotropy disappears, because within a room the code is intact. This is the control
        that shows the error is the fold and not a bad decoder.

    PYTHONPATH=. python3 analysis/anisotropic_decoding.py --ckpt-root <dir> --data-root <dir> \
        --out Report/data/anisotropic_decoding.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_compartments import collect, label_rooms  # noqa: E402
from project5_symmetry.environments.compartment_arenas import (A0, B0,  # noqa: E402
                                                               make_compartment_env)
from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

ROW_OFFSET = B0[0] - A0[0]          # 13: how far apart the two rooms are along the repetition axis
T = 200                             # trajectory length, as in the other compartment analyses


def ensure_data(mode, data_root, n_traj, workers):
    """The compartment trajectories were generated on the pod and did not survive; regenerate.

    The factory has to be passed explicitly: the workers rebuild the environment from a recipe, and
    without it a CompartmentArena is silently downcast to a plain SymmetryArena and its walls
    vanish. Generation is seeded by trajectory index, so this is reproducible.
    """
    d = Path(data_root) / mode
    generate_dataset(make_compartment_env(mode), n_traj=n_traj, T=T, out_dir=str(d),
                     n_workers=workers, desc=f'{mode} trajectories',
                     env_factory=make_compartment_env, factory_kwargs={'mode': mode})
    return d


def decode_error(H, targets, seed=0):
    """Classify the cell the agent occupies, then return the per-state error vector.

    A classifier over discrete cells (rather than a regression onto coordinates) is what Spiers
    used, and it is the only kind of decoder that CAN make the mistake we are looking for: a
    regression would hedge between the two rooms and land in the corridor, whereas a classifier
    commits to a cell, so a folded code puts the animal confidently in the wrong room.
    """
    key = {t: i for i, t in enumerate(sorted(set(map(tuple, targets))))}
    inv = {i: t for t, i in key.items()}
    y = np.array([key[tuple(t)] for t in targets])
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(H[tr])
    clf = LogisticRegression(max_iter=600, C=1.0, n_jobs=-1).fit(sc.transform(H[tr]), y[tr])
    pred = clf.predict(sc.transform(H[te]))
    P = np.array([inv[i] for i in pred], float)
    T = np.array([inv[i] for i in y[te]], float)
    return np.abs(P - T)                       # (n, 2): |error| along each axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--modes', nargs='+', default=['translation', 'rotation'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
    ap.add_argument('--n-traj', type=int, default=800)
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)

    ds = {}
    for m in a.modes:
        ensure_data(m, a.data_root, a.n_traj, a.threads)
        ds[m] = TrajectoryDataset(str(Path(a.data_root) / m))

    rows = []
    for mode in a.modes:
        for s in a.seeds:
            p = ck / 'multi' / 'compartment' / mode / f'seed_{s:02d}' / 'ckpt_final.pt'
            if not p.exists():
                continue
            model = model_from_checkpoint(
                torch.load(p, map_location='cpu', weights_only=False), dev)
            H, pos = collect(model, ds[mode], a.n_states, seed=s)
            room, loc = label_rooms(pos, mode)
            keep = room >= 0
            H, pos, room, loc = H[keep], pos[keep], room[keep], loc[keep]

            # (i) the full arena: which CELL are you in? The decoder may put you in the wrong room.
            full_t = np.stack([pos[:, 1], pos[:, 0]], 1)          # (row, col)
            e_full = decode_error(H, full_t, seed=s)

            # (ii) collapsed: which cell WITHIN a room are you in? Room identity is discarded, so a
            # folded code should do just as well as an unfolded one. This is the control.
            e_coll = decode_error(H, loc, seed=s)

            r = {'mode': mode, 'seed': s, 'n_states': int(keep.sum()),
                 'err_along_full': round(float(e_full[:, 0].mean()), 3),
                 'err_across_full': round(float(e_full[:, 1].mean()), 3),
                 'err_along_collapsed': round(float(e_coll[:, 0].mean()), 3),
                 'err_across_collapsed': round(float(e_coll[:, 1].mean()), 3),
                 'frac_wrong_room': round(float((e_full[:, 0] > ROW_OFFSET / 2).mean()), 3)}
            r['anisotropy_full'] = round(r['err_along_full'] - r['err_across_full'], 3)
            r['anisotropy_collapsed'] = round(
                r['err_along_collapsed'] - r['err_across_collapsed'], 3)
            rows.append(r)
            print(f"  {mode}/s{s:02d}  full: along={r['err_along_full']:.2f} "
                  f"across={r['err_across_full']:.2f}  | collapsed: along="
                  f"{r['err_along_collapsed']:.2f} across={r['err_across_collapsed']:.2f}  "
                  f"| wrong room {100*r['frac_wrong_room']:.0f}%", flush=True)

    if not rows:
        raise SystemExit('no checkpoints found')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} networks)')


if __name__ == '__main__':
    main()
