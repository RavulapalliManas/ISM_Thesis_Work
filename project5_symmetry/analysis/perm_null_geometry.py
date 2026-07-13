"""#1 permutation null for orbit-phase decoding + #2 Procrustes test that folding is the
quotient map. CPU-only, from local checkpoints; regenerates trajectory data locally.

    python3 perm_null_geometry.py --ckpt-root <dir> --data-root <dir> --out-dir <dir> \
        [--n-perm 500] [--n-traj 1500] [--n-states 12000] [--quick]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa
from project5_symmetry.analysis.run_phase_decoding import (orbit_and_phase, canonical,  # noqa
                                                           _balance, decode, ARENA)
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa


def perm_null(H, pos, group, n_perm, seed=0, arena=ARENA):
    rng = np.random.default_rng(seed)
    orbit, phase = orbit_and_phase(pos, group, arena)
    keep = _balance(phase, rng)
    Hk, y, g = H[keep], phase[keep], orbit[keep]

    def one(yy):
        acc = []
        for tr, te in GroupKFold(5).split(Hk, yy, groups=g):
            sc = StandardScaler().fit(Hk[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Hk[tr]), yy[tr])
            acc.append(clf.score(sc.transform(Hk[te]), yy[te]))
        return float(np.mean(acc))

    obs = one(y)
    null = np.array([one(rng.permutation(y)) for _ in range(n_perm)])
    p_above = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(null.mean()), float(null.std()), p_above


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--n-traj', type=int, default=1500)
    ap.add_argument('--n-states', type=int, default=12000)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt_root)

    # targets: (cond, hd, group). folded (expect at-chance) + one non-folded control.
    targets = [('s2', 'axis', 'c2'), ('s2', 'const', 'c2'), ('s2', 'parity', 'c2'),
               ('s4', 'const', 'c4'), ('s4', 'axis', 'c4')]
    if a.quick:
        targets = [('s2', 'axis', 'c2'), ('s2', 'parity', 'c2')]
        a.n_perm = 40

    conds = sorted({c for c, _, _ in targets})
    ds = {}
    for c in conds:
        ensure_data(c, a.data_root, a.n_traj, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    rows = []
    for cond, hd, grp in targets:
        p = ck / 'hd_invariance' / cond / hd / 'seed_00' / 'ckpt_final.pt'
        if not p.exists():
            print(f'  MISSING {p}'); continue
        model = model_from_checkpoint(torch.load(p, map_location='cpu', weights_only=False), dev)
        H, pos = collect(model, ds[cond], hd, a.n_states, dev)
        obs, nmean, nsd, p_above = perm_null(H, pos, grp, a.n_perm)
        rows.append({'cond': cond, 'hd': hd, 'group': grp, 'observed': round(obs, 4),
                     'null_mean': round(nmean, 4), 'null_sd': round(nsd, 4),
                     'chance': round(1 / (2 if grp == 'c2' else 4), 3), 'p_above_null': round(p_above, 4)})
        print(f'  {cond}/{hd} ({grp}): obs={obs:.4f} null={nmean:.4f}+/-{nsd:.4f} '
              f'p_above={p_above:.4f}', flush=True)

    with open(out / 'perm_null.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {out / "perm_null.csv"}')


if __name__ == '__main__':
    main()
