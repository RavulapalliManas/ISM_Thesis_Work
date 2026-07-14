"""#2 Is folding literally the quotient map X -> X/G? Test the representational geometry, not
just decodability. For axis-encoded networks in C1 (unfolded) and C2 (folded), take the
position-conditioned mean hidden state, restrict to the C2 fundamental domain, and Procrustes-align.

Claims tested:
  (a) fold coincidence: in C2/axis, mean H(x) == mean H(R^2 x) over the domain (the two orbit
      halves are one point). Reported as median cosine between a cell and its 180-deg image.
  (b) quotient equality: the C2/axis domain-manifold matches the C1/axis domain-manifold up to a
      rigid transform (Procrustes disparity), i.e. the folded code represents the fundamental
      domain the same way the unfolded code does. Baselined against cross-seed disparity within
      C1/axis (the "same geometry" reference).

    python3 procrustes_quotient.py --ckpt-root <dir> --data-root <dir> --out <csv> [--n-traj 2000]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
from scipy.spatial import procrustes

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa
from project5_symmetry.analysis.run_phase_decoding import canonical, images, ARENA  # noqa
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa


def mean_by_position(H, pos):
    """Return {(x,y): mean hidden vector} over visited positions."""
    key = [tuple(int(v) for v in p) for p in pos]
    acc, cnt = {}, {}
    for k, h in zip(key, H):
        acc[k] = acc.get(k, 0) + h; cnt[k] = cnt.get(k, 0) + 1
    return {k: acc[k] / cnt[k] for k in acc}


def domain_matrix(mbp, arena):
    """Matrix of mean-H over the C2 fundamental domain (one rep per orbit), in a fixed order.

    The representative of each orbit is the CANONICAL position itself, not whichever member the
    trajectory happened to visit first. Using the first-visited member makes the choice depend on
    trajectory order, and since s1 and s2 use different datasets they can pick different members of
    the same orbit; for the unfolded C1 baseline the two members have genuinely different hidden
    states, so the "domain manifold" would become an arbitrary patchwork of the two arena halves.
    """
    order = sorted({tuple(canonical(np.array([[x, y]]), 'c2', arena)[0].astype(int))
                    for (x, y) in mbp} & set(mbp))
    M = np.array([mbp[c] for c in order])
    return M, order


def load_meanH(ck, cond, hd, seed, ds, dev, n_states):
    p = ck / 'hd_invariance' / cond / hd / f'seed_{seed:02d}' / 'ckpt_final.pt'
    if not p.exists():
        return None
    model = model_from_checkpoint(torch.load(p, map_location='cpu', weights_only=False), dev)
    H, pos = collect(model, ds, hd, n_states, dev)
    return mean_by_position(H, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-traj', type=int, default=2000)
    ap.add_argument('--n-states', type=int, default=15000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)

    ds = {}
    for c in ('s1', 's2'):
        ensure_data(c, a.data_root, a.n_traj, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    rows = []
    # per-seed mean-H over the domain for C1/axis and C2/axis
    dom = {('s1', s): None for s in a.seeds}; dom.update({('s2', s): None for s in a.seeds})
    for (cond, s) in list(dom):
        mbp = load_meanH(ck, cond, 'axis', s, ds[cond], dev, a.n_states)
        if mbp is None:
            print(f'  MISSING {cond}/axis/seed{s}'); continue
        M, order = domain_matrix(mbp, ARENA)
        dom[(cond, s)] = M
        # (a) fold coincidence for C2: cosine between H(x) and H(R^2 x)
        if cond == 's2':
            cos = []
            for (x, y) in list(mbp)[:400]:
                img = tuple(images(np.array([[x, y]]), 'c2', ARENA)[0, 1].astype(int))
                if img in mbp:
                    u, v = mbp[(x, y)], mbp[img]
                    cos.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
            print(f'  s2/axis/seed{s} fold coincidence: median cos(H(x),H(R2x)) = {np.median(cos):.3f}')
    # (b) Procrustes disparities
    def disp(A, B):
        n = min(len(A), len(B)); _, _, d = procrustes(A[:n], B[:n]); return d
    s0 = a.seeds[0]
    if dom[('s1', a.seeds[0])] is not None and dom[('s1', a.seeds[1])] is not None:
        base = disp(dom[('s1', a.seeds[0])], dom[('s1', a.seeds[1])])
        cross = disp(dom[('s1', a.seeds[0])], dom[('s2', a.seeds[0])])
        print(f'\n  Procrustes disparity  cross-seed C1/axis (reference) = {base:.4f}')
        print(f'  Procrustes disparity  C1/axis vs C2/axis (quotient)   = {cross:.4f}')
        rows.append({'cross_seed_C1_disparity': round(base, 4),
                     'C1_vs_C2_domain_disparity': round(cross, 4)})
    if rows:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
