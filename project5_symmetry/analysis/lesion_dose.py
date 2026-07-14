"""An in-silico head-direction lesion, with a dose, in an adult network.

WHY THIS AND NOT RETRAINING. Harland et al. (2017) lesioned the lateral mammillary nuclei of ADULT
rats that already had established place fields, and then recorded. The right model of that is not a
network retrained without a compass; it is a network that DEVELOPED with a good compass and then
has it taken away. So we take the `full`-encoding networks and corrupt the heading at test time on
a fraction p of steps, replacing it with a random heading. p = 0 is the sham; p = 1 is a compass
that carries no information at all.

WHAT IT BUYS. Two things nothing else in the paper can give.

1. A DOSE-RESPONSE. Harland reports, without quantifying it, that "animals with larger lesions
   tended to show more place field repetition in the radial compartments". That is the single most
   valuable unreported analysis in their paper, and it is exactly what this sweep predicts. We can
   hand the field a curve.

2. IT RECONCILES THE TWO LESION STUDIES IN ONE EXPERIMENT. Harland (identical compartments, no
   polarising cue, i.e. a symmetric world) reports lower spatial information AND field repetition.
   Calton et al. (2003) (a cue-controlled cylinder, i.e. an asymmetric world) reports NO change in
   field number, size, rate or sparsity, but lower coherence. Under the quotient law those are the
   same lesion in two different worlds: removing the compass degrades the map wherever it is
   removed, but it can only FOLD a map onto a symmetry that exists. Running the identical dose
   sweep in the C1 arena (nothing to fold onto) and the C2 arena (a symmetry to fold onto) should
   reproduce Calton and Harland respectively, with no parameter free to fit.

PREDICTIONS, before the run:
    C1 and C2 alike: spatial information falls, sparsity rises, coherence falls with p.
    C2 only:         orbit phase collapses toward 0.5 and fields multiply.
    C1:              fields multiply much less -- there is no symmetry to fold onto.

    PYTHONPATH=. python3 analysis/lesion_dose.py --ckpt-root <dir> --data-root <dir> \
        --out Report/data/lesion_dose.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_phase_decoding import orbit_and_phase, _balance, ARENA  # noqa: E402
from project5_symmetry.analysis.cell_properties import properties  # noqa: E402
from project5_symmetry.environments.hd_encodings import apply_hd  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def collect_lesioned(model, dataset, p_lesion, n_states, device, seed=0, mode='silence'):
    """Hidden states with the compass lesioned on a fraction p of steps.

    THE TWO LESIONS ARE NOT THE SAME, and which one we use decides whether this models an animal.

    `randomize` replaces the heading one-hot with a uniformly random one. In the mutual-information
        sense the compass then says nothing -- but it does not say nothing, it says something FALSE,
        and it says it loudly. The network still receives a valid egocentric view (the observation is
        first-person: get_frame(agent_pov=True), and the same place seen from four headings gives
        four different images), so a randomized compass puts the two channels in direct conflict, a
        state the network never met in training. It is driven off its training manifold, and the map
        does not fold so much as die: within-room R^2 goes NEGATIVE at high dose.

    `silence` zeroes the heading block instead. The compass is then absent, not lying, which is what
        a lesion actually produces -- Bassett et al. (2007) find 0 of 41 ADN cells still directional
        after LMN lesion, i.e. the signal carries no information; it does not carry wrong
        information. The network must now fall back on the egocentric view.

    That fallback is the whole point, because it is Calton et al.'s (2003) own proposed mechanism,
    verbatim: "the HD system serves to catalog the different 'local views' of a given place into a
    cohesive nondirectional spatial representation. Without an intact HD system, the convergence of
    directionally dependent sensory information onto the place cell system could result in
    'directional fragmentation' of the place field, giving the appearance of a place field modulated
    by the directional heading of the animal." Our architecture IS that: a compass that binds
    egocentric views into an allocentric code. Silence it and the views should fragment.

    PREDICTION for `silence`, stated before the run:
        within_r2 stays POSITIVE (the view still localises; the map survives) -- the control passes;
        the fold still appears wherever a symmetry exists, because the egocentric view is IDENTICAL
            at x and g.x and so cannot break the symmetry either;
        in-field directional information RISES, where under `randomize` it fell to zero.
            [Calton: 0.25 -> 0.48, F(2,91) = 5.96, p < 0.01]
    """
    rng = np.random.default_rng(seed)
    hs, ps, hd, total = [], [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, heading = dataset[int(idx)]
        act = act.clone()
        if p_lesion > 0:
            T = act.shape[0]
            hit = torch.from_numpy(rng.random(T) < p_lesion)
            if hit.any():
                if mode == 'silence':
                    act[hit, 1:] = 0.0
                elif mode == 'randomize':
                    rand_h = torch.from_numpy(rng.integers(0, 4, int(hit.sum())))
                    block = torch.zeros((int(hit.sum()), 4), dtype=act.dtype)
                    block[torch.arange(len(rand_h)), rand_h] = 1.0
                    act[hit, 1:] = block
                else:
                    raise ValueError(f'unknown lesion mode {mode!r}')
        act = apply_hd(act, 'full')
        with torch.no_grad():
            _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0).cpu().numpy()
        take = min(h.shape[0], n_states - total)
        hs.append(h[:take])
        ps.append(pos.numpy()[:h.shape[0]][:take])
        hd.append(heading.numpy().astype(np.int64)[:h.shape[0]][:take])
        total += take
        if total >= n_states:
            break
    return (np.concatenate(hs, 0), np.concatenate(ps, 0).astype(np.int64),
            np.concatenate(hd, 0))


def phase_acc(H, pos, group, seed=0):
    """Orbit-phase decoding, orbits held out across folds (as everywhere else in the paper)."""
    rng = np.random.default_rng(seed)
    orbit, phase = orbit_and_phase(pos, group, ARENA)
    keep = _balance(phase, rng)
    Hk, y, g = H[keep], phase[keep], orbit[keep]
    acc = []
    for tr, te in GroupKFold(5).split(Hk, y, groups=g):
        sc = StandardScaler().fit(Hk[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Hk[tr]), y[tr])
        acc.append(clf.score(sc.transform(Hk[te]), y[te]))
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2'])
    ap.add_argument('--doses', type=float, nargs='+',
                    default=[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(6)))
    ap.add_argument('--lesion-modes', nargs='+', default=['silence', 'randomize'],
                    choices=['silence', 'randomize'])
    ap.add_argument('--n-traj', type=int, default=800)
    ap.add_argument('--n-states', type=int, default=30_000)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)

    ds = {}
    for c in a.conds:
        ensure_data(c, a.data_root, a.n_traj, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    rows = []
    for cond in a.conds:
        for s in a.seeds:
            p = ck / 'hd_invariance' / cond / 'full' / f'seed_{s:02d}' / 'ckpt_final.pt'
            if not p.exists():
                continue
            model = model_from_checkpoint(
                torch.load(p, map_location='cpu', weights_only=False), dev)
            for lmode in a.lesion_modes:
                for dose in a.doses:
                    H, pos, hd = collect_lesioned(model, ds[cond], dose, a.n_states, dev,
                                                  seed=s, mode=lmode)
                    r = {'condition': cond, 'seed': s, 'lesion_mode': lmode, 'dose': dose}
                    r.update({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in properties(H, pos, hd).items()})
                    r['phase_acc'] = round(phase_acc(H, pos, 'c2', seed=s), 4)
                    rows.append(r)
                    print(f"  {cond}/s{s:02d} {lmode:<9s} dose={dose:.2f}  "
                          f"phase={r['phase_acc']:.3f} SI={r['spatial_info']:.3f} "
                          f"dirF={r['dir_info_field']:.3f} fields={r['n_fields']:.2f}",
                          flush=True)

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
