"""Harland et al. (2017), in silico: the head-direction lesion, run in their own 2x2 design.

This is the sharpest biological test in the paper, because it is not a re-description of our own
result in someone else's vocabulary -- it is an experiment that has ALREADY BEEN RUN IN RATS, whose
outcome the quotient law predicts with no free parameter, and whose outcome is otherwise a puzzle.

THE RAT EXPERIMENT. Harland, Grieves, Bett, Stentiford, Wood & Dudchenko (Curr Biol 27:2706, 2017)
lesioned the lateral mammillary nuclei -- which abolishes the head-direction signal outright (Bassett
et al. 2007 find 0 of 41 ADN cells still directional; Blair et al. 1999 find tuning curves
permanently flat) -- in rats with established place fields, and recorded in a multicompartment box
whose compartments were arranged either in PARALLEL (related by a translation) or RADIALLY (related
by a rotation). Their result, which reads as a paradox:

    parallel (translation):  sham 65% vs lesion 63%   t(10) = 1.06,  p = 0.31    NOTHING HAPPENS
    radial   (rotation):     repetition RETURNS        t(5.2) = 3.30, p = 0.021
    interaction:             F(1,10) = 13.60, p < 0.005, eta^2 = 0.58

Why would destroying the compass leave one arrangement untouched and fold the other? Because a
compass is TRANSLATION-INVARIANT BY CONSTRUCTION: heading does not change when you walk sideways, so
the compass was never the thing keeping the parallel compartments apart, and destroying it cannot
change them. The compass IS, however, the only thing that distinguishes a rotated compartment from
its original -- so destroy it and the rotational symmetry closes and the map folds. The lesion's
effect is not on the map; it is on the GROUP.

That is the quotient law, stated as a causal intervention. Nothing else predicts an interaction.

OUR VERSION. Two observationally identical compartments related by a group element that acts on
heading in exactly those two ways (environments/compartment_arenas.py):

    translation (parallel)   (x, h) -> (x + delta, h)      compass invariant   -> already folded
    rotation    (radial)     (x, h) -> (R x,     h + 2)    compass equivariant -> held apart

Both networks TRAINED with an intact compass, as Harland's rats developed with one. We then take the
compass away at test time on a fraction p of steps, which is the right model of an adult lesion (a
network retrained without a compass is not a lesioned animal -- it is an animal that grew up
blindfolded, and had 80,000 steps to learn a heading-invariant code instead).

PREDICTIONS, BEFORE THE RUN:

    translation:  repetition is ALREADY high at p = 0 and stays flat. The compass never broke this
                  symmetry, so removing it must do nothing. FLAT LINE.  [Harland: 65% -> 63%, n.s.]
    rotation:     repetition is LOW at p = 0 and RISES with dose, approaching the translation curve.
                  [Harland: repetition returns, p = 0.021]
    interaction:  slope(rotation) >> slope(translation) ~ 0.
    within_r2:    stays high in BOTH, at every dose. This is the control that says the lesion has
                  not simply destroyed the map: a folded code still knows where in a room it is, it
                  has only stopped knowing WHICH room. Without this, the whole thing is degradation.

The reachable null is real and it is the translation arm: if our lesion were merely a generalised
insult to the code, repetition would move there too. It must not.

    PYTHONPATH=. python3 analysis/lesion_compartments.py \
        --ckpt-root <dir> --data-root <dir> --out Report/data/lesion_compartments.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_compartments import (decode_room, label_rooms,  # noqa: E402
                                                         repetition_index)
from project5_symmetry.environments.compartment_arenas import make_compartment_env  # noqa: E402
from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

T = 200


def ensure_data(mode, data_root, n_traj, workers):
    """The factory must be passed explicitly, or the workers silently downcast a CompartmentArena
    to a plain SymmetryArena and the interior walls vanish."""
    d = Path(data_root) / mode
    generate_dataset(make_compartment_env(mode), n_traj=n_traj, T=T, out_dir=str(d),
                     n_workers=workers, desc=f'{mode} trajectories',
                     env_factory=make_compartment_env, factory_kwargs={'mode': mode})
    return d


@torch.no_grad()
def collect_lesioned(model, ds, p_lesion, n_states, seed=0, mode='silence'):
    """Hidden states with the compass lesioned on a fraction p of steps.

    `silence` zeroes the heading block: the compass is ABSENT. This is what a lesion produces --
        Bassett et al. (2007) find 0 of 41 ADN cells still directional after LMN lesion, i.e. the
        signal carries no information. The network falls back on the egocentric view.
    `randomize` replaces the heading with a random one: the compass LIES, loudly. Because the
        observation is first-person (get_frame(agent_pov=True)) and therefore already carries
        heading, a random compass puts the two channels in conflict -- a state the network never met
        in training. It drives the network off its training manifold, and the map does not fold, it
        DIES: within-room R^2 goes negative. Kept only as the control that shows this.

    The compartment models were trained on the raw 5-d action, so no HD re-encoding is applied.
    """
    rng = np.random.default_rng(seed)
    hs, ps, tot = [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(ds), generator=g):
        obs, act, pos, _ = ds[int(idx)]
        act = act.clone()
        if p_lesion > 0:
            hit = torch.from_numpy(rng.random(act.shape[0]) < p_lesion)
            if hit.any():
                if mode == 'silence':
                    act[hit, 1:] = 0.0
                elif mode == 'randomize':
                    block = torch.zeros((int(hit.sum()), 4), dtype=act.dtype)
                    block[torch.arange(int(hit.sum())),
                          torch.from_numpy(rng.integers(0, 4, int(hit.sum())))] = 1.0
                    act[hit, 1:] = block
                else:
                    raise ValueError(f'unknown lesion mode {mode!r}')
        _, h, _ = model(obs.unsqueeze(0), act.unsqueeze(0))
        h = h.squeeze(0).numpy()
        take = min(h.shape[0], n_states - tot)
        hs.append(h[:take]); ps.append(pos.numpy()[:h.shape[0]][:take])
        tot += take
        if tot >= n_states:
            break
    return np.concatenate(hs, 0), np.concatenate(ps, 0).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--modes', nargs='+', default=['translation', 'rotation'])
    ap.add_argument('--doses', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
    ap.add_argument('--lesion-modes', nargs='+', default=['silence', 'randomize'],
                    choices=['silence', 'randomize'])
    ap.add_argument('--n-traj', type=int, default=800)
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
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
                torch.load(p, map_location='cpu', weights_only=False), torch.device('cpu'))
            for lmode in a.lesion_modes:
                for dose in a.doses:
                    H, pos = collect_lesioned(model, ds[mode], dose, a.n_states, seed=s,
                                              mode=lmode)
                    room, loc = label_rooms(pos, mode)
                    keep = room >= 0
                    H, room, loc = H[keep], room[keep], loc[keep]
                    gen, seen, within = decode_room(H, room, loc, seed=s)
                    rep, ncells = repetition_index(H, room, loc)
                    rows.append({'mode': mode, 'seed': s, 'lesion_mode': lmode, 'dose': dose,
                                 'repetition': round(float(rep), 4),
                                 'room_gen': round(float(gen), 4),
                                 'room_seen': round(float(seen), 4),
                                 'within_r2': round(float(within), 4),
                                 'n_cells': int(ncells)})
                    print(f"  {mode:<11s}/s{s:02d} {lmode:<9s} dose={dose:.2f}  "
                          f"repetition={rep:+.4f}  within_r2={within:+.3f}", flush=True)

    if not rows:
        raise SystemExit('no checkpoints found')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
