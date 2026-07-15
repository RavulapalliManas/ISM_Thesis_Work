"""The DIRECTIONAL shadow of the fold: multidirectional tuning, derived rather than cited.

WHY THIS EXISTS. Two experimental papers report a phenomenon we have so far only CITED, as evidence
that a G-invariant compass exists in cortex and is not a device we invented to make a theorem land:

    Zhang, Grieves & Jeffery (2022) recorded directionally tuned retrosplenial neurons in rats
        exploring multicompartment environments of different global rotational symmetry, and found
        tuning curves carrying the environment's OWN symmetry: unidirectional in a onefold circular
        arena, BIDIRECTIONAL (two peaks 180 deg apart) in a twofold two-box, TETRADIRECTIONAL (four
        peaks at 90 deg) in a fourfold four-box.

    LaChance & Hasselmo (2024) duplicated a cue card so two identical cards faced each other across a
        square, and postrhinal and retrosplenial head-direction cells became BIDIRECTIONAL, while
        entorhinal head-direction cells did not, and grid maps did not change.

Both report a phenomenon. Under the quotient law they are THEOREMS, and this script measures them.

--------------------------------------------------------------------------------------------------
THEOREM 1 (the global tuning curve carries the group). Folding means the code is G-invariant:
u(g.x, g.h) = u(x, h). Occupancy in a symmetric arena is G-invariant by construction. Then the
position-marginalised direction tuning curve T_u(h) = E_{x ~ p(x|h)} [u(x, h)] obeys

    T_u(g.h) = T_u(h)      for every g in G.

(Substitute x = g.x'; p(g.x'|g.h) = p(x'|h) since occupancy is G-invariant and g is an isometry;
then apply the invariance of u.) A FOLDED CODE IS FORCED TO HAVE DIRECTION TUNING WITH THE ARENA'S
SYMMETRY. On a 4-way compass the tuning curve has exactly two resolvable harmonics, and G = C_n
annihilates every harmonic that is not a multiple of n:

    C1 arena   nothing annihilated              -> unidirectional allowed      (m1 survives)
    C2 arena   the ODD harmonic is annihilated  -> two peaks 180 deg apart     (m1 -> 0, m2 lives)
    C4 arena   both resolvable harmonics die    -> four peaks at 90 deg, which
                                                   on a 4-way compass ALIAS TO FLAT  (m1, m2 -> 0)

The C4 aliasing is a real limit of our four-way heading and we state it rather than bury it: on four
sampled headings, tetradirectional tuning and no tuning are the same object. We reproduce Zhang's
twofold result exactly; for the fourfold arena we can only show that every RESOLVABLE directional
harmonic is annihilated -- consistent with tetradirectionality, unable to display four peaks.

--------------------------------------------------------------------------------------------------
THEOREM 2 (the local tuning curve ROTATES WITH THE COMPARTMENT). This is the sharper one, because it
predicts a cell type Zhang measured and named, and it is not a restatement of Theorem 1.

Do not marginalise over the whole arena. Take the tuning curve WITHIN one fundamental domain D_j
(Zhang's `subcompartment'). Write the arena's generator as g, and note that g maps domain D_j onto
domain D_{j-1}. Then invariance gives, with no further assumption,

    T_{j-1}(g.h) = T_j(h),

i.e. THE TUNING CURVE OF ONE COMPARTMENT IS THE TUNING CURVE OF ITS NEIGHBOUR, ROTATED BY THE GROUP
ELEMENT THAT RELATES THEM. A cell obeying this is unidirectional inside any single compartment and
its preferred direction steps round by 360/|G| degrees from compartment to compartment; summed over
compartments its GLOBAL curve is multidirectional. That is exactly Zhang's between-compartment (BC)
cell -- 9/67 of his tetradirectional and 9/48 of his bidirectional cells -- and he measures precisely
our equation: "the global preferred firing direction in each subcompartment rotated by 90 degrees in
successive subcompartments; cross-correlations of tuning curves from adjacent subcompartment pairs
peaked at ~90 deg (circular V-test against 90 deg, V = 7.98, p < 0.001)".

WHAT WE DO *NOT* PREDICT, AND SAY SO. Zhang's other class -- the within-compartment (WC) cells, which
are multidirectional even inside a single compartment (29/67 TD, 22/48 BD) -- does NOT follow from
invariance. At a fixed position, invariance relates u(x, g.h) to u(g^-1.x, h), a DIFFERENT PLACE, so
it puts no constraint on the tuning curve at that position. Zhang reaches the same conclusion from the
data and attributes WC cells to a learned re-expression of the global symmetry locally. If our
networks produce BC-type units and not WC-type units, that is a clean, falsifiable statement of what
the quotient law does and does not explain, and it is worth more than a claim to explain everything.

--------------------------------------------------------------------------------------------------
THE PREDICTION, stated before the run. G is the subgroup of the ARENA's group that the ENCODING
cannot break, so the fold -- and hence the annihilation -- is a property of the PAIR, not of either
alone. m1 is the unidirectional modulation of the GLOBAL curve:

    encoding   subgroup it cannot break     m1 in s1   m1 in s2   m1 in s4
    full       trivial                      high       high       high     <- the reachable null
    parity     trivial ({E,S} vs {W,N})     high       high       high     <- the MATCHED null
    axis       C2      ({E,W} vs {N,S})     high       LOW        LOW
    const      C4      (no compass at all)  high       LOW        LOW

and m2 (bidirectional modulation) separates the last two rows in the fourfold arena:

    axis  in s4:   folds by C2 only  ->  m2 SURVIVES  (bidirectional)
    const in s4:   folds by C4       ->  m2 DIES too  (flat = aliased tetradirectional)

`axis` and `parity` carry EXACTLY ONE BIT EACH. If multidirectional tuning were driven by how much
the compass knows, they would behave identically. The law says the axis network's units go
bidirectional and the parity network's stay unidirectional. That is Zhang's phenomenon produced by
symmetry and not by information, and no experiment on an animal can run it.

For Theorem 2 the null is built the same way as everywhere else in this paper: the SAME pretend group
is applied in EVERY arena. In s1 it is a symmetry of nothing, so the local tuning must NOT rotate with
the pretend compartment, and `rot_gain` must sit at zero. And `best_shift` is scanned over all four
rotations rather than tested only at the predicted one, so the measurement can return any answer.

--------------------------------------------------------------------------------------------------
WHAT ACTUALLY HAPPENED (n = 10/10/8 networks; recorded after the run, prediction left standing above).

THE PREDICTION WAS HALF WRONG, AND THE HALF THAT WAS WRONG IS THE INTERESTING ONE.

The harmonic ladder came out exactly as predicted -- but it is set by the ENCODING'S invariance group
and is very nearly INDIFFERENT TO THE ARENA:

    m1 (unidirectional)     s1      s2      s4          m2 (bidirectional)    s1      s2      s4
    full                  0.338   0.308   0.300         full                0.226   0.215   0.220
    parity                0.274   0.237   0.224         parity              0.028   0.030   0.017
    axis                  0.047   0.008   0.009         axis                0.405   0.419   0.367
    const                 0.052   0.011   0.011         const               0.021   0.025   0.009

`axis` (C2-invariant) annihilates the odd harmonic and keeps the even one: BIDIRECTIONAL tuning.
`const` (C4-invariant) annihilates BOTH resolvable harmonics: FLAT, the aliased tetradirectional
curve. `parity` -- the same one bit as `axis` -- keeps m1 and kills m2 instead. Theorem 1 holds.

BUT THE REACHABLE NULL FIRED. In s1, an arena with NO SYMMETRY AT ALL, `axis` already gives m1 = 0.047
and bi_frac = 0.88, and its local tuning already rotates through 180 degrees with a FICTITIOUS
half-arena (rot_gain = +0.62). The arena adds only a sharpening on top (m1 0.047 -> 0.008, bi_frac
0.88 -> 0.99, rot_gain +0.62 -> +0.79). The ENCODING effect dwarfs the FOLD effect, exactly as the
sRSA confound did before it.

SO THE GLOBAL TUNING CURVE IS NOT A MEASURE OF FOLDING, AND WE DO NOT USE IT AS ONE. A compass that
cannot tell East from West, combined with ordinary egocentric-boundary tuning -- a unit that fires
when facing a nearby wall peaks North in the North half and South in the South half -- produces
bidirectional tuning in ANY arena whatever. That is precisely the deflationary account Alexander et
al. (2020) give of Jacob's bidirectional retrosplenial cells, and our model reproduces their
mechanism in an arena we KNOW has no symmetry.

THE RESULT IS THEREFORE A DISSOCIATION, AND IT IS WORTH MORE THAN THE CONFIRMATION WOULD HAVE BEEN.
Set the tuning beside the map (phase decoding, `phase_full_n10.csv`, chance = 0.5):

                              directional tuning        the MAP
    s1 / axis                 bidirectional  0.88       0.971   NOT FOLDED
    s2 / axis                 bidirectional  0.99       0.552   FOLDED
    s2 / full                 unidirectional 0.36       0.978   not folded
    s2 / parity  (same 1 bit) unidirectional 0.12       0.955   not folded

SAME TUNING, OPPOSITE MAPS. Multidirectional tuning is NECESSARY BUT NOT SUFFICIENT for the fold: the
compass must be G-invariant AND the arena must be G-symmetric. It follows that no amount of
multidirectional tuning in cortex licenses an inference about the hippocampal map -- which dissolves
the apparent paradox in LaChance & Hasselmo (2024), where the postrhinal compass folds hard
(p = 2e-10) while entorhinal grid maps do not move at all. Two compasses, two maps, one law.

A SIGN ERROR, RECORDED. `pred_shift` for C4 was originally computed as +1 heading step per quarter
turn. It is -1: `rot90` is (x, y) -> (y, N+1-x), which sends East to North. The data said so before
we did -- `best_shift` came back as 3, deterministically, in 28 of 28 `const` networks -- which is
what scanning all four shifts instead of testing only the predicted one is FOR. `heading_shift()` now
reads the answer off the rotation function rather than trusting the algebra twice.

    PYTHONPATH=. python3 analysis/compass_symmetry.py \
        --ckpt-root <dir> --data-root <dir> --out Report/data/compass_symmetry.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_phase_decoding import (  # noqa: E402
    orbit_and_phase, rot90, rot180, ARENA)
from project5_symmetry.analysis.cell_types import collect_hd  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

N_HD = 4
ORDER = {'c2': 2, 'c4': 4}

# 0=E, 1=S, 2=W, 3=N, in (x, y) with y increasing downward (MiniGrid's convention).
_DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])


def heading_shift(group):
    """How many heading steps the group's GENERATOR turns a direction through.

    DERIVED FROM THE ROTATION FUNCTION ITSELF, not by hand. A rotation of the arena turns positions
    AND headings, and the two must be turned by the same group element or Theorem 2 is being tested
    against the wrong prediction. The handedness is easy to get backwards -- `rot90` here is
    (x, y) -> (y, N+1-x), whose linear part sends East to North, i.e. it steps the heading index
    DOWN by one, not up. Rather than re-derive that algebra and risk the same slip twice, we read it
    straight off the function: displace a point by each unit heading, rotate both, and see which
    heading the displacement became.

    Returns 2 for C2 (a half-turn is its own inverse, so the sign cannot bite) and 3 for C4.
    """
    f = rot180 if group == 'c2' else rot90
    p = np.array([[ARENA // 2, ARENA // 2]])
    shifts = set()
    for h, d in enumerate(_DIRS):
        turned = f(p + d, ARENA)[0] - f(p, ARENA)[0]
        h2 = int(np.flatnonzero((_DIRS == turned).all(1))[0])
        shifts.add((h2 - h) % N_HD)
    assert len(shifts) == 1, f'{group}: rotation is not a rigid heading shift ({shifts})'
    return shifts.pop()


def _tuning(H, cx, hd, keep=None):
    """Occupancy-balanced direction tuning, T[U, 4], over the position bins in `keep`.

    THE OCCUPANCY CONFOUND IS THE ONE THAT WOULD FAKE THIS RESULT, and in simulation we can simply
    delete it. Rats run along walls, so near a wall they mostly face along it: p(x | h) is strongly
    non-uniform, and a purely SPATIAL cell then shows apparent direction tuning whose symmetry is
    inherited from the arena's symmetry rather than from the code's. That is the first thing a
    referee will say, and it is why the real recordings are hard to read -- Zhang has to argue the
    point from the uniformity of preferred directions (Rayleigh z = 0.76, p = 0.47) rather than
    remove it.

    We remove it. Build the per-heading rate map, keep only position bins visited from ALL FOUR
    headings, and average over those bins with UNIFORM weight. The curve is then what the unit would
    do if the agent spent equal time in every place facing every way, so any residual symmetry in it
    belongs to the code and not to the behaviour.
    """
    U = H.shape[1]
    ncell = ARENA * ARENA
    sums = np.zeros((N_HD, ncell, U))
    counts = np.zeros((N_HD, ncell))
    np.add.at(sums, (hd, cx), H)
    np.add.at(counts, (hd, cx), 1)

    vis = (counts > 0).all(0)                        # visited from every heading
    if keep is not None:
        vis &= keep
    if vis.sum() < 2:
        return np.zeros((U, N_HD)), 0
    rate = sums[:, vis] / counts[:, vis, None]       # [4, nvis, U]
    return rate.mean(1).T, int(vis.sum())            # [U, 4], uniform over place


def raw_tuning(H, hd):
    """The NAIVE occupancy-weighted marginal -- what an experimentalist computes from a real animal.

    We compute it alongside the balanced curve so the size of the occupancy confound is a number in
    the CSV rather than an assertion in the text. In a thigmotactic animal the gap between the two is
    exactly \citet{alexander2020}'s mechanism; in our agent, which samples headings close to
    uniformly, it should be small -- and if it is, that is a fact about our locomotor policy that we
    are obliged to report, not to assume.
    """
    return np.stack([H[hd == d].mean(0) if (hd == d).any() else np.zeros(H.shape[1])
                     for d in range(N_HD)], 1)


def harmonics(T):
    """Modulation depth of each resolvable harmonic of a 4-point circular tuning curve.

    F0 is the mean rate, F1 the unidirectional (one-peak) component, F2 the bidirectional (two peaks
    180 deg apart) component. On four samples that is all there is: F3 is the conjugate of F1, and
    any four-peaked structure aliases onto F0.

    We report modulation depths m_k = |F_k| / F0, dimensionless and comparable across units and
    networks (a unit that doubles its gain does not move them), plus the share of directional power
    sitting in the bidirectional harmonic,

        bi_frac = |F2|^2 / (|F1|^2 + |F2|^2),

    which is the nearest thing a four-way compass has to an experimentalist's "is this cell
    bidirectional?" call: bi_frac -> 1 is a unit firing in two opposite directions that cannot tell
    them apart. It is undefined for a unit with no directional modulation at all, so it is averaged
    only over units clearing a modulation floor, and we report how many those are.
    """
    F0 = T.mean(1)
    ang = 2 * np.pi * np.arange(N_HD) / N_HD
    F1 = (T * np.exp(-1j * ang)).mean(1)
    F2 = (T * np.exp(-2j * ang)).mean(1)

    ok = F0 > 1e-9
    m1 = np.divide(np.abs(F1), F0, out=np.zeros_like(F0), where=ok)
    m2 = np.divide(np.abs(F2), F0, out=np.zeros_like(F0), where=ok)

    p1, p2 = np.abs(F1) ** 2, np.abs(F2) ** 2

    # THE FLOOR MATTERS, AND OUR FIRST ONE WAS FAR TOO LOW. bi_frac is a RATIO of two harmonics; on a
    # unit with no directional tuning at all it is a ratio of two noise terms and will happily return
    # any value in [0, 1]. At a floor of 0.02 the `const` networks -- whose curves are FLAT
    # (m1 = 0.011, m2 = 0.025) -- were passing, and reporting bi_frac = 0.77, which reads as "77%
    # bidirectional" for a cell that is not directional at all. That number is meaningless and it
    # nearly went into the paper.
    #
    # The floor is now 0.10, an order of magnitude above the noise level of a flat curve and well
    # below the tuned encodings (`full` m1 ~ 0.31, `axis` m2 ~ 0.42). `n_tuned` is returned so that
    # every reported bi_frac carries the count of units it was actually averaged over, and a
    # condition where that count collapses cannot be quoted as though it had a bidirectionality.
    tuned = (m1 + m2) > 0.10
    denom = p1 + p2
    bi = np.divide(p2, denom, out=np.full_like(p1, np.nan), where=(denom > 0) & tuned)
    return m1, m2, bi, int(tuned.sum())


def _corr(A, B):
    """Per-unit Pearson correlation between two [U, 4] tuning curves."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def rotation(H, cx, hd, phase, group):
    """THEOREM 2. Does the within-compartment tuning curve rotate with the compartment?

    The prediction is T_{j-1}(g.h) = T_j(h), i.e. T_{j-1} = roll(T_j, k) with k = 4 / |G| heading
    steps per compartment step (k = 1 for C4, a 90 deg rotation; k = 2 for C2, a 180 deg rotation).

    We do NOT test only at the predicted shift, because a test that can only confirm is not a test.
    We scan all four rotations, report which one maximises the mean cross-correlation over adjacent
    compartment pairs (`best_shift` -- it can come back as anything), and report

        rot_gain = corr at the PREDICTED shift  -  corr at NO shift,

    which is positive only if the tuning genuinely turns with the compartment rather than staying put.
    In the s1 arena the group is a symmetry of nothing, the compartments are a fiction, and rot_gain
    must sit at zero. That is the reachable null.

    `local_m1` is the modulation of the WITHIN-compartment curve. Zhang's between-compartment cells
    are unidirectional inside a compartment (local_m1 high) while their global curve is not; that
    combination is the signature, and it is what we predict.
    """
    order = ORDER[group]
    k = heading_shift(group)

    T = []
    for j in range(order):
        keep = np.zeros(ARENA * ARENA, dtype=bool)
        keep[np.unique(cx[phase == j])] = True
        Tj, _ = _tuning(H, cx, hd, keep=keep)
        T.append(Tj)

    shifts = np.zeros(N_HD)
    for s in range(N_HD):
        c = [_corr(T[(j - 1) % order], np.roll(T[j], s, axis=1)) for j in range(order)]
        shifts[s] = float(np.mean(c))

    local_m1 = float(np.mean([harmonics(Tj)[0].mean() for Tj in T]))
    return {
        'best_shift': int(np.argmax(shifts)),
        'pred_shift': k,
        'rot_gain': round(float(shifts[k] - shifts[0]), 5),
        'corr_pred': round(float(shifts[k]), 5),
        'corr_none': round(float(shifts[0]), 5),
        'local_m1': round(local_m1, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--encodings', nargs='+', default=['full', 'axis', 'parity', 'const'])
    ap.add_argument('--groups', nargs='+', default=['c2', 'c4'], choices=['c2', 'c4'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
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
        for enc in a.encodings:
            for s in a.seeds:
                p = ck / 'hd_invariance' / cond / enc / f'seed_{s:02d}' / 'ckpt_final.pt'
                if not p.exists():
                    continue
                model = model_from_checkpoint(
                    torch.load(p, map_location='cpu', weights_only=False), dev)
                with torch.no_grad():
                    H, pos, hd = collect_hd(model, ds[cond], enc, a.n_states, dev, seed=s)
                cx = (pos[:, 0] - 1) * ARENA + (pos[:, 1] - 1)

                # Theorem 1: the GLOBAL curve. It does not depend on a group, so compute it once.
                T, nvis = _tuning(H, cx, hd)
                m1, m2, bi, ntuned = harmonics(T)
                # ... and the uncorrected version, so the occupancy confound is measured, not asserted.
                rm1, rm2, rbi, _ = harmonics(raw_tuning(H, hd))

                for group in a.groups:
                    # Theorem 2: the LOCAL curves. The SAME pretend group is applied in every arena,
                    # so the compartment count is identical across s1/s2/s4 and the numbers are
                    # comparable -- and in s1 the group is a symmetry of nothing, which is exactly
                    # what makes s1 the reachable null: the tuning there must NOT rotate.
                    _, phase = orbit_and_phase(pos, group, ARENA)
                    r = {'condition': cond, 'encoding': enc, 'seed': s, 'group': group,
                         'm1': round(float(np.mean(m1)), 5),
                         'm2': round(float(np.mean(m2)), 5),
                         'bi_frac': round(float(np.nanmean(bi)), 5),
                         'raw_m1': round(float(np.mean(rm1)), 5),
                         'raw_m2': round(float(np.mean(rm2)), 5),
                         'raw_bi_frac': round(float(np.nanmean(rbi)), 5),
                         'n_tuned': ntuned, 'n_units': int(H.shape[1]), 'n_cells_vis': nvis}
                    r.update(rotation(H, cx, hd, phase, group))
                    rows.append(r)
                    print(f"  {cond}/{enc:<6s}/s{s:02d} {group}  m1={r['m1']:.4f} m2={r['m2']:.4f} "
                          f"bi={r['bi_frac']:.3f} | rot_gain={r['rot_gain']:+.4f} "
                          f"best_shift={r['best_shift']} (pred {r['pred_shift']}) "
                          f"local_m1={r['local_m1']:.4f}", flush=True)

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
