"""Ground-truth validation for compass_symmetry.py.

The metrics in compass_symmetry.py became load-bearing for a claim in the paper before anyone had
checked that they return the right answer on data whose answer is known. That is the wrong order, and
this file is the correction. Every test below builds a synthetic population whose symmetry we chose,
and asserts that the metric recovers it.

    PYTHONPATH=. python3 analysis/test_compass_symmetry.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.compass_symmetry import (  # noqa: E402
    _tuning, harmonics, rotation, heading_shift, N_HD)
from project5_symmetry.analysis.run_phase_decoding import orbit_and_phase, rot90, rot180, ARENA

rng = np.random.default_rng(0)
OK = []


def check(name, cond, detail=''):
    OK.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def synth(fn, n=60_000):
    """Sample (pos, heading) uniformly and build H[i, u] = fn(pos_i, h_i) for a few units."""
    xy = rng.integers(1, ARENA + 1, size=(n, 2))
    hd = rng.integers(0, N_HD, size=n)
    H = np.stack([fn(xy, hd)], 1).astype(float)
    cx = (xy[:, 0] - 1) * ARENA + (xy[:, 1] - 1)
    return H, xy, hd, cx


# ---------------------------------------------------------------- heading_shift
# The group element must turn positions and headings by the SAME rotation. If this is wrong,
# Theorem 2 is tested against the wrong prediction -- which is exactly the bug we shipped once.
print('\nheading_shift: does the derived turn match the rotation applied to positions?')
for group, f in (('c2', rot180), ('c4', rot90)):
    k = heading_shift(group)
    # An egocentric "facing the +x wall" vector must map consistently.
    d = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    p = np.array([[ARENA // 2, ARENA // 2]])
    turned = f(p + d[0], ARENA)[0] - f(p, ARENA)[0]
    idx = int(np.flatnonzero((d == turned).all(1))[0])
    check(f'{group}: E({0}) -> index {idx}, shift {k}', (0 + k) % 4 == idx)
check('c2 shift is 2', heading_shift('c2') == 2)
check('c4 shift is 3 (rot90 sends E to N, stepping the index DOWN)', heading_shift('c4') == 3)

# ---------------------------------------------------------------- harmonics
print('\nharmonics: does it recover a tuning curve whose symmetry we chose?')
# A unidirectional unit: cosine-tuned about East. (NOT a delta -- see the limitation below.)
H, xy, hd, cx = synth(lambda p, h: 1.0 + np.cos(2 * np.pi * h / N_HD))
T, _ = _tuning(H, cx, hd)
m1, m2, bi, _ = harmonics(T)
check('unidirectional (cosine) -> m1 >> m2', m1[0] > 0.4 and m2[0] < 1e-9,
      f'm1={m1[0]:.3f} m2={m2[0]:.2e}')
check('unidirectional (cosine) -> bi_frac ~ 0', bi[0] < 1e-9, f'bi={bi[0]:.2e}')

# LIMITATION, recorded rather than hidden. A unit tuned to exactly ONE of four headings is a delta,
# and a delta has a FLAT spectrum: |F1| == |F2| exactly, so bi_frac = 0.5 and the measure cannot say
# whether it is uni- or bidirectional. This is a property of a 4-way compass, not a bug. It matters
# only for cells sharper than our networks actually are: our `full` units read bi_frac = 0.33-0.37,
# i.e. broader than a delta and safely unidirectional.
H, xy, hd, cx = synth(lambda p, h: (h == 0).astype(float) + 0.5)
T, _ = _tuning(H, cx, hd)
m1, m2, bi, _ = harmonics(T)
check('LIMIT: a delta-tuned unit is ambiguous (bi_frac == 0.5), as it must be',
      abs(bi[0] - 0.5) < 1e-6, f'bi={bi[0]:.4f}')

# A pure BIDIRECTIONAL unit: fires facing East OR West. Odd harmonic must be annihilated exactly.
H, xy, hd, cx = synth(lambda p, h: np.isin(h, [0, 2]).astype(float) + 0.5)
T, _ = _tuning(H, cx, hd)
m1, m2, bi, _ = harmonics(T)
check('bidirectional -> m1 ~ 0', m1[0] < 1e-9, f'm1={m1[0]:.2e}')
check('bidirectional -> m2 > 0', m2[0] > 0.1, f'm2={m2[0]:.3f}')
check('bidirectional -> bi_frac ~ 1', bi[0] > 0.999, f'bi={bi[0]:.4f}')

# A FLAT unit (the aliased C4 case on a 4-way compass): both harmonics must die.
H, xy, hd, cx = synth(lambda p, h: np.ones(len(h)))
T, _ = _tuning(H, cx, hd)
m1, m2, _, ntuned = harmonics(T)
check('flat -> m1 ~ 0 and m2 ~ 0', m1[0] < 1e-9 and m2[0] < 1e-9, f'm1={m1[0]:.2e} m2={m2[0]:.2e}')
check('flat -> excluded by the tuning floor', ntuned == 0)

# ---------------------------------------------------------------- Theorem 2
print('\nrotation: does it detect a code that IS G-invariant, and reject one that is not?')

# A genuinely C2-INVARIANT conjunctive unit: u(x, h) = u(rot180 x, h+2), by construction.
# Fires when it is in the northern half AND facing East; by invariance it must also fire in the
# southern half facing West. Its GLOBAL tuning is bidirectional; its LOCAL tuning rotates.
def c2_invariant(p, h):
    north = p[:, 0] <= ARENA // 2
    return ((north & (h == 0)) | (~north & (h == 2))).astype(float) + 0.5


H, xy, hd, cx = synth(c2_invariant)
_, phase = orbit_and_phase(xy, 'c2', ARENA)
r = rotation(H, cx, hd, phase, 'c2')
check('C2-invariant unit -> best_shift == 2', r['best_shift'] == 2, f"got {r['best_shift']}")
check('C2-invariant unit -> rot_gain > 0', r['rot_gain'] > 0.5, f"rot_gain={r['rot_gain']:+.3f}")

# A unit that BREAKS C2: fires facing East everywhere, regardless of half. Local tuning does NOT
# rotate -- both halves prefer East. This is the reachable null and it must fire.
H, xy, hd, cx = synth(lambda p, h: (h == 0).astype(float) + 0.5)
_, phase = orbit_and_phase(xy, 'c2', ARENA)
r = rotation(H, cx, hd, phase, 'c2')
check('C2-BREAKING unit -> best_shift == 0', r['best_shift'] == 0, f"got {r['best_shift']}")
check('C2-BREAKING unit -> rot_gain <= 0', r['rot_gain'] <= 0, f"rot_gain={r['rot_gain']:+.3f}")

# The same for C4, whose handedness is the thing we got wrong.
def c4_invariant(p, h):
    # u(x,h) = 1 iff (h + j*k) == 0 mod 4, where j is the domain of x. Check it IS invariant:
    # phase(R.x) = j-1 and g.h = h+k, so u(R.x, h+k) = 1 iff ((h+k) + (j-1)k) = (h + j*k) mod 4. QED.
    # Writing (h - j*k) instead is the same sign slip the analysis itself once shipped -- and note it
    # would pass silently under C2, where k=2 is self-inverse. That is why C4 is the real test.
    _, ph = orbit_and_phase(p, 'c4', ARENA)
    k = heading_shift('c4')
    return ((h + ph * k) % 4 == 0).astype(float) + 0.5


H, xy, hd, cx = synth(c4_invariant)
_, phase = orbit_and_phase(xy, 'c4', ARENA)
r = rotation(H, cx, hd, phase, 'c4')
check('C4-invariant unit -> best_shift == 3', r['best_shift'] == 3, f"got {r['best_shift']}")
check('C4-invariant unit -> rot_gain > 0', r['rot_gain'] > 0.3, f"rot_gain={r['rot_gain']:+.3f}")

# ---------------------------------------------------------------- occupancy
print('\n_tuning: does the occupancy balancing actually remove an occupancy artefact?')
# A PURELY SPATIAL unit (no directional tuning at all), sampled with a heading-biased occupancy that
# depends on position -- the thigmotaxis confound. The balanced curve must stay flat; the naive
# occupancy-weighted marginal is allowed to be fooled.
n = 120_000
xy = rng.integers(1, ARENA + 1, size=(n, 2))
north = xy[:, 0] <= ARENA // 2
hd = np.where(rng.random(n) < 0.9, np.where(north, 0, 2), rng.integers(0, 4, n))  # faces E in N half
H = (north.astype(float) + 0.5)[:, None]                       # depends on POSITION only
cx = (xy[:, 0] - 1) * ARENA + (xy[:, 1] - 1)
T, _ = _tuning(H, cx, hd)
m1b, m2b, _, _ = harmonics(T)
raw = np.stack([H[hd == d].mean(0) for d in range(N_HD)], 1)
m1r, m2r, _, _ = harmonics(raw)
check('spatial-only unit: BALANCED curve is flat', m1b[0] < 1e-9 and m2b[0] < 1e-9,
      f'bal m1={m1b[0]:.2e} m2={m2b[0]:.2e}')
check('spatial-only unit: NAIVE curve is fooled by occupancy', m1r[0] > 0.05,
      f'raw m1={m1r[0]:.3f} (this is the artefact the balancing removes)')

print(f"\n{sum(OK)}/{len(OK)} passed")
sys.exit(0 if all(OK) else 1)
