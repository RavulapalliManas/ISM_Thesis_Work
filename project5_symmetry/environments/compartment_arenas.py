"""Two identical compartments, related by translation or by rotation.

The experiment
--------------
Two papers, one dissociation. Spiers, Hayman, Jovalekic, Marozzi & Jeffery (Cereb. Cortex
2015) recorded CA1 place cells in four visually identical compartments arranged IN A ROW off
a corridor, and found "a high degree of spatial repetition with a slight degree of rate-based
discrimination"; they conclude that "path integration is not used, or at least not
spontaneously" to separate them.

Grieves, Jenkins, Harland, Wood & Dudchenko (Hippocampus 2016, 26(1):118-134) then ran the
SAME four compartments in two arrangements. Parallel (all facing the same way): "place cells
often exhibited repeated fields". Radial (60 degrees apart): "significantly less place field
repetition was apparent" (all p < 0.0001). It survived removing the orientation landmark, so
it is not extra-maze vision, and they attribute it to "directional information derived from
the animal's self-motion", hypothesising the head-direction system.

(The parallel-versus-radial comparison is GRIEVES, not Spiers. Spiers studied only the
in-a-row arrangement. This module previously mis-attributed it.)

Our account supplies the reason:

    a self-motion signal resolves only those symmetries under which it is not
    itself invariant.

    translation:  g . (pos, h) = (pos + delta, h)     the compass reads the same
                  -> HD cannot disambiguate -> the code folds -> FIELD REPETITION
    rotation:     g . (pos, h) = (R180 pos, h + 2)    the compass turns with the room
                  -> HD disambiguates      -> the code lifts -> REMAPPING

Why this works here and the old `two_room.py` did not
------------------------------------------------------
MiniGrid sets `see_through_walls = False`, so an agent inside a CLOSED room cannot see out.
The previous design used rooms that spanned the arena width, so the agent saw the outer
boundary at different distances in each room and translation was never a symmetry
(measured then: max|delta| = 0.6471).

Here each compartment is a closed 6x6 box whose walls AND door are exact group images of
the other's, joined by a corridor that lives entirely outside both wall rings and presents
an identical one-cell stub to each door. Measured (see test_compartment_arenas.py):

    rotation     144 / 144 views identical
    translation  144 / 144 views identical

i.e. obs(x, h) == obs(g.x, g.h) for every cell of the room and every heading. The ONLY
thing that differs between the two conditions is how the group acts on heading. That is the
entire manipulation.
"""
from __future__ import annotations

import numpy as np

from project5_symmetry.environments.arena import (BLUE, PixelObsWrapper, RED,
                                                  SymmetryArena, YELLOW)

# The arena is 24, not 18. In an 18x18 box the corridor cannot be locally identical at both
# doors -- B's door ends up against the outer boundary, so the cell diagonally past it is a
# wall where A's is corridor, and the compartments stop being observationally identical
# (measured: 112/144 views matched). At 24 each door opens into its own straight 7-cell
# corridor and the junction sits beyond the 6-cell view depth. The model is unaffected: the
# observation is a 7x7 egocentric view whatever the arena size.
ARENA = 24
SIZE = 6                       # compartment interior is SIZE x SIZE
A0, B0 = (2, 9), (15, 9)       # top-left interior cell; B is A shifted 13 rows down
DOOR_LOCAL = (2, SIZE)         # on the room's right wall, local row 2

# Deliberately asymmetric inside the room, so position WITHIN a compartment is well
# identified and the only ambiguity is which compartment you are in.
PATTERN = {(0, 0): BLUE, (0, 1): BLUE, (1, 0): BLUE,
           (3, 4): RED, (4, 4): RED, (4, 3): RED,
           (2, 2): YELLOW}

MODES = ('translation', 'rotation')


def group(mode: str):
    """The group element acting on room-local coordinates, and on heading."""
    if mode == 'translation':
        return (lambda i, j: (i, j)), (lambda h: h)
    if mode == 'rotation':
        return (lambda i, j: (SIZE - 1 - i, SIZE - 1 - j)), (lambda h: (h + 2) % 4)
    raise ValueError(f'mode must be one of {MODES}, got {mode!r}')


CORRIDOR = 7          # straight run past each door: longer than the 6-cell view depth
JUNCTION = 22         # the column (translation) / columns (rotation) where the paths join


def _cells(mode: str):
    g, _ = group(mode)
    A = {(A0[0] + i, A0[1] + j) for i in range(SIZE) for j in range(SIZE)}
    B = {(B0[0] + g(i, j)[0], B0[1] + g(i, j)[1]) for i in range(SIZE) for j in range(SIZE)}
    dA = (A0[0] + DOOR_LOCAL[0], A0[1] + DOOR_LOCAL[1])            # (4, 15), A's right wall
    rA = dA[0]

    if mode == 'translation':
        dB = (B0[0] + DOOR_LOCAL[0], B0[1] + DOOR_LOCAL[1])        # (17, 15), B's right wall
        rB = dB[0]
        # each door opens east into an identical straight corridor, then they meet far away
        runA = {(rA, c) for c in range(dA[1] + 1, JUNCTION + 1)}
        runB = {(rB, c) for c in range(dB[1] + 1, JUNCTION + 1)}
        link = {(r, JUNCTION) for r in range(rA, rB + 1)}
    else:
        dB = (B0[0] + (SIZE - 1 - DOOR_LOCAL[0]), B0[1] - 1)       # (18, 8), B's LEFT wall
        rB = dB[0]
        # A's door faces east, B's (its 180-degree image) faces west: mirror the corridors.
        # The link must go AROUND room B, never through its wall ring.
        runA = {(rA, c) for c in range(dA[1] + 1, JUNCTION + 1)}    # row 4, cols 16..22
        runB = {(rB, c) for c in range(1, dB[1])}                   # row 18, cols 1..7
        link = ({(r, JUNCTION) for r in range(rA, ARENA - 1)}       # down the right edge
                | {(ARENA - 2, c) for c in range(1, JUNCTION + 1)}  # along the bottom
                | {(r, 1) for r in range(rB, ARENA - 1)})           # up the left edge

    return A | B | {dA, dB} | runA | runB | link, A, B, dA, dB


class CompartmentArena(SymmetryArena):
    """Two closed compartments joined by a corridor; B is the group image of A."""

    def __init__(self, mode: str, F: int = 7, seed: int = 0, tint: float = 0.0, **kw):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
        self.mode = mode
        self.tint_b = float(tint)          # faint symmetry-breaking cue on room B (0 = none)
        self.passable, self.room_a, self.room_b, self.door_a, self.door_b = _cells(mode)
        super().__init__(shape='square', size=ARENA, U=0, F=F, seed=seed,
                         use_landmarks=True, symmetry_condition=None, **kw)

    def _is_passable(self, row: int, col: int) -> bool:
        return (row, col) in self.passable

    def _get_landmark_tiles(self) -> dict:
        g, _ = group(self.mode)
        tiles = {(A0[0] + i, A0[1] + j): v for (i, j), v in PATTERN.items()}
        tiles.update({(B0[0] + g(i, j)[0], B0[1] + g(i, j)[1]): v
                      for (i, j), v in PATTERN.items()})
        return {rc: v for rc, v in tiles.items() if rc in self.passable}

    def group_action(self, pos_rc: tuple[int, int], heading: int):
        """Map (position, heading) by the non-identity element. Heading is MiniGrid's
        0=E 1=S 2=W 3=N."""
        g, gh = group(self.mode)
        r, c = pos_rc
        if (r, c) in self.room_a:
            i, j = r - A0[0], c - A0[1]
            gi, gj = g(i, j)
            return (B0[0] + gi, B0[1] + gj), gh(heading)
        if (r, c) in self.room_b:
            # g is an involution on local coords, so the inverse is g itself
            i, j = r - B0[0], c - B0[1]
            gi, gj = g(i, j)
            return (A0[0] + gi, A0[1] + gj), gh(heading)
        raise ValueError(f'{pos_rc} is not inside a compartment')


def make_compartment_env(mode: str, F: int = 7, seed: int = 0, tint: float = 0.0):
    """Top-level factory: picklable, so `generate_dataset` can rebuild it in each worker."""
    return PixelObsWrapper(CompartmentArena(mode, F=F, seed=seed, tint=tint), tile_size=1)


def observation_mismatch(mode: str) -> np.ndarray:
    """max |obs(x,h) - obs(g.x, g.h)| over every cell of compartment A and every heading.

    Zero everywhere means the two compartments are exactly indistinguishable from vision,
    and the conditions differ only in how the group acts on heading.
    """
    env = make_compartment_env(mode)
    env.reset(seed=0)
    e = env.unwrapped
    g, gh = group(mode)
    out = []
    for i in range(SIZE):
        for j in range(SIZE):
            ra, ca = A0[0] + i, A0[1] + j
            gi, gj = g(i, j)
            rb, cb = B0[0] + gi, B0[1] + gj
            for h in range(4):
                e.agent_pos = np.array([ca, ra]); e.agent_dir = h
                oa = np.asarray(e.gen_obs()['image'], dtype=np.int16)
                e.agent_pos = np.array([cb, rb]); e.agent_dir = gh(h)
                ob = np.asarray(e.gen_obs()['image'], dtype=np.int16)
                out.append(np.abs(oa - ob).max())
    return np.array(out)
