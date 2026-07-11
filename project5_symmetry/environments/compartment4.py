r"""Four identical compartments in a row (Spiers 2015), the full multi-compartment paradigm.

The 2-room module (`compartment_arenas.py`) reduces the rodent experiment to a single pair.
Here we build the actual four-room arrangement \citet{spiers2015} recorded: four visually
identical closed rooms off a common corridor, each the translate of the last. Head direction is
invariant under translation, so a predictive code cannot tell the rooms apart and must fold all
four onto one map -- the "high degree of spatial repetition" Spiers reported, now at the true
room count. Chance room-decoding is therefore 1/4, and a place unit that folds carries up to
four fields.

Geometry follows the 2-room design: each room is a closed SIZE x SIZE box whose door opens east
into an identical straight corridor longer than the 6-cell view depth, and the rooms are spaced
so no room lies within another's egocentric view. Observational identity is asserted by
`observation_mismatch` (0 across all rooms, headings) before the arena is used.
"""
from __future__ import annotations

import numpy as np

from project5_symmetry.environments.arena import (BLUE, PixelObsWrapper, RED,
                                                  SymmetryArena, YELLOW)

ARENA = 40
SIZE = 6                      # room interior
N_ROOMS = 4
STEP = 8                      # vertical spacing of room top-left corners (>= 7: no cross-view)
COL0 = 2                      # rooms share the same left column
ROW0 = 2                      # first room's top interior row
DOOR_LOCAL = (2, SIZE)        # local (row, col) of the door on the right wall
CORRIDOR = 8                  # straight run east of each door (> 6-cell view depth)
JUNCTION = COL0 + SIZE + 1 + CORRIDOR   # column of the shared vertical junction corridor

# asymmetric interior pattern, so within-room position is identified; only the room is ambiguous
PATTERN = {(0, 0): BLUE, (0, 1): BLUE, (1, 0): BLUE,
           (3, 4): RED, (4, 4): RED, (4, 3): RED,
           (2, 2): YELLOW}


def _room_origin(i):
    return (ROW0 + i * STEP, COL0)


def _cells():
    """(all passable cells, list of per-room cell sets, list of door cells)."""
    rooms, doors, runs = [], [], []
    for i in range(N_ROOMS):
        r0, c0 = _room_origin(i)
        rooms.append({(r0 + a, c0 + b) for a in range(SIZE) for b in range(SIZE)})
        dr, dc = r0 + DOOR_LOCAL[0], c0 + DOOR_LOCAL[1]        # door cell (in the wall opening)
        doors.append((dr, dc))
        runs.append({(dr, c) for c in range(dc + 1, JUNCTION + 1)})   # straight corridor east
    door_rows = [d[0] for d in doors]
    junction = {(r, JUNCTION) for r in range(min(door_rows), max(door_rows) + 1)}
    passable = set().union(*rooms) | set(doors) | set().union(*runs) | junction
    return passable, rooms, doors


class Compartment4Arena(SymmetryArena):
    """Four closed compartments in a row, related by translation, joined by one corridor."""

    def __init__(self, F: int = 7, seed: int = 0, **kw):
        self.passable, self.rooms, self.doors = _cells()
        super().__init__(shape='square', size=ARENA, U=0, F=F, seed=seed,
                         use_landmarks=True, symmetry_condition=None, **kw)

    def _is_passable(self, row: int, col: int) -> bool:
        return (row, col) in self.passable

    def _get_landmark_tiles(self) -> dict:
        tiles = {}
        for i in range(N_ROOMS):
            r0, c0 = _room_origin(i)
            tiles.update({(r0 + a, c0 + b): v for (a, b), v in PATTERN.items()})
        return {rc: v for rc, v in tiles.items() if rc in self.passable}

    def room_of(self, pos_rc):
        for i, room in enumerate(self.rooms):
            if tuple(pos_rc) in room:
                return i
        return -1


def make_compartment4_env(F: int = 7, seed: int = 0):
    return PixelObsWrapper(Compartment4Arena(F=F, seed=seed), tile_size=1)


def observation_mismatch() -> np.ndarray:
    """max |obs(room0 cell, h) - obs(room i cell, h)| over rooms i>0, cells, headings.

    Zero everywhere means all four rooms are exactly indistinguishable from vision at matched
    local position and heading, so the only thing that could separate them is self-motion.
    """
    env = make_compartment4_env()
    env.reset(seed=0)
    e = env.unwrapped
    out = []
    for a in range(SIZE):
        for b in range(SIZE):
            r0, c0 = _room_origin(0)
            for h in range(4):
                e.agent_pos = np.array([c0 + b, r0 + a]); e.agent_dir = h
                o0 = np.asarray(e.gen_obs()['image'], dtype=np.int16)
                for i in range(1, N_ROOMS):
                    ri, ci = _room_origin(i)
                    e.agent_pos = np.array([ci + b, ri + a]); e.agent_dir = h
                    oi = np.asarray(e.gen_obs()['image'], dtype=np.int16)
                    out.append(np.abs(o0 - oi).max())
    return np.array(out)
