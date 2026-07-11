"""Two-compartment arenas that isolate *which symmetry a compass can resolve*.

The claim
---------
A self-motion signal can only resolve the symmetries under which it is not
itself invariant.

  * Translating the world does not change your compass reading. Head direction is
    blind to translation symmetry, so the (observation, heading) pair stays
    ambiguous and the code must collapse onto the quotient  ->  field repetition.
  * Rotating the world does change the compass relative to the landmarks, so
    (observation, heading) is unique and the quotient is lifted  ->  remapping.

This is exactly the geometry dependence Spiers et al. (Cereb. Cortex 2015) report
in multicompartment environments -- fields repeat across *parallel* identical
rooms, but remap across *radial* ones -- and which they note is not explained by
self-motion cues.

The design
----------
Two 8x18 rooms (rows 1-8 and 11-18) separated by a two-tile divider (rows 9-10)
with a central doorway (cols 9-10). Geometry is IDENTICAL between the two
conditions; only the landmark layout of room B differs:

    translation:  room B tiles = room A tiles shifted by +10 rows
    rotation:     room B tiles = room A tiles mapped by (r,c) -> (19-r, 19-c)

Both maps send room A's landmark tiles onto room B's, so the two arenas have the same
walls, the same doorway, and the same multiset of landmark tiles. What differs is the
group action on *heading*:

    translation:  g . (pos, h) = (pos + delta, h)          h unchanged
    rotation:     g . (pos, h) = (R180 pos,   h + 2)       h rotated

STATUS: ROTATION WORKS, TRANSLATION DOES NOT. Measured against the observation oracle
(2026-07-10), comparing obs(x, h) with obs(g.x, g.h) away from the doorway:

    rotation     max |delta| = 0.0000   -- an exact symmetry of the arena
    translation  max |delta| = 0.6471   -- NOT a symmetry

This is not a bug to be fixed by tuning the layout. A 180-degree rotation maps the whole
bounded grid onto itself; a translation cannot -- it moves the outer walls, and the 7x7
egocentric view sees them. Any bounded, connected arena has this problem. Realising a
translation symmetry needs the Spiers et al. (2015) side-corridor geometry, where the two
compartments are locally identical because the agent never sees the outer boundary from
inside one.

DO NOT USE `room_symmetry='translation'` for the invariance claim until that is rebuilt.
The claim it was meant to support -- that a compass resolves only the symmetries under
which it is not itself invariant -- is instead established in the same arena by the HD
encoding manipulation (`environments/hd_encodings.py`), which varies the group action on
heading while holding the arena, the data and the architecture fixed.
"""
from __future__ import annotations

import numpy as np

from project5_symmetry.environments.arena import (BLUE, GREEN, PixelObsWrapper,
                                                  RED, SymmetryArena, YELLOW)

ARENA = 18
ROOM_A_ROWS = range(1, 9)        # 1..8
ROOM_B_ROWS = range(11, 19)      # 11..18
DIVIDER_ROWS = (9, 10)
DOOR_COLS = (9, 10)
ROW_SHIFT = 10                   # room A row r  ->  room B row r + 10


def _room_a_pattern() -> dict[tuple[int, int], list]:
    """Landmarks inside room A (rows 1..8). Deliberately asymmetric within the
    room, so position *within* a room is well identified and the only ambiguity
    is which room you are in."""
    tiles: dict[tuple[int, int], list] = {}
    for r in range(2, 5):
        for c in range(2, 5):
            tiles[(r, c)] = BLUE
    for r in range(2, 5):
        for c in range(15, 18):
            tiles[(r, c)] = RED
    for r in range(6, 9):
        for c in range(8, 11):
            tiles[(r, c)] = YELLOW
    for r in range(6, 8):
        for c in range(2, 4):
            tiles[(r, c)] = GREEN
    return tiles


def translate(rc: tuple[int, int]) -> tuple[int, int]:
    r, c = rc
    return (r + ROW_SHIFT, c)


def rotate180(rc: tuple[int, int]) -> tuple[int, int]:
    r, c = rc
    return (ARENA + 1 - r, ARENA + 1 - c)


class TwoRoomArena(SymmetryArena):
    """Two identical rooms; room B is a translated or rotated copy of room A."""

    def __init__(self, room_symmetry: str, F: int = 7, seed: int = 0, **kw):
        if room_symmetry not in ('translation', 'rotation'):
            raise ValueError(f'room_symmetry must be translation|rotation, got {room_symmetry!r}')
        self.room_symmetry = room_symmetry
        super().__init__(shape='square', size=ARENA, U=0, F=F, seed=seed,
                         use_landmarks=True, symmetry_condition=None, **kw)

    # ── geometry: two rooms + a central doorway ──────────────────────────────
    def _is_passable(self, row: int, col: int) -> bool:
        if not (1 <= row <= ARENA and 1 <= col <= ARENA):
            return False
        if row in DIVIDER_ROWS:
            return col in DOOR_COLS          # the only way through
        return True

    # ── landmarks: room B is the image of room A under the group ────────────
    def _get_landmark_tiles(self) -> dict:
        a = _room_a_pattern()
        m = translate if self.room_symmetry == 'translation' else rotate180
        tiles = dict(a)
        for rc, colour in a.items():
            tiles[m(rc)] = colour
        return {rc: v for rc, v in tiles.items() if self._is_passable(*rc)}

    # ── the group action, for the analysis code ─────────────────────────────
    def group_action(self, pos_rc: tuple[int, int], heading: int) -> tuple[tuple[int, int], int]:
        """Map (position, heading) by the non-identity element of the room group.

        heading uses MiniGrid's convention 0=E, 1=S, 2=W, 3=N.
        """
        if self.room_symmetry == 'translation':
            r, c = pos_rc
            new = (r + ROW_SHIFT, c) if r in ROOM_A_ROWS else (r - ROW_SHIFT, c)
            return new, heading                       # compass is blind to translation
        new = rotate180(pos_rc)
        return new, (heading + 2) % 4                 # compass sees a 180-deg turn


def make_two_room_env(room_symmetry: str, F: int = 7, seed: int = 0):
    return PixelObsWrapper(TwoRoomArena(room_symmetry, F=F, seed=seed), tile_size=1)


def room_of(row: int) -> int | None:
    if row in ROOM_A_ROWS:
        return 0
    if row in ROOM_B_ROWS:
        return 1
    return None


def near_doorway(row: int, col: int, view: int = 7) -> bool:
    """Positions whose egocentric view can see through the doorway -- the only
    place the symmetry is broken. Excluded from the equivalence oracle."""
    half = view // 2
    return (min(abs(row - DIVIDER_ROWS[0]), abs(row - DIVIDER_ROWS[1])) <= half
            and min(abs(col - DOOR_COLS[0]), abs(col - DOOR_COLS[1])) <= half + 1)
