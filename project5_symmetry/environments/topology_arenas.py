"""Arenas with controlled topology, built around the project's own landmark patterns.

Design constraints, in the order that fixed them
-----------------------------------------------
1. **Landmarks.** The four `s1` shapes -- staircase (blue, Q1), cross (red, Q2),
   chevron (green, Q3), castle (yellow, Q4) -- are reused verbatim from `arena.py`.
   Four distinct shapes in four quadrants means no rotation of the arena maps the
   landmark map onto itself, so nothing folds and topology is never confounded with
   the symmetry result.

2. **Hole size.** The shapes are 6-7 cells across and live near the arena border. A
   large central hole destroys them. Calibrated against an ideal place code with the
   network's measured field width (autocorrelation half-width 4.0 cells, sigma ~ 2.4),
   persistent H1 already separates an annulus from an open box at a 4x4 hole:

       hole   corridor   L1 / (b1=0 floor)   s1 tiles surviving
        4x4      7            4.75                93 / 94
        6x6      6            4.75                88 / 94
        8x8      5            5.17                81 / 94
       10x10     4            5.55                64 / 94

   A 6x6 central hole keeps 88 of 94 landmark tiles and is comfortably detectable.
   Bigger holes buy almost nothing and cost anchors.

3. **Topology.**

       layout    b0  b1  cells  landmarks kept  H1 / (b1=0 floor)
       open       1   0    324      94 / 94            1.00
       annulus    1   1    288      88 / 94            4.75
       theta      1   2    276      82 / 94            4.66   (stacked holes, bridged)
       figure8    1   2    292      91 / 94            4.87   (side-by-side holes)

   The last column is the longest H1 bar for an ideal place code of that arena at the
   network's measured field width, in units of the open arena's noise floor. Every loop
   arena is comfortably separable; `open` is the null.

   `theta` and `figure8` are HOMOTOPY EQUIVALENT (both b1 = 2); homology cannot tell
   them apart. That is the point -- they are a geometry control, testing that a
   topological readout is invariant to shape, as it must be.

Betti numbers are of the planar region, i.e. of the CUBICAL complex: unit squares whose
four corners are passable are filled in. The cycle rank of the 4-connectivity *graph* is
not b1 (it counts every unit square as a hole).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from project5_symmetry.environments.arena import (BLUE, GREEN, PixelObsWrapper, RED,
                                                  SymmetryArena, YELLOW, _castle_q4,
                                                  _chevron_q3, _cross_q2, _staircase_q1)

ARENA = 18

# Rectangular holes, 1-indexed inclusive (r0, r1, c0, c1). Strictly interior, and no two
# holes 4-adjacent (touching holes merge into one, changing b1).
HOLES: dict[str, list[tuple[int, int, int, int]]] = {
    'open':    [],
    'annulus': [(7, 12, 7, 12)],                    # one 6x6 central block
    'theta':   [(5, 8, 7, 12), (11, 14, 7, 12)],    # two stacked blocks, bridge at rows 9-10
    'figure8': [(8, 11, 4, 7), (8, 11, 11, 14)],    # two side-by-side blocks
}
# Holes must not touch, not even at a corner: two blocks meeting diagonally form ONE
# connected wall and bound ONE hole (measured: b1 = 1, not 2). The test pins this.

EXPECTED_B1 = {'open': 0, 'annulus': 1, 'theta': 2, 'figure8': 2}
LAYOUTS = tuple(HOLES)


def s1_landmarks() -> dict:
    """The project's four asymmetric landmark shapes, one per quadrant."""
    tiles: dict[tuple[int, int], list] = {}
    for build, colour in ((_staircase_q1, BLUE), (_cross_q2, RED),
                          (_chevron_q3, GREEN), (_castle_q4, YELLOW)):
        tiles.update(build(colour))
    return tiles


def is_passable(layout: str, row: int, col: int) -> bool:
    if not (1 <= row <= ARENA and 1 <= col <= ARENA):
        return False
    return not any(r0 <= row <= r1 and c0 <= col <= c1 for r0, r1, c0, c1 in HOLES[layout])


def betti(layout: str) -> tuple[int, int]:
    """(b0, b1) of the planar region, via the cubical complex: b1 = b0 - V + E - F."""
    cells = {(r, c) for r in range(1, ARENA + 1) for c in range(1, ARENA + 1)
             if is_passable(layout, r, c)}
    V = len(cells)
    E = sum(1 for (r, c) in cells for d in ((0, 1), (1, 0)) if (r + d[0], c + d[1]) in cells)
    F = sum(1 for (r, c) in cells if {(r, c + 1), (r + 1, c), (r + 1, c + 1)} <= cells)
    seen, b0 = set(), 0
    for s in cells:
        if s in seen:
            continue
        b0 += 1
        q = deque([s]); seen.add(s)
        while q:
            r, c = q.popleft()
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if (r + dr, c + dc) in cells and (r + dr, c + dc) not in seen:
                    seen.add((r + dr, c + dc)); q.append((r + dr, c + dc))
    return b0, b0 - V + E - F


def landmark_tiles(layout: str) -> dict:
    """The s1 shapes, clipped to the passable cells of this layout."""
    return {rc: col for rc, col in s1_landmarks().items() if is_passable(layout, *rc)}


def landmark_id_map(layout: str) -> np.ndarray:
    """(ARENA, ARENA) int array: -1 wall, 0 plain floor, 1..4 landmark colour id."""
    m = np.full((ARENA, ARENA), -1)
    for r in range(1, ARENA + 1):
        for c in range(1, ARENA + 1):
            if is_passable(layout, r, c):
                m[r - 1, c - 1] = 0
    order = {tuple(c): i + 1 for i, c in enumerate(sorted(map(tuple, (BLUE, RED, GREEN, YELLOW))))}
    for (r, c), col in landmark_tiles(layout).items():
        m[r - 1, c - 1] = order[tuple(col)]
    return m


def render(layout: str) -> str:
    sym = {-1: '#', 0: '.', 1: 'a', 2: 'b', 3: 'c', 4: 'd'}
    m = landmark_id_map(layout)
    return '\n'.join(''.join(sym[v] for v in m[r]) for r in range(ARENA))


class TopologyArena(SymmetryArena):
    """A square arena with rectangular holes punched out of it, carrying the s1 landmarks."""

    def __init__(self, layout: str, F: int = 7, seed: int = 0, **kw):
        if layout not in HOLES:
            raise ValueError(f'unknown layout {layout!r}; want one of {LAYOUTS}')
        self.layout = layout
        super().__init__(shape='square', size=ARENA, U=0, F=F, seed=seed,
                         use_landmarks=True, symmetry_condition=None, **kw)

    def _is_passable(self, row: int, col: int) -> bool:
        return is_passable(self.layout, row, col)

    def _get_landmark_tiles(self) -> dict:
        return landmark_tiles(self.layout)


def make_topology_env(layout: str, F: int = 7, seed: int = 0):
    """Top-level factory: picklable, so `generate_dataset` can rebuild it in each worker."""
    return PixelObsWrapper(TopologyArena(layout, F=F, seed=seed), tile_size=1)
