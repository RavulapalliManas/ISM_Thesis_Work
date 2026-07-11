r"""City-block maze for the model-data correspondence with Hockeimer et al. (2023).

A rectilinear lattice of 1-cell-wide alleys: cells lie on a grid line (a passable alley) or in an
enclosed block (wall). Alleys carry no distinguishing landmarks, so every horizontal alley looks
like every other horizontal alley and every vertical alley like every other vertical one --- the
geometric repetition that drives place-field repetition in the rodent maze.

The prediction we test here is our own model's: two alleys related by TRANSLATION (same
orientation -> same travel axis -> the head-direction signal reads the same) are
HD-indistinguishable and fold onto a shared directional field; alleys related by ROTATION
(different orientation) lift apart. `cell_orientation` labels each passable cell H or V (or X at
an intersection) so the readout can group fields exactly as the data reanalysis does.
"""
from __future__ import annotations

import numpy as np

from project5_symmetry.environments.arena import PixelObsWrapper, SymmetryArena

ARENA = 18
GRID_LINES = (3, 7, 11, 15)          # 4x4 intersection lattice, alley segments 3 cells long


def _passable_set():
    s = set()
    for r in range(1, ARENA + 1):
        for c in range(1, ARENA + 1):
            if r in GRID_LINES or c in GRID_LINES:
                s.add((r, c))
    return s


def cell_orientation(r, c):
    """H = horizontal alley (row on a grid line), V = vertical, X = intersection (both)."""
    on_r, on_c = r in GRID_LINES, c in GRID_LINES
    if on_r and on_c:
        return 'X'
    if on_r:
        return 'H'
    if on_c:
        return 'V'
    return None


class CityBlockArena(SymmetryArena):
    """Lattice of identical 1-cell alleys; blocks are walls."""

    def __init__(self, F: int = 7, seed: int = 0, **kw):
        self.passable = _passable_set()
        super().__init__(shape='square', size=ARENA, U=0, F=F, seed=seed,
                         use_landmarks=False, symmetry_condition=None, **kw)

    def _is_passable(self, row: int, col: int) -> bool:
        return (row, col) in self.passable

    def _get_landmark_tiles(self) -> dict:
        return {}                    # identical alleys: no landmarks


def make_cityblock_env(F: int = 7, seed: int = 0):
    return PixelObsWrapper(CityBlockArena(F=F, seed=seed), tile_size=1)
