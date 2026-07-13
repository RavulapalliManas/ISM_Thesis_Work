"""Annulus arena with C2-symmetric landmarks, instead of the four-distinct-shape s1 scheme
topology_arenas.py uses elsewhere.

Why: does the population manifold measure the topology of X or of X/G? The annulus's own
physical hole (b1=1) is exactly centered, hence invariant under 180-degree rotation, so
pairing it with a C2-symmetric (foldable) landmark layout gives a valid instance of the
Theorem's hypotheses on a looped arena, not just a square one. A folded code should still
report b1=1 (the quotient of an annulus by a free C2 rotation is again an annulus), but the
population trajectory should traverse the neural loop twice per physical revolution, since a
position and its 180-degree image are now written as one place.

The landmark shapes (staircase, cross) and their pairing are copied verbatim from
arena.py's _get_landmark_tiles_s2 -- already validated for the square C2 arena -- only the
hole differs.
"""
from __future__ import annotations

from project5_symmetry.environments.arena import (BLUE, PixelObsWrapper, RED,
                                                  SymmetryArena, _cross_q2, _rotate180,
                                                  _staircase_q1)
from project5_symmetry.environments.topology_arenas import ARENA, is_passable

LAYOUT = 'annulus'   # the only hole geometry used here -- b1 = 1, the loop test case


def s2_landmarks() -> dict:
    q1 = _staircase_q1(BLUE)
    q2 = _cross_q2(RED)
    tiles: dict = {}
    tiles.update(q1)
    tiles.update(_rotate180(q1, N=ARENA))
    tiles.update(q2)
    tiles.update(_rotate180(q2, N=ARENA))
    return tiles


def landmark_tiles_c2() -> dict:
    return {rc: col for rc, col in s2_landmarks().items() if is_passable(LAYOUT, *rc)}


class AnnulusC2Arena(SymmetryArena):
    """The annulus hole (b1=1) with C2-symmetric (foldable) landmarks."""

    def __init__(self, F: int = 7, seed: int = 0, **kw):
        super().__init__(shape='square', size=ARENA, U=0, F=F, seed=seed,
                         use_landmarks=True, symmetry_condition=None, **kw)

    def _is_passable(self, row: int, col: int) -> bool:
        return is_passable(LAYOUT, row, col)

    def _get_landmark_tiles(self) -> dict:
        return landmark_tiles_c2()


def make_annulus_c2_env(F: int = 7, seed: int = 0):
    """Top-level factory: picklable, so `generate_dataset` can rebuild it in each worker."""
    return PixelObsWrapper(AnnulusC2Arena(F=F, seed=seed), tile_size=1)
