"""The arenas must have the topology we claim, AND the generated data must respect them.

Three ways this silently went wrong once and must never again:
  1. a hole touches the boundary or another hole -> holes merge, b1 is not what we think;
  2. `generate_dataset` rebuilds the env in each worker from kwargs, DOWNCASTING the
     subclass to a plain SymmetryArena -- the walls vanish, the agent walks through them,
     and every arena becomes the same open box (this happened: 100/196/80 illegal cells);
  3. the landmarks are rotationally symmetric -> the code folds, confounding topology with
     the symmetry-folding result.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from project5_symmetry.environments.generate_trajectories import generate_dataset
from project5_symmetry.environments.topology_arenas import (ARENA, EXPECTED_B1, LAYOUTS,
                                                            betti, is_passable,
                                                            landmark_id_map,
                                                            landmark_tiles,
                                                            make_topology_env,
                                                            s1_landmarks)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_betti_numbers_are_as_claimed(layout):
    b0, b1 = betti(layout)
    assert b0 == 1, f'{layout} is disconnected (b0={b0}); the agent cannot reach all of it'
    assert b1 == EXPECTED_B1[layout], f'{layout}: b1={b1}, expected {EXPECTED_B1[layout]}'


@pytest.mark.parametrize('layout', LAYOUTS)
def test_holes_are_strictly_interior(layout):
    for i in range(1, ARENA + 1):
        for r, c in ((1, i), (ARENA, i), (i, 1), (i, ARENA)):
            assert is_passable(layout, r, c), f'{layout}: border cell ({r},{c}) is a wall'


@pytest.mark.parametrize('layout', LAYOUTS)
def test_landmarks_break_every_rotation(layout):
    m = landmark_id_map(layout)
    for j in (1, 2, 3):
        assert not np.array_equal(np.rot90(m, j), m), \
            f'{layout} landmark map is invariant under {90*j} degrees'


@pytest.mark.parametrize('layout', LAYOUTS)
def test_all_four_landmark_shapes_survive_the_holes(layout):
    """The holes were sized to preserve the anchors. Every colour must still be present,
    and most tiles must survive, or the arena loses its anchor representation."""
    kept = landmark_tiles(layout)
    colours = {tuple(v) for v in kept.values()}
    assert len(colours) == 4, f'{layout}: only {len(colours)} landmark colours survive'
    assert len(kept) >= 0.85 * len(s1_landmarks()), \
        f'{layout}: only {len(kept)}/{len(s1_landmarks())} landmark tiles survive'


@pytest.mark.parametrize('layout', LAYOUTS)
def test_landmarks_never_sit_on_a_wall(layout):
    for (r, c) in landmark_tiles(layout):
        assert is_passable(layout, r, c)


def test_theta_and_figure8_are_homotopy_equivalent_but_different_shapes():
    assert EXPECTED_B1['theta'] == EXPECTED_B1['figure8'] == 2
    n = lambda l: sum(is_passable(l, r, c) for r in range(1, ARENA + 1) for c in range(1, ARENA + 1))
    assert n('theta') != n('figure8')


# ── the regression that matters: the data must live inside the arena ──────────────────

@pytest.mark.parametrize('layout', ['annulus', 'theta'])
def test_generated_trajectories_never_enter_a_wall(layout):
    """`generate_dataset` rebuilds the env inside each worker. If it downcasts the subclass,
    the walls disappear and the agent walks straight through them. This is the check that
    was missing when all seven topology datasets came out as the same open square."""
    tmp = Path(tempfile.mkdtemp())
    try:
        generate_dataset(make_topology_env(layout, F=7, seed=0), n_traj=4, T=60,
                         out_dir=str(tmp), n_workers=2, desc='test',
                         env_factory=make_topology_env,
                         factory_kwargs={'layout': layout, 'F': 7, 'seed': 0})
        files = sorted(tmp.glob('*.npz'))
        assert len(files) == 4
        pos = np.concatenate([np.load(f)['pos'] for f in files]).astype(int)
        illegal = [(x, y) for x, y in pos if not is_passable(layout, y, x)]   # pos is (col,row)
        assert not illegal, f'{layout}: {len(illegal)} positions inside walls, e.g. {illegal[:3]}'

        visited = {tuple(p) for p in pos}
        n_passable = sum(is_passable(layout, r, c)
                         for r in range(1, ARENA + 1) for c in range(1, ARENA + 1))
        assert len(visited) <= n_passable
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_generate_dataset_refuses_a_subclass_without_a_factory():
    """The downcast must be impossible to trigger by accident."""
    tmp = Path(tempfile.mkdtemp())
    try:
        with pytest.raises(TypeError, match='silently downcast'):
            generate_dataset(make_topology_env('annulus', F=7, seed=0), n_traj=2, T=10,
                             out_dir=str(tmp), n_workers=1, desc='test')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
