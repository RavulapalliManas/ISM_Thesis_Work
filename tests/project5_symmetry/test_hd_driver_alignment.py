"""Rows of the sampled batch must line up positionally with `spec`.

If they do not, every model trains on the wrong arena and the run looks completely
normal -- losses fall, checkpoints save, nothing errors. Pin it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.hd_encodings import hd_matrix  # noqa: E402
from project5_symmetry.experiments.run_hd_invariance import (DEFAULT_SEEDS,  # noqa: E402
                                                             build_spec,
                                                             init_seed,
                                                             sample_batches)

T, B = 6, 2


class FakeStore:
    """Emits a constant obs equal to a per-condition tag, and one-hot headings."""

    def __init__(self, tag):
        self.tag = tag

    def sample_parallel_batches(self, n, b):
        obs = torch.full((n, b, T + 1, 3), float(self.tag))
        hd = torch.eye(4)[torch.arange(T) % 4]                  # (T,4) cycles E,S,W,N
        act = torch.cat([torch.ones(T, 1), hd], -1)             # (T,5)
        return obs, act.expand(n, b, T, 5).clone()


def _stores():
    return {'s1': FakeStore(1), 's2': FakeStore(2), 's4': FakeStore(4)}


@pytest.mark.parametrize('seeds', [
    {'s1': 2, 's2': 2, 's4': 2},          # canonical order
    {'s2': 2, 's1': 2},                   # permuted -- the bug this test exists for
    {'s4': 1, 's2': 3, 's1': 2},          # permuted AND ragged
    {'s2': 2},                            # single condition
])
def test_obs_rows_match_spec_condition_under_any_condition_order(seeds):
    spec = build_spec(seeds, ('full', 'axis'))
    hd_stack = torch.stack([hd_matrix(h) for _, h, _ in spec])
    obs, _ = sample_batches(_stores(), spec, B, hd_stack)
    assert obs.shape[0] == len(spec)
    for i, (cond, _, _) in enumerate(spec):
        tag = {'s1': 1.0, 's2': 2.0, 's4': 4.0}[cond]
        assert obs[i].unique().tolist() == [tag], f'row {i} is {cond} but carries tag {obs[i].unique()}'


def test_default_allocation_powers_the_decisive_contrast():
    """s2 axis-vs-parity is the headline. n=6 vs 6 gives an exact Mann-Whitney floor
    of 2/C(12,6) = 0.0022, which survives correction; n=4 vs 4 gives 0.029, which does not."""
    from math import comb
    assert DEFAULT_SEEDS['s2'] >= 6
    n = DEFAULT_SEEDS['s2']
    assert 2 / comb(2 * n, n) < 0.005
    assert len(build_spec(DEFAULT_SEEDS, ('full', 'axis', 'parity', 'const'))) == 48


def test_hd_transform_is_applied_per_model_row():
    spec = build_spec({'s2': 1}, ('full', 'axis', 'parity', 'const'))
    hd_stack = torch.stack([hd_matrix(h) for _, h, _ in spec])
    _, act = sample_batches(_stores(), spec, B, hd_stack)

    assert torch.equal(act[..., :1], torch.ones_like(act[..., :1]))       # speed preserved
    full, axis, parity, const = act[0, 0], act[1, 0], act[2, 0], act[3, 0]

    assert torch.allclose(full[..., 1:].sum(-1), torch.ones(T))           # still one-hot
    assert torch.allclose(const[..., 1:], torch.full((T, 4), 0.25))       # 0 bits
    # heading t and heading t+2 are opposite: axis folds them, parity does not
    assert torch.allclose(axis[0, 1:], axis[2, 1:])
    assert not torch.allclose(parity[0, 1:], parity[2, 1:])


def test_init_seeds_are_unique_across_cells():
    hds = ('full', 'axis', 'parity', 'const')
    spec = build_spec(DEFAULT_SEEDS, hds)
    seeds = [init_seed(c, h, s, list(DEFAULT_SEEDS), hds) for c, h, s in spec]
    assert len(set(seeds)) == len(seeds) == 48


def test_init_seed_does_not_depend_on_the_runs_condition_subset():
    """A cell's init must be a property of the cell, not of what else was in the run.

    Group B ran all three conditions; the s1 extension runs s1 alone; a future s4-only
    extension must still get s4's inits, not s1's.
    """
    hds = ('full', 'axis', 'parity', 'const')
    for cond in ('s1', 's2', 's4'):
        full_run = init_seed(cond, 'axis', 3, ['s1', 's2', 's4'], hds)
        alone = init_seed(cond, 'axis', 3, [cond], hds)
        assert full_run == alone, f'{cond} init depends on the run composition'

    # and distinct conditions never collide
    seeds = {init_seed(c, 'axis', 3, [c], hds) for c in ('s1', 's2', 's4')}
    assert len(seeds) == 3


def test_group_b_inits_are_unchanged_by_the_fix():
    """Group B ran with conditions=['s1','s2','s4'], which already matched CONDITIONS.
    The fix must not silently redefine what those 48 trained models were."""
    hds = ('full', 'axis', 'parity', 'const')
    for ci, c in enumerate(('s1', 's2', 's4')):
        for hi, h in enumerate(hds):
            for s in range(6):
                assert init_seed(c, h, s, ['s1', 's2', 's4'], hds) == 1000 * ci + 100 * hi + s
