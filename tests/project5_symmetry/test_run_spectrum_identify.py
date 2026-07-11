"""Provenance of each spectrum row must be recoverable, whatever wrote the checkpoint.

Group A (run_ensemble_sweep) records no condition/hd_mode/seed in `meta` -- only the
directory layout knows. Group B and the horizon runs record all three. Mislabelling a
row silently swaps conditions and inverts the headline result, so pin every layout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from project5_symmetry.analysis.run_spectrum import identify


def test_group_a_layout_is_recovered_from_the_path():
    """.../symmetry_sweep/{cond}/seed_NN/ckpt_final.pt, meta has none of it."""
    p = Path('/root/runs/symmetry_sweep/s2/seed_03/ckpt_final.pt')
    assert identify(p, {'k': 5}) == ('s2', 'full', 3)


def test_group_b_layout_prefers_metadata():
    p = Path('/root/runs/hd_invariance/s4/parity/seed_05/ckpt_final.pt')
    meta = {'condition': 's4', 'hd_mode': 'parity', 'seed': 5, 'k': 5}
    assert identify(p, meta) == ('s4', 'parity', 5)


def test_horizon_layout_has_a_k_dir_that_must_not_confuse_it():
    p = Path('/root/runs/horizon/k0/s2/axis/seed_01/ckpt_final.pt')
    assert identify(p, {'condition': 's2', 'hd_mode': 'axis', 'seed': 1, 'k': 0}) == ('s2', 'axis', 1)
    # and from the path alone
    assert identify(p, {'k': 0}) == ('s2', 'axis', 1)


def test_metadata_wins_over_a_misleading_path():
    p = Path('/tmp/scratch/s1/full/seed_00/ckpt_final.pt')
    meta = {'condition': 's4', 'hd_mode': 'const', 'seed': 7, 'k': 3}
    assert identify(p, meta) == ('s4', 'const', 7)


def test_unidentifiable_condition_raises_rather_than_guessing():
    with pytest.raises(ValueError, match='cannot identify condition'):
        identify(Path('/tmp/whatever/ckpt_final.pt'), {'k': 5})
