"""The seed block must EXTEND the ensemble, never retrace or overwrite it.

Seeds here are deterministic: `--n-seeds 8` recomputes seeds 0..3 bit-for-bit and writes them
over themselves, burning ~45 GPU-minutes to learn nothing. `--seed-offset 4 --n-seeds 4` adds
seeds 4..7 for the same cost. These tests pin that, and pin the paired design that the init
sweep depends on.
"""
from __future__ import annotations

from project5_symmetry.experiments.run_multi import (INIT_VARIANTS, KAIMING_VARIANTS,
                                                     build_spec, torch_seed)

GROUPS = ('topology', 'init', 'kaiming', 'compartment')


def _key(e):
    return (e['group'], e['name'], e['seed'])


def test_default_offset_is_the_original_behaviour():
    spec = build_spec(4, GROUPS)
    assert len(spec) == 64
    assert sorted({e['seed'] for e in spec}) == [0, 1, 2, 3]


def test_offset_block_is_disjoint_from_the_base_block():
    base = build_spec(4, GROUPS, 0)
    ext = build_spec(4, GROUPS, 4)
    assert len(ext) == len(base) == 64
    assert sorted({e['seed'] for e in ext}) == [4, 5, 6, 7]
    assert not ({_key(e) for e in base} & {_key(e) for e in ext}), \
        'the offset block would overwrite existing checkpoints'


def test_offset_block_draws_new_networks():
    """Disjoint output dirs are not enough -- the torch seeds must differ too, or the 'new'
    seeds are copies of the old ones under a different filename."""
    base = {_key(e): torch_seed(e) for e in build_spec(4, GROUPS, 0)}
    ext = {_key(e): torch_seed(e) for e in build_spec(4, GROUPS, 4)}
    assert not (set(base.values()) & set(ext.values())), 'extended seeds reuse a base draw'


def test_init_and_kaiming_stay_paired_within_a_seed():
    """Only the recurrent matrix may differ between init arms: W_in / W_out / bias come from
    the same draw, so torch_seed must be the seed alone. Break this and the sweep stops being
    a controlled comparison."""
    spec = build_spec(4, ('init', 'kaiming'), 4)
    for s in range(4, 8):
        arms = {e['name']: torch_seed(e) for e in spec if e['seed'] == s}
        assert set(arms) == set(INIT_VARIANTS) | set(KAIMING_VARIANTS)
        assert len(set(arms.values())) == 1, f'init arms unpaired at seed {s}: {arms}'


def test_topology_and_compartment_arms_get_distinct_draws_per_arena():
    spec = build_spec(2, ('topology', 'compartment'), 4)
    per_seed = {}
    for e in spec:
        per_seed.setdefault(e['seed'], set()).add(torch_seed(e))
    for s, seeds in per_seed.items():
        n = sum(1 for e in spec if e['seed'] == s)
        assert len(seeds) == n, f'two arenas share a draw at seed {s}'
