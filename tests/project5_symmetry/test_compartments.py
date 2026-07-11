"""Room decoding and the repetition index must be right, or the Grieves replication says nothing.

The subtle one is `label_rooms`: room B's local coordinates must be pulled back THROUGH the group
element, so that local (i, j) names the same place within a compartment in both rooms. Get that
wrong and a repeating code looks like a remapping one, because you are correlating a unit's map in
A against a rotated copy of its map in B.
"""
from __future__ import annotations

import numpy as np
import pytest

from project5_symmetry.analysis.run_compartments import (decode_room, label_rooms,
                                                         repetition_index)
from project5_symmetry.environments.compartment_arenas import A0, B0, SIZE, group

N_UNITS = 60


def _states(mode, reps=40):
    """Every local cell of both rooms, `reps` times."""
    g, _ = group(mode)
    pos, room, loc = [], [], []
    for i in range(SIZE):
        for j in range(SIZE):
            gi, gj = g(i, j)
            for _ in range(reps):
                pos.append((A0[1] + j, A0[0] + i)); room.append(0); loc.append((i, j))
                pos.append((B0[1] + gj, B0[0] + gi)); room.append(1); loc.append((i, j))
    return np.array(pos), np.array(room), np.array(loc)


def _code(loc, room, regime, seed=0, noise=0.02):
    """Gaussian place fields over the local cell, in one of three regimes.

    fold          same field centres, same rates in both rooms.  Repetition 1, room at chance.
    global_remap  DIFFERENT field centres per room.               Repetition low, room decodable.
    rate_remap    same centres, different gains per room.         Repetition 1, room decodable.

    The third is the one that matters: Pearson correlation is invariant to a per-room gain, so a
    rate-remapped code repeats spatially AND names the room. Spiers et al. report exactly this
    ("high spatial repetition with a slight degree of rate-based discrimination"). Repetition and
    room-decodability are independent axes and must both be reported.
    """
    rng = np.random.default_rng(seed)
    C = rng.uniform(0, SIZE - 1, (N_UNITS, 2))
    D2 = ((loc[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    H = np.exp(-D2 / 2.0)
    if regime == 'global_remap':
        C2 = rng.uniform(0, SIZE - 1, (N_UNITS, 2))
        D2b = ((loc[:, None, :] - C2[None, :, :]) ** 2).sum(-1)
        H = np.where(room[:, None] == 0, H, np.exp(-D2b / 2.0))
    elif regime == 'rate_remap':
        gain = 1.0 + 2.0 * (np.arange(N_UNITS)[None, :] % 2)
        H = H * np.where(room[:, None] == 0, 1.0, gain)
    elif regime != 'fold':
        raise ValueError(regime)
    return H + rng.normal(0, noise, H.shape)


@pytest.mark.parametrize('mode', ['translation', 'rotation'])
def test_label_rooms_pulls_room_b_back_through_the_group(mode):
    pos, room, loc = _states(mode, reps=1)
    r, l = label_rooms(pos, mode)
    assert np.array_equal(r, room)
    assert np.array_equal(l, loc), 'room B local coords must be the group pre-image'


def test_label_rooms_marks_corridor_cells_as_minus_one():
    pos = np.array([[1, 1], [22, 22]])               # far outside either compartment
    r, _ = label_rooms(pos, 'translation')
    assert (r == -1).all()


@pytest.mark.parametrize('mode', ['translation', 'rotation'])
def test_a_folded_code_gives_chance_room_decoding_and_repetition_one(mode):
    pos, room, loc = _states(mode)
    H = _code(loc, room, 'fold')
    gen, seen, within = decode_room(H, room, loc, seed=0)
    rep, ncells = repetition_index(H, room, loc)
    assert gen == pytest.approx(0.5, abs=0.08), f'folded code decoded room at {gen}'
    assert seen == pytest.approx(0.5, abs=0.08), f'folded code: even a seen-cell decoder fails? {seen}'
    assert rep > 0.95, f'folded code should repeat perfectly; got {rep}'
    assert within > 0.8, 'positive control: within-room position must still decode'
    assert ncells == SIZE * SIZE


@pytest.mark.parametrize('mode', ['translation', 'rotation'])
def test_global_remapping_kills_repetition_and_is_decodable_only_where_seen(mode):
    """Global remapping writes room identity differently at every cell, so there is no room
    direction to generalise: `room_gen` sits at or below chance while `room_seen` is at ceiling.
    Only `repetition` and `room_seen` detect it."""
    pos, room, loc = _states(mode)
    H = _code(loc, room, 'global_remap')
    gen, seen, within = decode_room(H, room, loc, seed=0)
    rep, _ = repetition_index(H, room, loc)
    assert seen > 0.9, f'a kNN seen-cell decoder should name the room; got {seen}'
    assert gen <= 0.55, f'no generalisable room direction should exist; got {gen}'
    assert seen - gen > 0.35, 'the seen/gen gap IS the signature of global remapping'
    assert rep < 0.5, f'remapped code should not repeat; got {rep}'
    assert within > 0.8


@pytest.mark.parametrize('mode', ['translation', 'rotation'])
def test_rate_remapping_repeats_spatially_yet_still_names_the_room(mode):
    """Spiers' actual finding. Repetition and room-decodability are INDEPENDENT: a per-room gain
    leaves the rate-map correlation at 1 while the room stays perfectly decodable. Never read a
    high repetition index as 'the code cannot tell the rooms apart'."""
    pos, room, loc = _states(mode)
    H = _code(loc, room, 'rate_remap')
    gen, seen, _ = decode_room(H, room, loc, seed=0)
    rep, _ = repetition_index(H, room, loc)
    assert rep > 0.95, f'rate remapping must leave spatial repetition intact; got {rep}'
    assert gen > 0.9, f'a consistent gain direction must generalise; got {gen}'
    assert seen > 0.9


def test_the_readouts_can_come_out_either_way():
    """Guard against metrics that always say 'folded'."""
    pos, room, loc = _states('translation')
    _, f_seen, _ = decode_room(_code(loc, room, 'fold'), room, loc, seed=0)
    _, u_seen, _ = decode_room(_code(loc, room, 'global_remap'), room, loc, seed=0)
    assert u_seen - f_seen > 0.35, f'fold {f_seen:.3f} vs global remap {u_seen:.3f}'
    f_rep, _ = repetition_index(_code(loc, room, 'fold'), room, loc)
    u_rep, _ = repetition_index(_code(loc, room, 'global_remap'), room, loc)
    assert f_rep - u_rep > 0.4


def test_repetition_index_is_nan_when_too_few_cells_are_sampled():
    pos, room, loc = _states('translation', reps=1)
    H = _code(loc, room, 'fold')
    rep, n = repetition_index(H, room, loc, min_count=99)   # nothing passes the threshold
    assert np.isnan(rep) and n == 0
