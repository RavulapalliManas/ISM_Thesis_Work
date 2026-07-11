"""The two compartments must be observationally indistinguishable, in BOTH conditions.

If they are not, vision can disambiguate them and the experiment measures nothing: the whole
point is that the only cue separating room A from room B is how the group acts on HEADING.

The old `two_room.py` failed this (max|delta| = 0.6471) because its rooms spanned the arena
width and the agent could see the outer boundary. Closed compartments plus MiniGrid's
`see_through_walls = False` fix it -- but only if the corridor is also locally identical at
each door, which took two attempts:

    18x18, one-cell stub    translation 112/144   rotation 144/144
    24x24, link through B   translation 144/144   rotation  87/144   (B had a second opening)
    24x24, link around B    translation 144/144   rotation 144/144
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from project5_symmetry.environments.compartment_arenas import (A0, ARENA, B0, MODES, SIZE,
                                                               CompartmentArena, group,
                                                               make_compartment_env,
                                                               observation_mismatch)


@pytest.mark.parametrize('mode', MODES)
def test_compartments_are_observationally_identical(mode):
    d = observation_mismatch(mode)
    assert len(d) == SIZE * SIZE * 4
    assert d.max() == 0, (f'{mode}: {int((d > 0).sum())}/{len(d)} views differ '
                          f'(max {d.max()}); vision can tell the compartments apart')


@pytest.mark.parametrize('mode', MODES)
def test_the_agent_can_walk_from_one_compartment_to_the_other(mode):
    env = make_compartment_env(mode); env.reset(seed=0)
    e = env.unwrapped
    start = next(iter(e.room_a))
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (r + dr, c + dc)
            if e._is_passable(*n) and n not in seen:
                seen.add(n); q.append(n)
    assert e.room_b <= seen, f'{mode}: room B is unreachable from room A'


@pytest.mark.parametrize('mode', MODES)
def test_each_compartment_is_a_closed_box_with_exactly_one_door(mode):
    env = make_compartment_env(mode); env.reset(seed=0)
    e = env.unwrapped
    for room, door in ((e.room_a, e.door_a), (e.room_b, e.door_b)):
        openings = set()
        for (r, c) in room:
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (r + dr, c + dc)
                if n not in room and e._is_passable(*n):
                    openings.add(n)
        assert openings == {door}, f'{mode}: room has openings {openings}, expected {{{door}}}'


def test_the_group_acts_on_heading_only_in_the_rotation_condition():
    """This is the manipulation. Translation leaves the compass alone; rotation turns it."""
    _, gh_t = group('translation')
    _, gh_r = group('rotation')
    for h in range(4):
        assert gh_t(h) == h
        assert gh_r(h) == (h + 2) % 4


@pytest.mark.parametrize('mode', MODES)
def test_group_action_is_an_involution_and_swaps_the_rooms(mode):
    env = make_compartment_env(mode); env.reset(seed=0)
    e = env.unwrapped
    for (r, c) in sorted(e.room_a):
        for h in range(4):
            (r2, c2), h2 = e.group_action((r, c), h)
            assert (r2, c2) in e.room_b
            (r3, c3), h3 = e.group_action((r2, c2), h2)
            assert (r3, c3) == (r, c) and h3 == h


def test_the_two_conditions_share_the_room_geometry():
    """Only the corridor and the heading action may differ; the rooms themselves must not."""
    a = CompartmentArena('translation'); b = CompartmentArena('rotation')
    assert a.room_a == b.room_a
    assert a.room_b == b.room_b
    assert a.door_a == b.door_a


def test_landmarks_inside_a_room_are_asymmetric():
    """Position WITHIN a compartment must be identifiable, or the fold is trivial."""
    e = CompartmentArena('translation')
    tiles = e._get_landmark_tiles()
    local = np.zeros((SIZE, SIZE), int)
    for (r, c), v in tiles.items():
        if (r, c) in e.room_a:
            local[r - A0[0], c - A0[1]] = sum(v) * 7 + 1
    for j in (1, 2, 3):
        assert not np.array_equal(np.rot90(local, j), local), 'room pattern is rotationally symmetric'
