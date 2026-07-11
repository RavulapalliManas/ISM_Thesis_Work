"""The four Spiers compartments must be observationally indistinguishable.

Same requirement as the two-room case (`test_compartment_arenas.py`): if vision could tell the
rooms apart the experiment would measure nothing, because the only cue that separates them is
that head direction is invariant under the translation relating them. Every cell of every room,
at every heading, must produce the identical egocentric view.
"""
from __future__ import annotations

import numpy as np

from project5_symmetry.environments.compartment4 import (ARENA, N_ROOMS, SIZE,
                                                         Compartment4Arena,
                                                         make_compartment4_env,
                                                         observation_mismatch)


def test_four_rooms_are_observationally_identical():
    d = observation_mismatch()
    assert d.max() == 0, f'rooms distinguishable by vision: max|delta|={d.max()}'
    assert len(d) == (N_ROOMS - 1) * SIZE * SIZE * 4        # 3 rooms x 36 cells x 4 headings


def test_rooms_are_disjoint_and_sized():
    env = make_compartment4_env()
    env.reset(seed=0)
    e = env.unwrapped
    assert len(e.rooms) == N_ROOMS
    for room in e.rooms:
        assert len(room) == SIZE * SIZE
    all_cells = set().union(*e.rooms)
    assert len(all_cells) == N_ROOMS * SIZE * SIZE          # no overlap between rooms


def test_room_of_maps_cells_back():
    env = make_compartment4_env()
    env.reset(seed=0)
    e = env.unwrapped
    for i, room in enumerate(e.rooms):
        for cell in room:
            assert e.room_of(cell) == i
    assert e.room_of((0, 0)) == -1                          # outside any room
