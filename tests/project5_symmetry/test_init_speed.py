"""`steps_to_threshold` decides the whole init study, so pin its edge cases.

The dangerous failure is silent: a model that never reaches the threshold must read `nan`,
not the last step, or "never learned" is scored identically to "learned at the very end".
"""
from __future__ import annotations

import numpy as np
import pytest

from project5_symmetry.analysis.init_speed import smooth, steps_to_threshold


def test_crossing_is_interpolated_not_snapped_to_the_logging_grid():
    steps = [0, 100, 200]
    loss = [1.0, 1.0, 0.0]              # smoothing off via w=1
    t = steps_to_threshold(steps, loss, thresh=0.5, w=1)
    assert t == pytest.approx(150.0)


def test_never_reaching_the_threshold_is_nan_not_the_last_step():
    steps = [0, 100, 200]
    loss = [1.0, 0.9, 0.8]
    assert np.isnan(steps_to_threshold(steps, loss, thresh=0.1, w=1))


def test_already_below_at_step_zero():
    assert steps_to_threshold([0, 100], [0.01, 0.01], thresh=0.5, w=1) == 0.0


def test_faster_model_gets_a_smaller_number():
    steps = list(range(0, 1000, 100))
    fast = [1.0 * np.exp(-s / 100) for s in steps]
    slow = [1.0 * np.exp(-s / 500) for s in steps]
    tf = steps_to_threshold(steps, fast, thresh=0.3, w=1)
    ts = steps_to_threshold(steps, slow, thresh=0.3, w=1)
    assert tf < ts


def test_smoothing_does_not_shift_a_monotone_curve_much():
    steps = np.arange(0, 2000, 100)
    loss = np.exp(-steps / 500)
    raw = steps_to_threshold(steps, loss, thresh=0.3, w=1)
    sm = steps_to_threshold(steps, loss, thresh=0.3, w=3)
    assert abs(raw - sm) < 150


def test_a_single_noise_spike_does_not_trigger_an_early_crossing():
    steps = list(range(0, 1100, 100))
    loss = [1.0] * 11
    loss[3] = 0.0                        # one bad log line
    assert np.isnan(steps_to_threshold(steps, loss, thresh=0.5, w=5))
    assert steps_to_threshold(steps, loss, thresh=0.5, w=1) == pytest.approx(250.0)


def test_smooth_preserves_length():
    assert len(smooth([1.0] * 10, w=5)) == 10
    assert len(smooth([1.0, 2.0], w=5)) == 2
