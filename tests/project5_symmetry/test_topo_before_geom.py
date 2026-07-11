"""Saturation-step logic decides the topology-before-geometry claim. Pin its failure modes.

The dangerous one is a flickering indicator: b1 that reads correct at step 250 by luck,
wrong at 500, correct again at 1000 has not "locked in at 250". `first_stable` must require
the property to hold from that step onward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project5_symmetry.analysis.topo_before_geom import (first_stable, per_model,
                                                         saturation_step)


def test_first_stable_ignores_a_lucky_early_hit_that_does_not_persist():
    steps = [0, 250, 500, 1000, 2000]
    ok = [False, True, False, True, True]        # the 250 hit does not hold
    assert first_stable(steps, ok) == 1000.0


def test_first_stable_is_step_zero_when_correct_throughout():
    assert first_stable([0, 100, 200], [True, True, True]) == 0.0


def test_first_stable_is_nan_when_never_stable():
    assert np.isnan(first_stable([0, 100, 200], [True, True, False]))
    assert np.isnan(first_stable([0, 100], [False, False]))


def test_saturation_step_finds_the_first_persistent_crossing():
    steps = [0, 100, 200, 300]
    vals = [0.0, 0.5, 0.95, 1.0]                 # frac=0.9 of final 1.0 -> 0.9
    assert saturation_step(steps, vals, frac=0.9) == 200.0


def test_saturation_step_rejects_a_transient_spike():
    steps = [0, 100, 200, 300]
    vals = [0.95, 0.2, 0.5, 1.0]                 # early spike, then falls back
    assert saturation_step(steps, vals, frac=0.9) == 300.0


def test_saturation_step_is_nan_when_the_metric_never_rose():
    assert np.isnan(saturation_step([0, 100], [0.0, 0.0], frac=0.9))
    assert np.isnan(saturation_step([0, 100], [-0.2, -0.3], frac=0.9))


def _frame(b1_hat, metric, b1_true=1, steps=(0, 250, 500, 1000)):
    return pd.DataFrame({'layout': 'annulus_w4', 'seed': 0, 'step': list(steps),
                         'b1_true': b1_true, 'b1_hat': b1_hat, 'metric': metric,
                         'decode_r2': metric})


def test_per_model_reports_topology_inherited_from_the_initialisation():
    """b1 correct at step 0 => topology was never learned."""
    pm = per_model(_frame([1, 1, 1, 1], [0.1, 0.4, 0.8, 0.9]))
    assert pm.topo_step.iloc[0] == 0.0
    assert pm.b1_at_init.iloc[0] == 1
    assert pm.geom_step.iloc[0] == 1000.0        # metric only reaches 0.9*final at the end


def test_per_model_reports_topology_learned_before_geometry():
    pm = per_model(_frame([0, 1, 1, 1], [0.05, 0.1, 0.5, 0.9]))
    assert pm.topo_step.iloc[0] == 250.0
    assert pm.geom_step.iloc[0] == 1000.0
    assert pm.b1_at_init.iloc[0] == 0


def test_per_model_drops_the_duplicate_final_checkpoint():
    df = _frame([0, 1, 1, 1], [0.05, 0.1, 0.5, 0.9])
    dup = df.iloc[[-1]].copy(); dup['step'] = 'final'
    pm = per_model(pd.concat([df, dup], ignore_index=True))
    assert len(pm) == 1
    assert pm.topo_step.iloc[0] == 250.0


def test_per_model_can_report_geometry_before_topology():
    """The test must be able to come out the other way, or it proves nothing."""
    pm = per_model(_frame([0, 0, 1, 1], [0.9, 0.95, 0.98, 1.0]))
    assert pm.geom_step.iloc[0] == 0.0
    assert pm.topo_step.iloc[0] == 500.0
    assert pm.geom_step.iloc[0] < pm.topo_step.iloc[0]
