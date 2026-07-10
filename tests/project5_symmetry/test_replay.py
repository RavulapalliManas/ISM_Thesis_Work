"""The offline loop must be the SAME dynamical system as the model's own forward.

If `manual_rollout` does not reproduce `pRNN_th`'s hidden states when driven by the real
observations, then `offline_rollout` is simulating some other network and every replay number
is fiction. This is the only test that matters here; the rest are sanity checks on the readouts.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from project5_symmetry.analysis.run_replay import (READOUTS, manual_rollout, offline_rollout,
                                                   path_stats, random_actions, summary_row)
from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCellEager

OBS, ACT, H, T = 24, 5, 64, 15


def _model(seed=0):
    torch.manual_seed(seed)
    return pRNN_th(obs_size=OBS, act_size=ACT, k=1, hidden_size=H,
                   cell=LayerNormRNNCellEager, dropp=0.0, trunc=T,
                   neuralTimescale=2, predOffset=0, hidden_init_sigma=0.1).eval()


def test_manual_rollout_reproduces_the_models_own_hidden_states():
    """Bit-for-bit against pRNN_th.forward with noise off and dropout off."""
    m = _model()
    g = torch.Generator().manual_seed(1)
    obs = torch.rand(1, T + 1, OBS, generator=g)
    act = torch.rand(1, T, ACT, generator=g)
    h0 = torch.rand(1, H, generator=g) * 0.1

    with torch.no_grad():
        _, h_model, _ = m(obs, act, anchor_idx=torch.arange(3), state=h0,
                          noise_main=torch.zeros(1, T, H),
                          noise_roll=torch.zeros(1, 1, 3, H))
    h_manual = manual_rollout(m, obs[0, :T], act[0], h0[0])

    n = h_model.shape[1]
    d = (h_model[0, :n] - h_manual[:n]).abs().max().item()
    assert d < 1e-5, f'manual recurrence differs from the model by {d}'


def test_offline_rollout_runs_without_the_environment_and_stays_finite():
    m = _model()
    h0 = torch.rand(H) * 0.1
    acts = random_actions(30, np.random.default_rng(0))
    assert acts.shape == (30, 5)
    Ht = offline_rollout(m, h0, acts)
    assert Ht.shape == (30, H)
    assert torch.isfinite(Ht).all()
    assert (Ht >= 0).all(), 'ReLU output must be non-negative'


def test_random_actions_are_speed_hd_encoded():
    a = random_actions(200, np.random.default_rng(0)).numpy()
    assert set(np.unique(a[:, 0])) <= {0.0, 1.0}         # speed is binary
    assert np.allclose(a[:, 1:].sum(1), 1.0)             # heading is one-hot
    assert len(np.unique(np.argmax(a[:, 1:], 1))) > 1    # the heading actually turns


def test_path_stats_separates_a_smooth_path_from_a_shuffled_one():
    t = np.linspace(0, 4 * np.pi, 200)
    smooth = np.stack([9 + 5 * np.cos(t), 9 + 5 * np.sin(t)], 1)
    rng = np.random.default_rng(0)
    shuffled = smooth[rng.permutation(len(smooth))]
    s, sh = path_stats(smooth), path_stats(shuffled)
    assert s['continuity'] > 0.95
    assert sh['continuity'] < 0.3
    assert s['step'] < sh['step']


def test_path_stats_coverage_is_a_fraction():
    P = np.stack([np.arange(18) + 1.0, np.arange(18) + 1.0], 1)
    st = path_stats(P, arena=18)
    assert 0 < st['coverage'] <= 1
    assert st['coverage'] == pytest.approx(18 / 324)


def test_every_offline_readout_carries_a_wake_and_shuffle_baseline():
    """The bug this pins: `off_cov` was reported for four horizons with no `wake_cov` anywhere
    in the CSV, so 'coverage = 0.358' compared against nothing. An offline path of N steps
    cannot cover more cells than a wake path of N steps, so the bare number is not a result.
    Any readout added to READOUTS must bring its baselines with it."""
    stats = {'step': 1.0, 'continuity': 0.5, 'coverage': 0.25}
    row = summary_row({'seed': 0}, [stats], [stats], stats, stats)
    for short in READOUTS.values():
        for prefix in ('off', 'dom', 'wake', 'shuf'):
            assert f'{prefix}_{short}' in row, f'{prefix}_{short} missing from the CSV row'
    assert row['seed'] == 0


def test_summary_row_averages_offline_rollouts_but_not_the_baselines():
    a = {'step': 1.0, 'continuity': 0.0, 'coverage': 0.1}
    b = {'step': 3.0, 'continuity': 1.0, 'coverage': 0.3}
    wake = {'step': 9.0, 'continuity': 9.0, 'coverage': 9.0}
    row = summary_row({}, [a, b], [a, b], wake, wake)
    assert row['off_step'] == pytest.approx(2.0)      # mean over rollouts
    assert row['off_cov'] == pytest.approx(0.2)
    assert row['wake_step'] == pytest.approx(9.0)     # single reference path, not averaged
