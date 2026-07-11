"""The init variants must vary exactly one thing at a time.

Two ways this silently goes wrong:
  1. `baseline` drifts from what pRNN_th actually builds, so the whole sweep is measured
     against the wrong reference;
  2. `orth` is not scale-matched to `uniform`, so "structure" is really "gain" and the
     conclusion is about the wrong variable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from project5_symmetry.training.inits import (VARIANTS, apply_init, recurrent_weight,
                                              spectral_radius)
from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCellEager

H = 200


def _model(tau=2):
    torch.manual_seed(0)
    return pRNN_th(obs_size=12, act_size=5, k=1, hidden_size=H, cell=LayerNormRNNCellEager,
                   dropp=0.0, trunc=8, neuralTimescale=tau, predOffset=0, hidden_init_sigma=0.1)


@pytest.mark.parametrize('variant', tuple(VARIANTS))
def test_identity_component_is_exactly_one_minus_one_over_tau(variant):
    tau, gain, _ = VARIANTS[variant]
    W = recurrent_weight(H, variant, seed=0).numpy()
    off = W - np.diag(np.diag(W))
    # E[diag] = (1 - 1/tau) because the random part is zero-mean on the diagonal too
    assert np.mean(np.diag(W)) == pytest.approx(1 - 1 / tau, abs=0.02)
    if gain == 0.0:
        assert np.abs(off).max() == 0.0, 'zero_rec must have no random recurrence'


def test_baseline_matches_what_pRNN_th_actually_builds():
    """Statistically identical to the constructor's W, or the reference arm is wrong."""
    m = _model(tau=2)
    ref = m.W.detach().numpy()
    got = recurrent_weight(H, 'baseline', seed=0).numpy()
    a = (1.0 / H) ** 0.5
    for M in (ref, got):
        off = M[~np.eye(H, dtype=bool)]
        assert off.std() == pytest.approx(a / np.sqrt(3), rel=0.05)
        assert abs(off.mean()) < 0.01
    assert np.mean(np.diag(ref)) == pytest.approx(np.mean(np.diag(got)), abs=0.02)
    assert spectral_radius(ref) == pytest.approx(spectral_radius(got), rel=0.10)


def test_orthogonal_is_scale_matched_to_uniform():
    """`orth` must differ from `baseline` in STRUCTURE only: same Frobenius norm and same
    spectral radius for the random part."""
    I = np.eye(H) * 0.5
    Ru = recurrent_weight(H, 'baseline', seed=0).numpy() - I
    Ro = recurrent_weight(H, 'orth', seed=0).numpy() - I

    assert np.linalg.norm(Ru) == pytest.approx(np.linalg.norm(Ro), rel=0.05)
    assert spectral_radius(Ru) == pytest.approx(spectral_radius(Ro), rel=0.12)
    # and it really is orthogonal, up to the scale
    G = Ro @ Ro.T
    scale2 = np.mean(np.diag(G))
    assert np.allclose(G, scale2 * np.eye(H), atol=1e-5)


def test_gain_scales_only_the_random_part():
    I = np.eye(H) * 0.5
    base = recurrent_weight(H, 'baseline', seed=3).numpy() - I
    lo = recurrent_weight(H, 'gain_lo', seed=3).numpy() - I
    hi = recurrent_weight(H, 'gain_hi', seed=3).numpy() - I
    assert np.allclose(lo, 0.5 * base, atol=1e-6)
    assert np.allclose(hi, 2.0 * base, atol=1e-6)


def test_variants_are_deterministic_and_seed_dependent():
    a = recurrent_weight(H, 'baseline', seed=1)
    b = recurrent_weight(H, 'baseline', seed=1)
    c = recurrent_weight(H, 'baseline', seed=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_apply_init_preserves_the_weight_alias_and_overwrites_the_constructor():
    m = _model(tau=2)
    before = m.W.detach().clone()
    apply_init(m, 'tau8', seed=0)
    assert m.W is m.rnn.cell.weight_hh
    assert not torch.equal(m.W.detach(), before)
    # the constructor's 0.5*I must be REPLACED by 0.875*I, not added to it
    assert float(m.W.detach().diagonal().mean()) == pytest.approx(1 - 1 / 8, abs=0.02)


def test_tau1_has_no_identity_and_zero_rec_is_pure_leak():
    W1 = recurrent_weight(H, 'tau1', seed=0)
    assert float(W1.diagonal().mean()) == pytest.approx(0.0, abs=0.02)
    Wz = recurrent_weight(H, 'zero_rec', seed=0).numpy()
    assert np.allclose(Wz, 0.5 * np.eye(H), atol=1e-7)


@pytest.mark.parametrize('variant', tuple(VARIANTS))
def test_every_variant_trains_without_blowing_up(variant):
    """gain_hi has spectral radius 1.68 and tau8 has 1.46. LayerNorm should hold them, but a
    variant that produces NaN losses would silently poison the whole ensemble (vmap shares
    the graph), so check each one end to end."""
    import torch.nn.functional as F

    m = _model()
    apply_init(m, variant, seed=0)
    m.train()
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-3)
    obs, act = torch.rand(2, 9, 12), torch.rand(2, 8, 5)
    aidx = torch.arange(3)
    for _ in range(5):
        pred, _, target = m(obs, act, anchor_idx=aidx)
        loss = F.mse_loss(pred, target)
        assert torch.isfinite(loss), f'{variant}: non-finite loss'
        opt.zero_grad(); loss.backward()
        for n, p_ in m.named_parameters():
            if p_.grad is not None:
                assert torch.isfinite(p_.grad).all(), f'{variant}: non-finite grad in {n}'
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    assert torch.isfinite(m.W).all()


def test_uniform_variants_share_the_same_base_random_matrix_at_a_given_seed():
    """The sweep is PAIRED: within a seed, `gain_lo` must be exactly 0.5x the baseline's
    random part and `tau8` exactly the baseline's random part with a different diagonal.
    Drawing a fresh R per variant would make every comparison a between-subjects one."""
    I = np.eye(H)
    base = recurrent_weight(H, 'baseline', seed=7).numpy() - 0.5 * I
    assert np.allclose(recurrent_weight(H, 'gain_lo', seed=7).numpy() - 0.5 * I,
                       0.5 * base, atol=1e-6)
    assert np.allclose(recurrent_weight(H, 'tau8', seed=7).numpy() - (1 - 1 / 8) * I,
                       base, atol=1e-6)
    assert np.allclose(recurrent_weight(H, 'tau1', seed=7).numpy(), base, atol=1e-6)


def test_kaiming_reproduces_the_initialisation_the_authors_abandoned():
    """thetaRNN.py carries `#(kaiming he initialization - blows up :'( )`. Reproduce it, and
    pin the spectral radii that explain why: the leak term pushes Kaiming to rho ~ 1.94, while
    the uniform draw it was replaced by lands at rho ~ 1.08."""
    H = 500
    rho = lambda W: spectral_radius(np.asarray(W))
    kai = recurrent_weight(H, 'kaiming', seed=0).numpy()
    kai_noid = recurrent_weight(H, 'kaiming_noid', seed=0).numpy()
    base = recurrent_weight(H, 'baseline', seed=0).numpy()

    assert rho(kai_noid) == pytest.approx(1.44, abs=0.10)   # Kaiming alone
    assert rho(kai) == pytest.approx(1.94, abs=0.12)        # Kaiming + 0.5*I
    assert rho(base) == pytest.approx(1.08, abs=0.08)       # the shipped default
    assert rho(kai) > rho(base)

    # the random parts differ by exactly sqrt(6) in scale (std sqrt(2/H) vs sqrt(1/(3H)))
    off_k = (kai_noid)[~np.eye(H, dtype=bool)]
    off_u = (base - 0.5 * np.eye(H))[~np.eye(H, dtype=bool)]
    assert off_k.std() / off_u.std() == pytest.approx(np.sqrt(6), rel=0.05)


def test_kaiming_has_the_identity_term_the_noid_variant_lacks():
    H = 300
    assert float(recurrent_weight(H, 'kaiming', 0).diagonal().mean()) == pytest.approx(0.5, abs=0.02)
    assert float(recurrent_weight(H, 'kaiming_noid', 0).diagonal().mean()) == pytest.approx(0.0, abs=0.02)


def test_spectral_stats_flag_a_diverged_matrix_instead_of_crashing():
    from project5_symmetry.analysis.spectral_trajectory import stats
    good = stats(recurrent_weight(50, 'baseline', 0).numpy())
    assert good['finite'] == 1 and good['rho'] > 0
    bad = stats(np.full((50, 50), np.nan))
    assert bad['finite'] == 0 and np.isnan(bad['rho'])


def test_kaiming_is_stable_under_layernorm_at_production_geometry():
    """The authors' `blows up` comment is in RNNCell, the PLAIN cell. The paper's model uses
    LayerNormRNNCell, which renormalises the pre-activation every step. Kaiming at rho=1.94
    trains without diverging there -- so the comment does not transfer, and the claim must be
    tested rather than inherited."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = pRNN_th(obs_size=24, act_size=5, k=1, hidden_size=128, cell=LayerNormRNNCellEager,
                dropp=0.0, trunc=10, neuralTimescale=2, predOffset=0, hidden_init_sigma=0.1)
    apply_init(m, 'kaiming', seed=0)
    assert spectral_radius(m.W.detach().numpy()) > 1.7
    m.train()
    opt = torch.optim.RMSprop(m.parameters(), lr=2e-3, alpha=0.95, eps=1e-6)
    for _ in range(30):
        pred, _, tgt = m(torch.rand(2, 11, 24), torch.rand(2, 10, 5), anchor_idx=torch.arange(4))
        loss = F.mse_loss(pred, tgt)
        assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    assert torch.isfinite(m.W).all()
