"""The ensemble must train at any rollout horizon, including k=0.

k=0 is the autoencoder control: the target is `obs[anchor]` itself, and the rollout
noise tensor has a zero-length horizon axis (S,B,0,A,H). That is exactly the kind of
degenerate shape that vmap or inductor can quietly mishandle, and it sits at the left
edge of the dose-response curve, so a silent failure there would look like a result.
"""
from __future__ import annotations

import pytest
import torch

from project5_symmetry.training import ensemble as ens
from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCellEager

T, OBS, ACT, H, B, A, S = 20, 12, 5, 16, 2, 4, 2


def _models(k, n=S):
    ms = []
    for s in range(n):
        torch.manual_seed(s)
        ms.append(pRNN_th(obs_size=OBS, act_size=ACT, k=k, hidden_size=H,
                          cell=LayerNormRNNCellEager, dropp=0.0, trunc=T,
                          neuralTimescale=2, predOffset=0, hidden_init_sigma=0.1))
    return ms


def _batch(k):
    g = torch.Generator().manual_seed(0)
    return (torch.rand(S, B, T + 1, OBS, generator=g),
            torch.rand(S, B, T, ACT, generator=g),
            torch.rand(S, B, H, generator=g) * 0.1,
            torch.zeros(S, B, T, H),
            torch.zeros(S, B, k, A, H),
            torch.arange(A))


@pytest.mark.parametrize('k', [0, 1, 3, 5])
def test_vmap_grads_are_finite_and_per_model_at_every_horizon(k):
    models = _models(k)
    obs, act, state, nm, nr, aidx = _batch(k)
    for i, m in enumerate(models):
        m.eval()
        ens.warm_buffers(m, obs[i], act[i], anchor_idx=aidx, state=state[i],
                         noise_main=nm[i], noise_roll=nr[i])
        m.train()

    params, buffers, base = ens.stack_models(models)
    base.train()
    gfn = ens.make_loss_and_grad_fn(base, buffers, randomness='different')
    grads, losses = gfn(ens.as_leaves(params), obs, act, state, nm, nr, aidx)

    assert losses.shape == (S,)
    assert torch.isfinite(losses).all(), f'non-finite loss at k={k}'
    for name in ens.TRAINABLE:
        g = grads[name]
        assert g.shape[0] == S
        assert torch.isfinite(g).all(), f'non-finite grad {name} at k={k}'
        assert g.abs().sum() > 0, f'zero grad {name} at k={k}'


def test_k0_target_is_the_anchor_observation():
    """Pins the meaning of the left edge of the horizon curve."""
    m = _models(0, n=1)[0].eval()
    obs, act, state, nm, nr, aidx = _batch(0)
    with torch.no_grad():
        _, _, target = m(obs[0], act[0], anchor_idx=aidx, state=state[0],
                         noise_main=nm[0], noise_roll=nr[0])
    assert torch.equal(target.reshape(-1), obs[0][:, aidx, :].reshape(-1))


def test_k0_and_k5_are_genuinely_different_objectives():
    obs, act, state, nm, nr, _ = _batch(5)
    aidx = torch.arange(A)
    with torch.no_grad():
        _, _, t5 = _models(5, n=1)[0].eval()(
            obs[0], act[0], anchor_idx=aidx, state=state[0], noise_main=nm[0], noise_roll=nr[0])
        _, _, t0 = _models(0, n=1)[0].eval()(
            obs[0], act[0], anchor_idx=aidx, state=state[0],
            noise_main=nm[0], noise_roll=torch.zeros(B, 0, A, H))
    assert t5.shape[1] == 6 and t0.shape[1] == 1        # horizon axis is k+1
