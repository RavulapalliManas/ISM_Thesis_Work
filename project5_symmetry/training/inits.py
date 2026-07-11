"""Recurrent-weight initialisations, for the question: what does the init preconfigure?

`pRNN_th` does not start from a blank slate. `Architectures.py` builds

    W = U(-1/sqrt(H), 1/sqrt(H))  +  (1 - 1/tau) * I          tau = neuralTimescale

so the recurrent matrix carries a leaky-integrator prior on its diagonal -- 0.5*I at the
paper's tau = 2, which is fixed throughout and never varied. Three separable knobs:

    tau     the identity component, (1 - 1/tau):  0, 0.5, 0.75, 0.875
    gain    the scale of the random part
    struct  its structure: uniform iid, or orthogonal

The optimiser is unaffected: `_build_optimizer` sets lr = lambda * sqrt(1/H), a function
of hidden size alone, not of the init's magnitude. So the init can be varied freely
without perturbing the learning rates.

Matching the orthogonal variant
-------------------------------
An orthogonal Q has ||Q||_F^2 = H and every singular value 1. The uniform draw has entry
std a/sqrt(3) with a = 1/sqrt(H), hence ||R||_F^2 = H^2 a^2/3 = H/3 and spectral radius
~ sqrt(H) * a/sqrt(3) = 1/sqrt(3). Scaling Q by 1/sqrt(3) matches BOTH the Frobenius norm
and the spectral radius, so `orth` differs from `uniform` in structure only -- not in size.
Without that scaling the comparison would confound structure with gain.
"""
from __future__ import annotations

import numpy as np
import torch

# name -> (tau, gain, structure)
#
# `kaiming` and `kaiming_noid` reproduce the initialisation the original authors tried and
# abandoned. thetaRNN.py still carries their commented-out code and the note
# "(kaiming he initialization - blows up :'( )", with the uniform draw that replaced it
# labelled "goodbug". The spectral radii explain the sad face:
#
#   Kaiming alone                       rho = 1.442
#   Kaiming + the 0.5*I pRNN_th adds    rho = 1.942   <- what actually ran
#   uniform ("goodbug") alone           rho = 0.585
#   uniform + 0.5*I  (the default)      rho = 1.079   <- barely supercritical
#
# Kaiming's std is sqrt(2/H) against the uniform draw's sqrt(1/(3H)): a factor sqrt(6) = 2.45
# in spectral radius. The "goodbug" is not luck. It is the only draw that lands near rho = 1
# once the leak term is added.
VARIANTS: dict[str, tuple[float, float, str]] = {
    'tau1':     (1.0, 1.0, 'uniform'),    # no identity component at all
    'baseline': (2.0, 1.0, 'uniform'),    # the paper: 0.5 * I
    'tau4':     (4.0, 1.0, 'uniform'),    # 0.75 * I
    'tau8':     (8.0, 1.0, 'uniform'),    # 0.875 * I
    'gain_lo':  (2.0, 0.5, 'uniform'),
    'gain_hi':  (2.0, 2.0, 'uniform'),
    'orth':     (2.0, 1.0, 'orthogonal'), # matched spectral radius, different structure
    'zero_rec': (2.0, 0.0, 'uniform'),    # pure leak: W = 0.5 * I, no random recurrence
    # the abandoned initialisation, and it without the leak term that pushes it over
    'kaiming':      (2.0, 1.0, 'gaussian'),   # rho ~ 1.94  -- the one that "blows up"
    'kaiming_noid': (1.0, 1.0, 'gaussian'),   # rho ~ 1.44  -- isolates the leak term's role
}

ORTH_SCALE = 1.0 / np.sqrt(3.0)   # makes ||Q||_F and rho(Q) match the uniform draw


def recurrent_weight(hidden_size: int, variant: str, seed: int) -> torch.Tensor:
    """W = gain * R + (1 - 1/tau) * I, with R uniform or orthogonal."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown init variant {variant!r}; want one of {tuple(VARIANTS)}')
    tau, gain, struct = VARIANTS[variant]
    g = torch.Generator().manual_seed(int(seed))
    if struct == 'uniform':
        a = (1.0 / hidden_size) ** 0.5
        R = torch.rand(hidden_size, hidden_size, generator=g) * 2 * a - a
    elif struct == 'gaussian':
        # Kaiming He for ReLU: std = sqrt(2 / fan_in). Exactly the authors' commented-out draw.
        R = torch.randn(hidden_size, hidden_size, generator=g) * (2.0 / hidden_size) ** 0.5
    elif struct == 'orthogonal':
        R = torch.empty(hidden_size, hidden_size)
        torch.nn.init.orthogonal_(R, generator=g)
        R *= ORTH_SCALE
    else:
        raise ValueError(struct)
    return gain * R + (1.0 - 1.0 / tau) * torch.eye(hidden_size)


def apply_init(model, variant: str, seed: int) -> None:
    """Overwrite the recurrent matrix in place, preserving the W / rnn.cell.weight_hh alias.

    `pRNN_th.__init__` has already added its own (1 - 1/2) * I; we replace W wholesale
    rather than adding to it, so the identity component is exactly (1 - 1/tau).
    """
    H = model.W.shape[0]
    with torch.no_grad():
        model.W.copy_(recurrent_weight(H, variant, seed).to(model.W.device, model.W.dtype))
    assert model.W is model.rnn.cell.weight_hh, 'the W / weight_hh alias was broken'


def spectral_radius(W) -> float:
    return float(np.abs(np.linalg.eigvals(np.asarray(W, dtype=np.float64))).max())
