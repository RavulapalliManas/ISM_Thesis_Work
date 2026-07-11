"""Train S independent pRNN_th models as one batched GPU job.

Why
---
A single training step issues ~15,227 kernel launches, only 1,042 of which are
GEMMs (5.2 per timestep). At 63.4 ms/step on an H100 that is 4.2 us per launch —
the CUDA launch-latency floor — and 0.16% of the card. The model is tiny; the
parallelism lives *across models*, not on the tensor.

T=200 is irreducibly sequential (nonlinear RNN, no parallel scan), so we cannot
shorten the chain. Instead we make each link carry S times the work:
`torch.func.stack_module_state` + `vmap(functional_call)` lowers every per-model
`mm` to one batched `bmm`, and the launch count becomes *independent of S*
(measured: 2,181 launches at S = 1, 2, 4, 8, versus 1,635 * S for the loop).

`pRNN_th` is reused unchanged. This module only stacks it.

Four preconditions, learned the hard way
----------------------------------------
1. The hidden state must be pre-drawn and passed via `forward(state=...)`.
   Otherwise `torch.empty(B,H).uniform_()` inside the forward trips vmap's
   "called random operation while in randomness error mode".
2. `theta_idx` / `act_theta_idx` are identical across models. Keep them
   UNbatched (closure), or vmap will batch the Toeplitz gather indices.
3. The base module must be real, not `.to('meta')`. pRNN_th aliases its
   parameters (`model.W_in is model.rnn.cell.weight_ih`) and `.to('meta')`
   breaks the tie; `functional_call` then silently leaves the cell's weights
   untouched.
4. Dropout stays INSIDE the model, under `randomness='different'`. It cannot be
   hoisted out by pre-masking `obs`, because `forward` derives the prediction
   *target* from the raw observation and only then drops the *input*. Feeding a
   pre-dropped `obs` corrupts the target (measured: max|delta| = 2.8).

Equivalence guarantees (see tests/project5_symmetry/test_ensemble.py)
--------------------------------------------------------------------
Given identical pre-drawn randomness, the batched path reproduces S separate
`pRNN_th` runs:

* the **forward is bit-exact**, always;
* **gradients are bit-exact when the injected noise is zero**, and agree to
  within fp32 reassociation once NOISE_STD=0.03 is on (worst case measured:
  2.3e-10 absolute, 9.8e-8 relative, against an fp32 epsilon of 1.19e-7). The
  difference is `bmm` accumulating in a different order than `mm`, not a
  semantic change.

Preserving even that much requires three things the naive stacking gets wrong:

* gradient clipping is PER MODEL in `train.py` — a joint clip over the stacked
  tensor computes one norm across all S and rescales differently;
* the loss is a SUM of per-model means, not one global mean (a global mean
  divides every gradient by an extra factor S);
* the optimiser is RMSprop with four per-group learning rates and bias-only
  weight decay, not plain Adam.
"""
from __future__ import annotations

import copy
import math
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, stack_module_state, vmap

from utils.Architectures import pRNN_th

# Mirrors project5_symmetry/training/train.py. Imported rather than duplicated
# where possible; these are the ones train.py exposes only as module constants.
GLOBAL_LR = 2e-3
BIAS_LR_SCALE = 0.1
WEIGHT_DECAY = 3e-3
RMSPROP_ALPHA = 0.95
RMSPROP_EPS = 1e-6
REFERENCE_BATCH_SIZE = 8

# Trainable parameters, in the order `_build_optimizer` groups them.
TRAINABLE = ('W', 'W_in', 'W_out', 'bias')


# ── stacking ──────────────────────────────────────────────────────────────────

def warm_buffers(model: pRNN_th, obs: torch.Tensor, act: torch.Tensor, **kw) -> None:
    """Populate the lazily-built Toeplitz buffers.

    `theta_idx` / `act_theta_idx` are registered as None and materialised on the
    first forward (Architectures.py). They must exist before `named_buffers()`
    can hand them to `functional_call`.
    """
    with torch.no_grad():
        model(obs, act, **kw)


def stack_models(models: Sequence[pRNN_th]) -> tuple[dict, dict, pRNN_th]:
    """Return (stacked params, shared buffers, base module).

    Buffers are returned UNbatched: they are identical across models and must
    not acquire a vmap batch dimension.
    """
    params, _ = stack_module_state(models)
    buffers = {name: buf for name, buf in models[0].named_buffers()}
    if not buffers:
        raise RuntimeError(
            'Toeplitz buffers are empty; call warm_buffers() on each model first.'
        )
    base = copy.deepcopy(models[0])   # real module: .to("meta") breaks tied params
    return params, buffers, base


def unstack_state_dict(params: dict, index: int) -> dict:
    """Extract model `index` as a state_dict loadable by a plain pRNN_th.

    pRNN_th aliases W_in/W/W_out/bias onto rnn.cell.* and outlayer.0.weight, and
    its state_dict lists both names. vmap keeps the original orientation, so no
    transposes are needed.

    Every slice is CLONED. `params['W'][index]` is a view into the stacked (S,500,500)
    tensor, and torch.save serialises a view's entire underlying storage -- so saving a
    view writes all S models' weights into every per-model checkpoint. That turned 3.2 MB
    of tensors into a 76.9 MB file at S=48, and 45 GB of checkpoints where 2 GB was due.
    """
    w_in = params['W_in'][index].detach().clone()
    w_hh = params['W'][index].detach().clone()
    w_out = params['W_out'][index].detach().clone()
    bias = params['bias'][index].detach().clone()
    scale = params['rnn.cell.scale'][index].detach().clone()
    return {
        'W_in': w_in, 'W': w_hh, 'W_out': w_out, 'bias': bias,
        'rnn.cell.weight_ih': w_in,
        'rnn.cell.weight_hh': w_hh,
        'rnn.cell.bias': bias,
        'rnn.cell.scale': scale,
        'outlayer.0.weight': w_out,
    }


# ── the batched step ──────────────────────────────────────────────────────────

# `anchor_idx` is a trailing ARGUMENT rather than a closure capture: it is
# resampled every step, and an argument lets the driver build (and compile) the
# step function exactly once. It is shared across models within a step, matching
# train_parallel_seeds, hence in_dims=None.
_IN_DIMS = (0, 0, 0, 0, 0, 0, None)


def make_loss_fn(base: pRNN_th, buffers: dict) -> Callable:
    """Per-model scalar loss."""

    def loss_fn(params, obs, act, state, noise_main, noise_roll, anchor_idx):
        pred, _, target = functional_call(
            base, (params, buffers), (obs, act),
            {'anchor_idx': anchor_idx, 'state': state,
             'noise_main': noise_main, 'noise_roll': noise_roll},
        )
        return F.mse_loss(pred, target)

    return loss_fn


def make_grad_fn(base: pRNN_th, buffers: dict, randomness: str = 'different') -> Callable:
    """vmap(grad(per-model loss)) -> stacked per-model gradients.

    Each model's loss is differentiated independently, so the result equals
    summing per-model means and calling one backward — with no 1/S shrinkage.

    `randomness='different'` gives each model its own dropout mask, which is what
    training S independent networks requires. Do NOT try to hoist dropout out by
    pre-masking `obs`: `pRNN_th.forward` builds the prediction *target* from the
    raw observation and only then drops the *input* (Architectures.py), so a
    pre-dropped `obs` silently corrupts the target.
    """
    loss_fn = make_loss_fn(base, buffers)
    return vmap(grad(loss_fn), in_dims=_IN_DIMS, randomness=randomness)


def make_loss_and_grad_fn(base: pRNN_th, buffers: dict,
                          randomness: str = 'different') -> Callable:
    """As make_grad_fn, but returns (grads, per-model loss of shape (S,))."""
    from torch.func import grad_and_value

    loss_fn = make_loss_fn(base, buffers)
    return vmap(grad_and_value(loss_fn), in_dims=_IN_DIMS, randomness=randomness)


# ── gradient clipping, per model slice ───────────────────────────────────────

def clip_per_slice(grads: dict, max_norm: float = 1.0,
                   keys: Iterable[str] = TRAINABLE) -> torch.Tensor:
    """Clip each model's gradients independently, in place.

    Reproduces `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
    applied separately to each of the S models.

    A single clip over the stacked tensors would instead compute ONE norm across
    all S models and rescale them together — a different, wrong operation. This
    is the easiest way to silently corrupt a batched run.

    Returns the pre-clip total norm per model, shape (S,).
    """
    keys = list(keys)
    S = grads[keys[0]].shape[0]
    sq = torch.zeros(S, device=grads[keys[0]].device, dtype=grads[keys[0]].dtype)
    for k in keys:
        sq = sq + grads[k].reshape(S, -1).pow(2).sum(dim=1)
    total = sq.sqrt()
    # torch's clip_grad_norm_: coef = max_norm / (total + 1e-6), clamped to <= 1
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for k in keys:
        g = grads[k]
        grads[k] = g * coef.view(-1, *([1] * (g.dim() - 1)))
    return total


# ── optimiser ────────────────────────────────────────────────────────────────

def single_rmsprop(model: pRNN_th, batch_size: int) -> torch.optim.Optimizer:
    """`train.py::_build_optimizer` for one un-stacked model, without importing
    train.py (which drags in tensorboard and pynapple).

    The CPU workers train one model per process: vmap buys nothing on CPU, where
    there are no kernel launches to amortise, and measurably loses (232 vs 138 ms
    per 4-thread slot). tests/project5_symmetry/test_ensemble.py asserts this is
    group-for-group identical to _build_optimizer.
    """
    k_io = 1.0 / model.rnn.cell.weight_ih.shape[1]
    k_h = 1.0 / model.rnn.cell.weight_hh.shape[0]
    lr_scale = math.sqrt(REFERENCE_BATCH_SIZE / batch_size)
    lr_h = GLOBAL_LR * k_h ** 0.5 * lr_scale
    lr_io = GLOBAL_LR * k_io ** 0.5 * lr_scale
    lr_bias = GLOBAL_LR * BIAS_LR_SCALE * lr_scale
    groups = [
        {'params': [model.W], 'lr': lr_h, 'weight_decay': 0.0},
        {'params': [model.W_in], 'lr': lr_io, 'weight_decay': 0.0},
        {'params': [model.W_out], 'lr': lr_h, 'weight_decay': 0.0},
        {'params': [model.bias], 'lr': lr_bias, 'weight_decay': WEIGHT_DECAY * lr_bias},
    ]
    if getattr(getattr(model.rnn.cell, 'scale', None), 'requires_grad', False):
        groups.append({'params': [model.rnn.cell.scale], 'lr': lr_bias, 'weight_decay': 0.0})
    return torch.optim.RMSprop(groups, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)


def stacked_rmsprop(params: dict, batch_size: int) -> torch.optim.Optimizer:
    """One RMSprop over the stacked tensors, with train.py's four param groups.

    RMSprop is elementwise, so a stacked optimiser equals S separate ones as long
    as the per-group learning rates and the bias-only weight decay are preserved.
    Mirrors `train.py::_build_optimizer`.
    """
    k_io = 1.0 / params['W_in'].shape[2]     # (S, 500, 152) -> 152
    k_h = 1.0 / params['W'].shape[1]         # (S, 500, 500) -> 500
    lr_scale = math.sqrt(REFERENCE_BATCH_SIZE / batch_size)
    lr_h = GLOBAL_LR * k_h ** 0.5 * lr_scale
    lr_io = GLOBAL_LR * k_io ** 0.5 * lr_scale
    lr_bias = GLOBAL_LR * BIAS_LR_SCALE * lr_scale

    groups = [
        {'params': [params['W']],     'lr': lr_h,    'weight_decay': 0.0},
        {'params': [params['W_in']],  'lr': lr_io,   'weight_decay': 0.0},
        {'params': [params['W_out']], 'lr': lr_h,    'weight_decay': 0.0},
        {'params': [params['bias']],  'lr': lr_bias, 'weight_decay': WEIGHT_DECAY * lr_bias},
    ]
    return torch.optim.RMSprop(groups, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)


def as_leaves(params: dict) -> dict:
    """Make the stacked tensors optimisable leaves.

    `rnn.cell.scale` is requires_grad=False in pRNN_th and is excluded from the
    optimiser; vmap(grad(...)) still returns a gradient for it, which we ignore.
    """
    out = {}
    for name, tensor in params.items():
        leaf = tensor.detach()
        leaf.requires_grad_(name in TRAINABLE)
        out[name] = leaf
    return out


def apply_grads(params: dict, grads: dict, keys: Iterable[str] = TRAINABLE) -> None:
    for k in keys:
        params[k].grad = grads[k]
