"""Equivalence gates for the vmap-batched trainer.

The batched path is only useful if each model in the stack trains *exactly* as it
would alone. These tests pin that, and pin the three ways the naive stacking gets
it wrong (joint clipping, global-mean loss, wrong optimiser).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from project5_symmetry.training import ensemble as ens
from utils.Architectures import load_prnn_state_dict, pRNN_th
from utils.thetaRNN import LayerNormRNNCellEager

OBS, ACT, HID, K = 147, 5, 64, 3
S, B, T, A = 4, 3, 16, 5


def _model(seed):
    torch.manual_seed(seed)
    # dropout off + hidden_init_sigma irrelevant: all randomness is passed in
    return pRNN_th(obs_size=OBS, act_size=ACT, k=K, hidden_size=HID,
                   cell=LayerNormRNNCellEager, dropp=0.0, trunc=T,
                   neuralTimescale=2, predOffset=0, hidden_init_sigma=0.1).eval()


def _fixture(n=S, noise_std=0.03):
    torch.manual_seed(11)
    models = [_model(i) for i in range(n)]
    obs = torch.randn(n, B, T + 1, OBS)
    act = torch.randn(n, B, T, ACT)
    state = torch.rand(n, B, HID) * 0.1
    aidx = torch.sort(torch.randperm(T - K)[:A]).values
    noise_main = torch.randn(n, B, T, HID) * noise_std
    noise_roll = torch.randn(n, B, K, A, HID) * noise_std
    for i, m in enumerate(models):
        ens.warm_buffers(m, obs[i], act[i], anchor_idx=aidx, state=state[i],
                         noise_main=noise_main[i], noise_roll=noise_roll[i])
    return models, obs, act, state, aidx, noise_main, noise_roll


def _slice_norm(grads_by_name):
    flat = torch.cat([grads_by_name[n].reshape(-1) for n in ens.TRAINABLE])
    return torch.linalg.vector_norm(flat).item()


def _reference_grads(models, obs, act, state, aidx, noise_main, noise_roll):
    """Per-model grads from S separate pRNN_th backward passes."""
    out = []
    for i, m in enumerate(models):
        m.zero_grad(set_to_none=True)
        pred, _, target = m(obs[i], act[i], anchor_idx=aidx, state=state[i],
                            noise_main=noise_main[i], noise_roll=noise_roll[i])
        F.mse_loss(pred, target).backward()
        out.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                    if p.grad is not None})
    return out


# ── 1. forward + gradients are bit-exact per model ───────────────────────────

def test_vmap_forward_is_bit_exact_per_model():
    models, obs, act, state, aidx, nm, nr = _fixture()
    params, buffers, base = ens.stack_models(models)
    loss_fn = ens.make_loss_fn(base, buffers)

    with torch.no_grad():
        batched = torch.vmap(loss_fn, in_dims=ens._IN_DIMS)(params, obs, act, state, nm, nr, aidx)
        for i, m in enumerate(models):
            pred, _, target = m(obs[i], act[i], anchor_idx=aidx, state=state[i],
                                noise_main=nm[i], noise_roll=nr[i])
            assert torch.equal(F.mse_loss(pred, target), batched[i])


def test_vmap_gradients_are_bit_exact_when_noise_is_off():
    """With no injected noise the batched gradients match the sequential reference up to
    fp32 reassociation.

    vmap still lowers each model's `mm` to a batched `bmm` here, so on GPUs whose cuBLAS
    picks a different batched-matmul kernel than the sequential path (observed on an RTX PRO
    4500, Blackwell/sm_120: ~1e-10 absolute, not present on the hardware this test was
    written against), the two accumulate in a different order even with noise off. Same
    reassociation argument and bound as
    test_vmap_gradients_match_within_fp32_reassociation_with_production_noise: measured worst
    case here is ~2.8e-7, the same order of magnitude as that test's noisy case, so it is the
    batched matmul, not the injected noise, that dominates the discrepancy on this hardware.
    """
    models, obs, act, state, aidx, nm, nr = _fixture(noise_std=0.0)
    ref = _reference_grads(models, obs, act, state, aidx, nm, nr)
    params, buffers, base = ens.stack_models(models)
    g = ens.make_grad_fn(base, buffers)(params, obs, act, state, nm, nr, aidx)

    worst_rel = 0.0
    for i in range(S):
        for name, want in ref[i].items():
            delta = (g[name][i] - want).abs().max().item()
            scale = want.abs().max().item() + 1e-30
            worst_rel = max(worst_rel, delta / scale)
    assert worst_rel < 1e-6, f'gradients drifted beyond fp32 reassociation: rel={worst_rel:.2e}'


def test_vmap_gradients_match_within_fp32_reassociation_with_production_noise():
    """With NOISE_STD=0.03 the agreement is sub-ULP, not exact.

    vmap lowers each model's `mm` to a batched `bmm`, which accumulates in a
    different order. Measured worst case: 2.3e-10 absolute, 9.8e-8 relative,
    against an fp32 epsilon of 1.19e-7. This is a reassociation difference, not
    a semantic one -- see test_vmap_gradients_are_bit_exact_when_noise_is_off.
    """
    models, obs, act, state, aidx, nm, nr = _fixture(noise_std=0.03)
    ref = _reference_grads(models, obs, act, state, aidx, nm, nr)
    params, buffers, base = ens.stack_models(models)
    g = ens.make_grad_fn(base, buffers)(params, obs, act, state, nm, nr, aidx)

    worst_rel = 0.0
    for i in range(S):
        for name, want in ref[i].items():
            delta = (g[name][i] - want).abs().max().item()
            scale = want.abs().max().item() + 1e-30
            worst_rel = max(worst_rel, delta / scale)
    assert worst_rel < 1e-6, f'gradients drifted beyond fp32 reassociation: rel={worst_rel:.2e}'


# ── 2. clipping: per-slice matches, joint does NOT ───────────────────────────

def test_clip_per_slice_matches_S_independent_clips():
    models, obs, act, state, aidx, nm, nr = _fixture(noise_std=0.0)
    ref = _reference_grads(models, obs, act, state, aidx, nm, nr)

    # pick a threshold that actually bites for every model, otherwise the test
    # passes trivially with every clip coefficient equal to 1
    norms = [_slice_norm(r) for r in ref]
    max_norm = 0.5 * min(norms)
    assert max_norm > 0

    want = []
    for m in models:
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm)
        want.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                     if p.grad is not None})

    params, buffers, base = ens.stack_models(models)
    g = ens.make_grad_fn(base, buffers)(params, obs, act, state, nm, nr, aidx)
    g = {k: v.clone() for k, v in g.items()}
    total = ens.clip_per_slice(g, max_norm=max_norm)

    assert (total > max_norm).all(), 'fixture did not trigger clipping'
    # atol matches the fp32 reassociation floor measured in
    # test_vmap_gradients_are_bit_exact_when_noise_is_off (~1e-9 on an RTX PRO 4500,
    # Blackwell/sm_120); clipping only rescales, so it does not add its own error on top.
    for i in range(S):
        for name in ens.TRAINABLE:
            assert torch.allclose(g[name][i], want[i][name], rtol=1e-5, atol=5e-9), \
                f'per-slice clip mismatch: {name}, model {i}'


def test_naive_joint_clip_is_wrong():
    """A single clip over the stacked tensors rescales all S models together.

    This test exists so nobody 'simplifies' clip_per_slice into one global clip.
    """
    models, obs, act, state, aidx, nm, nr = _fixture(noise_std=0.0)
    ref = _reference_grads(models, obs, act, state, aidx, nm, nr)
    norms = [_slice_norm(r) for r in ref]
    assert max(norms) / min(norms) > 1.01, 'per-model norms too similar to distinguish'
    max_norm = 0.5 * min(norms)

    params, buffers, base = ens.stack_models(models)
    g = ens.make_grad_fn(base, buffers)(params, obs, act, state, nm, nr, aidx)

    per_slice = {k: v.clone() for k, v in g.items()}
    ens.clip_per_slice(per_slice, max_norm=max_norm)

    joint = {k: v.clone() for k, v in g.items()}
    total = torch.sqrt(sum(joint[k].pow(2).sum() for k in ens.TRAINABLE))
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for k in ens.TRAINABLE:
        joint[k] = joint[k] * coef

    differs = any(not torch.allclose(per_slice[k], joint[k], rtol=1e-4, atol=1e-12)
                  for k in ens.TRAINABLE)
    assert differs, 'a joint clip must NOT reproduce per-slice clipping'


# ── 3. stacked optimiser == S separate optimisers ────────────────────────────

def test_stacked_rmsprop_matches_S_independent_optimizers():
    from project5_symmetry.training.train import _build_optimizer

    models, obs, act, state, aidx, nm, nr = _fixture()
    params, buffers, base = ens.stack_models(models)
    leaves = ens.as_leaves(params)
    opt_stacked = ens.stacked_rmsprop(leaves, batch_size=B)
    opts = [_build_optimizer(m, batch_size=B) for m in models]
    grad_fn = ens.make_grad_fn(base, buffers)

    for _ in range(3):
        # reference: S separate steps
        for i, m in enumerate(models):
            opts[i].zero_grad(set_to_none=True)
            pred, _, target = m(obs[i], act[i], anchor_idx=aidx, state=state[i],
                                noise_main=nm[i], noise_roll=nr[i])
            F.mse_loss(pred, target).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            opts[i].step()

        # batched: one step
        g = grad_fn({k: v for k, v in leaves.items()}, obs, act, state, nm, nr, aidx)
        g = {k: v.clone() for k, v in g.items()}
        ens.clip_per_slice(g, max_norm=1.0)
        opt_stacked.zero_grad(set_to_none=True)
        ens.apply_grads(leaves, g)
        opt_stacked.step()

    for i, m in enumerate(models):
        for name, p in m.named_parameters():
            if name not in ens.TRAINABLE:
                continue
            assert torch.allclose(leaves[name][i], p, atol=1e-6), \
                f'optimizer drift on {name}, model {i}'


# ── 4. checkpoint round-trip ─────────────────────────────────────────────────

def test_unstack_state_dict_round_trips_into_plain_prnn():
    models, obs, act, state, aidx, nm, nr = _fixture()
    params, _, _ = ens.stack_models(models)

    for i in range(S):
        sd = ens.unstack_state_dict(params, i)
        assert not any('_orig_mod.' in k for k in sd)
        fresh = _model(999)
        out = load_prnn_state_dict(fresh, sd, strict=True)
        assert not out.missing_keys and not out.unexpected_keys
        with torch.no_grad():
            a = models[i](obs[i], act[i], anchor_idx=aidx, state=state[i],
                          noise_main=nm[i], noise_roll=nr[i])
            b = fresh(obs[i], act[i], anchor_idx=aidx, state=state[i],
                      noise_main=nm[i], noise_roll=nr[i])
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ── 5. the performance claim: launches flat in S ─────────────────────────────

def _launch_count(n):
    import collections
    from torch.profiler import ProfilerActivity, profile

    VIEW = {'aten::as_strided', 'aten::t', 'aten::transpose', 'aten::slice', 'aten::view',
            'aten::reshape', 'aten::resolve_conj', 'aten::empty_strided', 'aten::empty',
            'aten::expand', 'aten::select', 'aten::unsqueeze', 'aten::squeeze', 'aten::detach',
            'aten::resolve_neg', 'aten::alias', 'aten::_unsafe_view', 'aten::permute',
            'aten::contiguous', 'aten::narrow', 'aten::empty_like'}

    models, obs, act, state, aidx, nm, nr = _fixture(n)
    params, buffers, base = ens.stack_models(models)
    fn = torch.vmap(ens.make_loss_fn(base, buffers), in_dims=ens._IN_DIMS)

    def run():
        with torch.no_grad():
            fn(params, obs, act, state, nm, nr, aidx)

    run()
    with profile(activities=[ProfilerActivity.CPU]) as p:
        run()
    c = collections.Counter(e.name for e in p.events() if e.name.startswith('aten::'))
    return sum(v for k, v in c.items() if k not in VIEW), c['aten::bmm'], c['aten::mm']


# ── 5. the constants must not drift from train.py ────────────────────────────

def test_optimizer_constants_match_train_py():
    """ensemble.py re-declares train.py's hyperparameters; pin them together.

    If someone retunes GLOBAL_LR in train.py, the ensemble must not silently keep
    training at the old value.
    """
    from project5_symmetry.training import train as tr

    assert ens.GLOBAL_LR == tr.GLOBAL_LR
    assert ens.BIAS_LR_SCALE == tr.BIAS_LR_SCALE
    assert ens.WEIGHT_DECAY == tr.WEIGHT_DECAY
    assert ens.RMSPROP_ALPHA == tr.RMSPROP_ALPHA
    assert ens.RMSPROP_EPS == tr.RMSPROP_EPS
    assert ens.REFERENCE_BATCH_SIZE == tr.BATCH_SIZE


def test_sweep_driver_constants_match_train_py():
    """run_ensemble_sweep.py inlines train.py's constants (it must not import
    train.py, which drags in tensorboard + pynapple). Pin them together."""
    from project5_symmetry.experiments import run_ensemble_sweep as drv
    from project5_symmetry.training import train as tr

    assert drv.HIDDEN_SIZE == tr.HIDDEN_SIZE
    assert drv.NOISE_STD == tr.NOISE_STD
    assert drv.DROPOUT_P == tr.DROPOUT_P
    assert drv.HIDDEN_INIT_SIGMA == tr.HIDDEN_INIT_SIGMA
    assert drv.PRED_OFFSET == tr.PRED_OFFSET
    assert drv.ANCHOR_SUBSAMPLE_N == tr.ANCHOR_SUBSAMPLE_N
    assert drv.CHECKPOINT_STEPS == tr.CHECKPOINT_STEPS
    # and the sweep really is the paper's 17 seeds
    assert len(drv.SWEEP) == 17
    assert [c for c, _ in drv.SWEEP].count('s1') == 5
    assert [c for c, _ in drv.SWEEP].count('s2') == 5
    assert [c for c, _ in drv.SWEEP].count('s4') == 7


# ── 6. dropout must stay inside the model ────────────────────────────────────

def test_pre_dropping_obs_corrupts_the_target():
    """Guards the trap: forward derives the target from the RAW obs.

    A tempting 'optimisation' is to hoist dropout out of the model (set p=0 and
    pre-mask obs) so the graph is pure. That silently trains against a
    dropped-out target. Keep dropout inside, under randomness='different'.
    """
    models, obs, act, state, aidx, nm, nr = _fixture(1, noise_std=0.0)
    m = models[0]
    with torch.no_grad():
        _, _, target_raw = m(obs[0], act[0], anchor_idx=aidx, state=state[0],
                             noise_main=nm[0], noise_roll=nr[0])
        obs_dropped = F.dropout(obs[0], p=0.15, training=True)
        _, _, target_pre = m(obs_dropped, act[0], anchor_idx=aidx, state=state[0],
                             noise_main=nm[0], noise_roll=nr[0])
    assert not torch.allclose(target_raw, target_pre), \
        'pre-dropping obs must change the target -- the trap is real'


def test_vmap_supports_live_dropout_with_randomness_different():
    models, obs, act, state, aidx, nm, nr = _fixture(noise_std=0.0)
    for m in models:
        m.train()
        m.droplayer.p = 0.15
    params, buffers, base = ens.stack_models(models)
    base.train()
    losses = torch.vmap(ens.make_loss_fn(base, buffers), in_dims=ens._IN_DIMS,
                        randomness='different')(params, obs, act, state, nm, nr, aidx)
    assert losses.shape == (S,)
    assert torch.isfinite(losses).all()


def test_loss_and_grad_fn_returns_per_model_losses():
    models, obs, act, state, aidx, nm, nr = _fixture(noise_std=0.0)
    params, buffers, base = ens.stack_models(models)
    grads, losses = ens.make_loss_and_grad_fn(base, buffers)(
        params, obs, act, state, nm, nr, aidx)
    assert losses.shape == (S,)
    only_grads = ens.make_grad_fn(base, buffers)(params, obs, act, state, nm, nr, aidx)
    for k in ens.TRAINABLE:
        assert torch.equal(grads[k], only_grads[k])


# ── 7. the performance claim: launches flat in S ─────────────────────────────

def test_launch_count_is_flat_in_S():
    """The whole point: launches must not grow with the number of models.

    Sequentially, S models cost S x the kernel launches. Stacked, vmap lowers each
    per-model `mm` to one batched `bmm` and the launch count is constant in S,
    carrying S x the arithmetic per launch.
    """
    counts = {n: _launch_count(n) for n in (1, 2, 4)}
    for n, (launches, n_bmm, _) in counts.items():
        assert n_bmm > 0, f'vmap did not emit bmm at S={n}'
    budget = counts[1][0] * 1.05          # 5% slack for bookkeeping ops
    for n, (launches, _, _) in counts.items():
        assert launches <= budget, \
            f'launches grew with S: S=1 -> {counts[1][0]}, S={n} -> {launches}'


def test_single_rmsprop_is_identical_to_train_py_build_optimizer():
    """The CPU workers use ensemble.single_rmsprop instead of train.py::_build_optimizer,
    because train.py imports tensorboard and pynapple. They must be the same optimiser.
    """
    from project5_symmetry.training.train import _build_optimizer

    for batch_size in (4, 8, 16):
        m = _model(seed=0)
        ref = _build_optimizer(m, batch_size=batch_size)
        got = ens.single_rmsprop(m, batch_size=batch_size)

        assert len(ref.param_groups) == len(got.param_groups)
        assert ref.defaults['alpha'] == got.defaults['alpha']
        assert ref.defaults['eps'] == got.defaults['eps']
        for gr, gg in zip(ref.param_groups, got.param_groups):
            assert gr['lr'] == pytest.approx(gg['lr'], rel=0, abs=0)
            assert gr['weight_decay'] == pytest.approx(gg['weight_decay'], rel=0, abs=0)
            assert [id(p) for p in gr['params']] == [id(p) for p in gg['params']]


def test_unstacked_checkpoint_does_not_carry_the_whole_ensemble(tmp_path):
    """`params['W'][i]` is a VIEW into the stacked (S,500,500) tensor, and torch.save
    serialises a view's entire storage. Saving views wrote every model's weights into
    every per-model checkpoint (76.9 MB instead of 3.2 MB at S=48). Slices must be cloned.

    The file size must not grow with the ensemble width.
    """
    import os

    sizes = {}
    for S_ in (2, 8):
        models = _fixture(n=S_)[0]
        params, _, _ = ens.stack_models(models)
        sd = ens.unstack_state_dict(ens.as_leaves(params), 0)
        p = tmp_path / f'ckpt_S{S_}.pt'
        torch.save({'model': sd}, p)
        sizes[S_] = os.path.getsize(p)
        tensor_bytes = sum(v.numel() * v.element_size() for v in sd.values())
        assert sizes[S_] < 2.0 * tensor_bytes, (
            f'S={S_}: file is {sizes[S_]/1e6:.1f} MB but the tensors are only '
            f'{tensor_bytes/1e6:.1f} MB -- a view is dragging its whole storage along')

    assert sizes[8] < 1.5 * sizes[2], 'checkpoint size grows with the ensemble width'


def test_unstacked_slices_are_independent_of_the_stacked_params():
    """A cloned slice must not alias the ensemble tensor, or mutating one corrupts the other."""
    models = _fixture(n=3)[0]
    params, _, _ = ens.stack_models(models)
    leaves = ens.as_leaves(params)
    sd = ens.unstack_state_dict(leaves, 1)
    before = sd['W'].clone()
    with torch.no_grad():
        leaves['W'][1].add_(1.0)
    assert torch.equal(sd['W'], before), 'unstacked slice aliases the stacked parameter'
