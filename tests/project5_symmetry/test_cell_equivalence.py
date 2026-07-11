"""Regression tests for the pRNN training-path optimisation.

Pins the three properties the optimisation depends on:

  1. The eager nn.Module cell is bit-exact with the legacy TorchScript cell.
  2. torch.compile can actually trace the eager cell (it cannot trace the
     TorchScript one, which is why the old `compiled: true` runs got nothing).
  3. Checkpoints written while the cell was compiled carry `_orig_mod.` in their
     keys, and still load.
"""
from __future__ import annotations

import torch

from utils.Architectures import (load_prnn_state_dict, pRNN_th, prnn_state_dict,
                                 strip_compile_prefix)
from utils.thetaRNN import LayerNormRNNCellEager, LayerNormRNNCellScript

OBS, ACT, HID, K = 147, 5, 64, 3
B, T = 3, 12


def _model(cell_cls):
    torch.manual_seed(0)
    return pRNN_th(obs_size=OBS, act_size=ACT, k=K, hidden_size=HID, cell=cell_cls,
                   dropp=0.0, trunc=T, neuralTimescale=2, predOffset=0,
                   hidden_init_sigma=0.1)


def _inputs():
    torch.manual_seed(5)
    obs = torch.randn(B, T + 1, OBS)
    act = torch.randn(B, T, ACT)
    aidx = torch.sort(torch.randperm(T - K)[:4]).values
    return obs, act, aidx


def test_eager_cell_forward_is_bit_exact_with_script_cell():
    ref, new = _model(LayerNormRNNCellScript), _model(LayerNormRNNCellEager)
    new.load_state_dict(ref.state_dict())
    ref.eval(), new.eval()
    obs, act, aidx = _inputs()

    with torch.no_grad():
        torch.manual_seed(9)
        y1, h1, t1 = ref(obs, act, anchor_idx=aidx)
        torch.manual_seed(9)
        y2, h2, t2 = new(obs, act, anchor_idx=aidx)

    assert torch.equal(y1, y2)
    assert torch.equal(h1, h2)
    assert torch.equal(t1, t2)


def test_eager_cell_backward_is_bit_exact_with_script_cell():
    ref, new = _model(LayerNormRNNCellScript), _model(LayerNormRNNCellEager)
    new.load_state_dict(ref.state_dict())
    obs, act, aidx = _inputs()

    def grads(m):
        m.train()
        m.zero_grad(set_to_none=True)
        torch.manual_seed(9)
        y, _, tgt = m(obs, act, anchor_idx=aidx)
        ((y - tgt) ** 2).mean().backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    g_ref = grads(ref)
    g_new = grads(new)
    assert set(g_ref) == set(g_new)
    for name in g_ref:
        assert torch.equal(g_ref[name], g_new[name]), f'gradient mismatch on {name}'


def test_eager_cell_traces_without_graph_breaks():
    """The whole timestep loop must land in one Dynamo graph.

    The TorchScript cell is opaque to Dynamo (graph_count == 0), so torch.compile
    silently captures nothing through it.
    """
    import torch._dynamo as dynamo

    def unroll(cell):
        x = torch.randn(B, T, OBS + ACT)
        noise = torch.zeros(B, T, HID)
        hx = torch.zeros(B, HID)
        out = []
        for t in range(T):
            hx, _ = cell(x[:, t, :], noise[:, t, :], (hx, 0))
            out.append(hx)
        return torch.stack(out, 1)

    torch.manual_seed(0)
    eager = LayerNormRNNCellEager(OBS + ACT, HID).eval()
    script = LayerNormRNNCellScript(OBS + ACT, HID).eval()

    dynamo.reset()
    with torch.no_grad():
        expl_eager = dynamo.explain(lambda: unroll(eager))()
    dynamo.reset()
    with torch.no_grad():
        expl_script = dynamo.explain(lambda: unroll(script))()

    assert expl_eager.graph_break_count == 0
    assert expl_eager.graph_count == 1
    # The legacy cell captures nothing at all.
    assert expl_script.graph_count == 0


def test_strip_compile_prefix_round_trips_a_compiled_cell():
    m = _model(LayerNormRNNCellEager)
    clean_before = set(prnn_state_dict(m))

    m.rnn.cell = torch.compile(m.rnn.cell)
    raw = m.state_dict()
    assert any('_orig_mod.' in k for k in raw), 'compile should mangle the keys'
    assert set(prnn_state_dict(m)) == clean_before
    assert not any('_orig_mod.' in k for k in strip_compile_prefix(raw))


def test_load_prnn_state_dict_accepts_legacy_and_compiled_checkpoints():
    ref = _model(LayerNormRNNCellEager)
    clean = prnn_state_dict(ref)
    prefixed = {k.replace('rnn.cell.', 'rnn.cell._orig_mod.'): v for k, v in clean.items()}

    for payload in (clean,                       # bare state_dict
                    {'model': clean},            # current checkpoint layout
                    {'model_state_dict': clean}, # older layout
                    {'model': prefixed}):        # written while the cell was compiled
        target = _model(LayerNormRNNCellEager)
        out = load_prnn_state_dict(target, payload, strict=True)
        assert not out.missing_keys and not out.unexpected_keys


def test_anchor_bounds_check_is_off_by_default_but_still_available(monkeypatch):
    import utils.Architectures as arch

    m = _model(LayerNormRNNCellEager).eval()
    obs, act, _ = _inputs()
    bad = torch.tensor([0, 999])

    # Default: no .item() sync, so no ValueError; PyTorch's own indexing catches it.
    assert arch._VALIDATE_ANCHOR_IDX is False

    monkeypatch.setattr(arch, '_VALIDATE_ANCHOR_IDX', True)
    with torch.no_grad():
        try:
            m(obs, act, anchor_idx=bad)
        except ValueError as e:
            assert 'anchor_idx must be within' in str(e)
        else:
            raise AssertionError('validation should raise when enabled')
