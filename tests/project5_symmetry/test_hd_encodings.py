"""The HD encodings must dissociate information from symmetry.

`axis` and `parity` carry the SAME one bit. Only `axis` is C2-invariant. If that
ever stops being true, the central experiment is confounded.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from project5_symmetry.environments.hd_encodings import (MODES, apply_hd, bits,
                                                         hd_matrix, is_invariant)


def _onehot(h):
    return torch.eye(4)[h]


def test_axis_and_parity_carry_identical_information():
    assert bits('full') == pytest.approx(2.0)
    assert bits('axis') == pytest.approx(1.0)
    assert bits('parity') == pytest.approx(1.0)
    assert bits('const') == pytest.approx(0.0)


def test_only_axis_and_const_are_c2_invariant():
    assert is_invariant('axis', generator=2)
    assert is_invariant('const', generator=2)
    assert not is_invariant('parity', generator=2)     # same 1 bit, different partition
    assert not is_invariant('full', generator=2)


def test_only_const_is_c4_invariant():
    """axis breaks C4 -- a 90-degree turn swaps the two axes. So in the C4 arena it
    should lift the quotient partially: a dose-response, not a binary."""
    assert is_invariant('const', generator=1)
    for m in ('full', 'axis', 'parity'):
        assert not is_invariant(m, generator=1)


@pytest.mark.parametrize('mode', MODES)
def test_encoding_preserves_dimension_and_column_sum(mode):
    M = hd_matrix(mode).numpy()
    assert M.shape == (4, 4)
    assert np.allclose(M.sum(axis=0), 1.0)             # activation magnitude preserved


def test_apply_hd_only_touches_the_heading_block():
    act = torch.randn(2, 3, 5)
    for mode in MODES:
        out = apply_hd(act, mode)
        assert out.shape == act.shape
        assert torch.equal(out[..., :1], act[..., :1])  # speed untouched


def test_axis_collapses_opposite_headings_parity_does_not():
    for h in range(4):
        a, b = _onehot(h), _onehot((h + 2) % 4)
        av = torch.cat([torch.ones(1), a])[None]
        bv = torch.cat([torch.ones(1), b])[None]
        assert torch.allclose(apply_hd(av, 'axis'), apply_hd(bv, 'axis'))
        assert not torch.allclose(apply_hd(av, 'parity'), apply_hd(bv, 'parity'))
