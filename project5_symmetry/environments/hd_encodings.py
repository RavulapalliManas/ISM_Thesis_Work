"""Head-direction encodings that dissociate INFORMATION from SYMMETRY.

The law under test
------------------
A self-motion signal can only resolve the symmetries under which it is not itself
invariant. If the HD encoding phi satisfies phi(h + g) = phi(h) for the generator
g of the arena's symmetry group G, then the entire (observation, action) input
process is G-equivariant, the posterior over position is G-invariant, and the
predictive code must collapse onto the quotient X/G.

The controls
------------
`axis` and `parity` both reduce the 4-way compass to exactly ONE BIT -- they
partition the four headings into two classes of two. They are matched in
information content, in dimensionality, and in activation statistics (two entries
at 0.5). They differ only in *which* partition:

    axis    {E,W} vs {N,S}    phi(h+2) == phi(h)   ->  C2-INVARIANT  -> must fold
    parity  {E,S} vs {W,N}    phi(h+2) != phi(h)   ->  breaks C2     -> can lift

So in the C2 arena they make opposite predictions:

    if folding is driven by how much information HD carries:  axis ~ parity
    if folding is driven by whether HD is G-invariant:        axis folds, parity does not

Note `axis` still breaks C4 (a 90-degree rotation swaps the two axes), so in the
C4 arena it should sit *between* `full` and `const` -- a dose-response, not a
binary. Only `const` is C4-invariant.

Encodings act on the 4-way one-hot heading block of the SpeedHD vector
(act = [speed, onehot(h)], act_size = 5). Each is a left-multiplication by a 4x4
matrix, so dimensionality and the column sums are preserved.
"""
from __future__ import annotations

import numpy as np
import torch

# cyclic shift P: e_h -> e_{h+1}.  h in {0=E, 1=S, 2=W, 3=N}
_P = np.roll(np.eye(4), 1, axis=0)


def _matrix(mode: str) -> np.ndarray:
    if mode == 'full':                       # 4-way compass, 2 bits
        return np.eye(4)
    if mode == 'axis':                       # {E,W} vs {N,S}: 1 bit, C2-INVARIANT
        return 0.5 * (np.eye(4) + _P @ _P)
    if mode == 'parity':                     # {E,S} vs {W,N}: 1 bit, breaks C2
        M = np.zeros((4, 4))
        M[0:2, 0:2] = 0.5
        M[2:4, 2:4] = 0.5
        return M
    if mode == 'const':                      # 0 bits (the classic HD ablation)
        return np.full((4, 4), 0.25)
    raise ValueError(f'unknown hd mode {mode!r}')


MODES = ('full', 'axis', 'parity', 'const')


def hd_matrix(mode: str, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(_matrix(mode), device=device, dtype=dtype)


def is_invariant(mode: str, generator: int) -> bool:
    """Is phi(h + generator) == phi(h) for every heading h?

    generator=2 -> the 180-degree rotation (C2);  generator=1 -> 90 degrees (C4).
    """
    M = _matrix(mode)
    shift = np.linalg.matrix_power(_P, generator)
    return bool(np.allclose(M @ shift, M))


def apply_hd(act: torch.Tensor, mode: str) -> torch.Tensor:
    """act (..., 5) = [speed, onehot(heading)]. Transform the heading block only.

    `mode='learned'` replaces the absolute compass with angular velocity (the per-step turn),
    matching run_hd_invariance.sample_batches, so a network trained with a learned compass is
    evaluated on the same input it saw.
    """
    if mode == 'full':
        return act
    if mode == 'learned':
        idx = act[..., 1:].argmax(-1)
        turn = (idx - torch.roll(idx, 1, dims=-1)) % 4
        turn[..., 0] = 0                                 # no reference at the first step
        hd = torch.nn.functional.one_hot(turn, 4).to(act.dtype)
        return torch.cat([act[..., :1], hd], dim=-1)
    M = hd_matrix(mode, device=act.device, dtype=act.dtype)
    speed, hd = act[..., :1], act[..., 1:]
    return torch.cat([speed, hd @ M.T], dim=-1)


def bits(mode: str) -> float:
    """Entropy of the encoding under a uniform heading prior, in bits."""
    M = _matrix(mode)
    classes = {tuple(np.round(M[:, h], 6)) for h in range(4)}
    n = len(classes)
    return float(np.log2(n)) if n > 1 else 0.0
