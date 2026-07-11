#!/usr/bin/env python3
"""Does compile rescue the ensemble at the production group size?

vmap alone measured ~170 ms/ensemble-step on an RTX 5090 (flat in S), i.e. ~10x
over the 73.6 ms/model-step sequential baseline at S=26. That is not enough.
This measures whether inductor fusion and CUDA graphs close the gap.

Prints incrementally: compile can take minutes at T=200 (the loop unrolls).
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/root/inductor_cache')

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.bench.bench_ensemble_gpu import (  # noqa: E402
    bench_ensemble, bench_sequential)

B, T = 16, 200


def main():
    S_list = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['17', '26'])]
    print(f'device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    print(f'shape : B={B} T={T}\n', flush=True)

    seq, _ = bench_sequential(B, T, iters=6, warmup=3)
    print(f'sequential baseline: {seq*1000:.1f} ms per model-step\n', flush=True)

    print(f"{'S':>4}{'mode':>18}{'ms/ens-step':>13}{'ms/model':>11}"
          f"{'speedup':>10}{'peak GB':>9}{'setup s':>9}")
    for S in S_list:
        for mode in (None, 'default', 'reduce-overhead'):
            label = mode or 'none'
            try:
                t0 = time.perf_counter()
                med, lo, peak, setup = bench_ensemble(S, B, T, mode, iters=5, warmup=2)
                print(f'{S:>4}{label:>18}{med*1000:>13.1f}{med*1000/S:>11.2f}'
                      f'{seq/(med/S):>9.1f}x{peak:>9.2f}{setup:>9.1f}', flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f'{S:>4}{label:>18}   OOM', flush=True)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'{S:>4}{label:>18}   FAILED {type(e).__name__}: {str(e)[:50]}', flush=True)
                torch.cuda.empty_cache()
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
