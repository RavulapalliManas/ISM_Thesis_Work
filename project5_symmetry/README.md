# project5_symmetry — landmark symmetry and predictive cognitive maps

A predictive RNN (`pRNN_th`, after Levenstein et al. 2024) trained in MiniGrid arenas with
graded rotational landmark symmetry (C1 / C2 / C4), matched on observational
discriminability (ODI = 0.336).

**Central claim.** A self-motion signal can only resolve the symmetries under which it is
not itself invariant — and only when the objective requires integrating it.

## Results at a glance

Orbit-phase decoding (chance = 0.500). Under folding onto the quotient `X/G`, the hidden
state obeys `h(x) = h(R²x)`, so which element of the orbit produced it is unrecoverable.

| arena | full (2 bit) | parity (1 bit) | axis (1 bit) | const (0 bit) |
|---|---|---|---|---|
| s1 (C1, no symmetry) | 0.982 | 0.973 | 0.968 | 0.967 |
| s2 (C2) | 0.978 | 0.953 | **0.553** | **0.551** |
| s4 (C4) | 0.984 | 0.930 | **0.547** | **0.520** |

`axis` and `parity` carry **identical information** (one bit each) and differ only in
whether they are C2-invariant. In s2 they differ by 0.40 (p = 0.00216, the exact two-sided
Mann-Whitney floor at n = 6 vs 6); in s1, where there is no symmetry to fold onto, they are
indistinguishable (p = 0.18). *Information quantity does not fold the code; invariance does.*

And the effect requires prediction. With `k = 0` (a pure autoencoder: target is `obs[anchor]`,
verified exactly) every encoding folds, including the full compass:

| k | full | parity | axis | const | axis vs parity |
|---|---|---|---|---|---|
| 0 (autoencoder) | 0.536 | 0.541 | 0.539 | 0.539 | p = 0.59 |
| 1 | 0.975 | 0.941 | 0.545 | 0.546 | p = 0.0022 |
| 5 | 0.978 | 0.953 | 0.553 | 0.551 | p = 0.0022 |

Full numbers, the C4 ladder, and the engineering notes are in `../outputs/gpu_runs/RESULTS.md`.

## Reproducing

Requires a CUDA GPU (measured on an RTX 5090, torch 2.11 + cu128; sm_120 needs torch >= 2.7).

```bash
# 1. trajectories (deterministic given the arena seed)
python project5_symmetry/experiments/run_ensemble_sweep.py --help   # generates data on first use

# 2. the symmetry x HD-encoding sweep: 48 models as ONE vmap ensemble, ~68 min
python project5_symmetry/experiments/run_hd_invariance.py \
    --out runs/hd_invariance --seeds s1=6 s2=6 s4=4 \
    --hd-modes full axis parity const --n-steps 80000

# 3. the objective control: one ensemble per rollout horizon (each k is a distinct graph)
for k in 0 1 3; do
  python project5_symmetry/experiments/run_hd_invariance.py \
      --k $k --seeds s2=6 --hd-modes full axis parity const \
      --out runs/horizon/k$k --n-steps 80000
done

# 4. readouts
python project5_symmetry/analysis/run_phase_decoding.py --runs runs/hd_invariance \
    --group c2 --out phase.csv          # primary readout
python project5_symmetry/analysis/run_phase_decoding.py --runs runs/hd_invariance/s4 \
    --group c4 --out phase_s4_c4.csv    # the C4 ladder
python project5_symmetry/analysis/run_spectrum.py --runs runs/hd_invariance --out spectrum.csv
```

## Which readout to trust

Three were tried. Only one survives.

| readout | what it measures | verdict |
|---|---|---|
| `RA = P0 - P2` | rotational autocorrelation | **blind to C2 folding**: reads ~0 for a C2-folded code and ~0 for noise alike. Cannot separate s2 from s1 (p = 0.095). |
| `odd = P1 + P3` | power in the characters a 180° rotation flips | **confounded**: drops under `axis` even in s1, where there is no symmetry. 85% of the s4-sized effect appears in the C1 arena. |
| **orbit-phase decoding** | can a decoder name which element of the orbit produced `h`? | **primary.** Immune to spectral reshaping; clean null in s1. |

`domain_r2` is *not* a positive control — recovering a folded coordinate from an unfolded
code needs a nonlinear map, so a linear decoder scores ~0 there. It is a second folding
indicator pointing the other way. The genuine sanity check is `max(raw_r2, domain_r2)`,
which separates "folded" from "stopped encoding space".

## Layout

```
environments/
  arena.py              SymmetryArena: the 18x18 MiniGrid, s1/s2/s4 landmark layouts
  hd_encodings.py       full / axis / parity / const: 4x4 maps on the heading block
  topology_arenas.py    open, annulus, theta, figure-8 (b1 = 0, 1, 2) for the topology study
  two_room.py           translation arena is NOT an exact symmetry -- see its docstring
training/
  ensemble.py           vmap model-ensembling: S models as one GPU job, bit-exact per model
  inits.py              recurrent-init variants (tau, gain, orthogonal)
experiments/
  run_hd_invariance.py  the symmetry x HD x horizon sweeps
  run_topology.py       arenas with controlled b1, log-spaced checkpoints from step 0
  run_init_study.py     paired init sweep on annulus_w4
  run_cpu_worker.py     single-model CPU trainer. Kept as a documented negative result:
                        a CPU fleet costs more GPU throughput than it is worth.
analysis/
  run_phase_decoding.py orbit-phase decoding, C2 and C4 quotients   <- primary readout
  run_spectrum.py       C4 isotypic spectrum (secondary)
  tda.py, run_tda.py    persistent H1 (ripser, PCA-6, cosine, Z_47)
  topo_before_geom.py   saturation-step comparison across training
```

## Notes for anyone rerunning this

- **Each distinct `k` is a distinct computation graph**, hence a distinct inductor compile.
  Conditions, HD encodings and seeds all share a graph and ride in one ensemble; horizons
  do not.
- **Do not train on the CPU alongside the GPU.** Measured: 27 pinned CPU workers drop GPU
  utilisation from 95% to 28% (memory-bandwidth contention; `taskset` does not help), while
  the GPU trains each model 153x faster.
- **`ripser` with `maxdim=2` is roughly n⁴**: n=300 costs ~1 s, n=1200 does not finish.
  These arenas are planar (b2 = 0), so `maxdim=1` suffices.
- The test suite pins the things that fail silently: per-model gradient clipping, the
  ensemble's bit-exactness, that `k=0`'s target is exactly `obs[anchor]`, that a folded
  synthetic code reads chance while a dead one is distinguishable, and that checkpoints do
  not serialise the whole ensemble.

```bash
python -m pytest tests/project5_symmetry -q
```
