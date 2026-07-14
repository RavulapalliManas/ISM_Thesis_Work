# Data catalogue and quality assessment

Every data asset for the symmetry paper, where it lives, and how much weight it can bear.
Compiled 2026-07-14 by auditing all four stores, loading a sample of checkpoints, and
cross-checking all 44 result CSVs against the manuscript.

Counting note: the SSD is exFAT, so every real file has an AppleDouble sibling (`._name`).
All counts below exclude these. A naive `ls | wc -l` doubles everything.

---

## 1. The four stores

| Store | Size | What it is | Backed up off-drive? |
|---|---|---|---|
| `Report/data/` (44 CSVs) | 1.4 MB | Every number in the paper | **Yes** — git + GitHub |
| `analysis/`, `Report/*.tex` | 2 MB | Scripts and manuscript | **Yes** — git + GitHub |
| `prnn_backup/checkpoints/` | 533 MB | 345 trained networks, finals only | **No** |
| `pod_backups/runs/` | 2.4 GB | 223 runs **with training ladders** | **No** |
| `outputs/gpu_runs/` | 5.8 GB | `symmetry_sweep` ladder + a 2.8 GB self-duplicate | **No** |

**~8.8 GB of trained networks exist on exactly one exFAT drive.** The results survive its
loss; the ability to run any *new* analysis does not.

---

## 2. Checkpoints

2,091 real `.pt` files across three stores. All 11 sampled loaded cleanly under torch 2.6
(uniform schema `['step','model','meta']`, hidden size 500, rich `meta`).

### Which experiments have a training ladder

This is the single most important distinction in the archive, because it decides which
developmental questions can be asked at all.

| Group | Runs | Ladder? | Structure |
|---|---|---|---|
| `hd_invariance` (**the headline sweep**) | 112 | **No — finals only** | s1/s2/s4 × 4 encodings × 10 seeds (s4: 8) |
| `horizon` (rotational k-sweep) | 72 | **No — finals only** | k∈{0,1,3} × s2 × 4 encodings × 6 seeds |
| `multi` (init / topology / compartment) | 128 | **No — finals only** | 16 subgroups × 8 seeds |
| `topology` | 16 | **No — finals only** | 4 layouts × 4 seeds |
| `symmetry_sweep` | 17 | **Yes** (in `outputs/` only) | s1×5, s2×5, s4×7; 7 rungs + loss curves |
| `weakbreak` | 64 | **Yes** | 8 d′ × 8 seeds; 11 rungs to 80k |
| `horizon_k0/k1/k3` (**compartment** maze) | 48 | **Yes** | translation/rotation × 8 seeds; 11 rungs |
| `exp0_axisconst` | 20 | **Yes** | s2 × (axis, const) × 10 seeds; 7 rungs |
| `exp0_learned` | 10 | **Yes** | s2 × learned × 10 seeds; 7 rungs |

Consequence: **the axis-vs-parity contrast that carries the paper has no ladder.** The fold's
developmental trajectory can be measured for `axis` and `const` (exp0), for the graded-cue
sweep (weakbreak), and across arenas (symmetry_sweep) — but the matched `parity` control
would have to be retrained. That needs a GPU.

Training curves (`training_log.json`, loss vs step) exist for the 223 pod runs and the 17
`symmetry_sweep` runs. They do **not** exist for `hd_invariance`, `horizon`, `multi`, or
`topology` — for those, only endpoint weights survive. Nothing anywhere records wall-clock,
validation loss, or learning rate.

### Traps

- **`weakbreak_bench2` is not a replicate of `weakbreak`.** Same 64-run 8×8 grid, same
  directory shape, logs that terminate normally — but 200 steps, not 80,000 (final loss
  4.8e-2 vs 3.1e-3). It is a timing benchmark. Globbing the grid structure picks it up.
- **`horizon_k5` is aborted**: 16 × `ckpt_0.pt`, no final, no log. It never trained. (No loss:
  k=5 is covered by `hd_invariance` via `phase_groupB.csv`.)
- **`symmetry_sweep` checkpoints carry no `seed`/`condition` in their `meta`.** Their identity
  is encoded *only in the directory path*. Flatten or move those directories and 17 runs
  become unidentifiable.
- **`outputs/` vs `prnn_backup` `symmetry_sweep` finals**: file hashes differ (27 MB vs
  1.6 MB) but **every weight tensor is bit-identical**. Same networks, fatter serialization —
  not a second training run.
- **Seed IDs collide across groups.** `symmetry_sweep/seed_03` and `hd_invariance/seed_03` are
  different networks. Merging `isotypic_symmetry` with `isotypic_hd` on `(condition, seed)`
  silently mixes two populations. Only the `path` column disambiguates, and
  `map_quality_group{A,B}.csv` lack it.

### Reclaimable

`outputs/gpu_runs/symmetry_sweep.tgz` (2.8 GB) is an archive of the directory sitting next to
it. Pure duplicate, zero loss. `prnn_backup/paper/` (53 MB) duplicates `Report/archive/`,
which is git-tracked.

---

## 3. Result CSVs (44, all git-tracked)

The headline files are clean and well-powered:

| File | n | Backs |
|---|---|---|
| `phase_full_n10.csv` | 10/10/8 × 4 | Table 2, abstract, Fig 2b — the central dissociation |
| `compartments.csv` | 8 + 8 | translation folds / rotation lifts, Fig 3 |
| `compartments4.csv` | 8 | the Spiers four-room fold, Fig 6 |
| `phase_learned_c2.csv` | 10/10/8 | the learned unanchored compass |
| `remapping.csv`, `isotypic_hd.csv` | 10/10/8 × 4 | Fig 5, population panels |
| `weakbreak_snr.csv` | 8 × 8 d′ | the honest negative on rate remapping |

No NaNs, no infinities, no out-of-range values anywhere in the load-bearing set.

**The defects are all in the robustness checks that defend the headline claims** — which is
exactly backwards, since those files exist to answer a referee's first objection.

---

## 4. Quality assessment: what will not survive scrutiny

### P0 — The permutation null is a debug run, and re-running it as planned will not fix it

`perm_null_validation.csv` (2 rows) is the output of `perm_null_geometry.py --quick`, which
overrides the target list to 2 cells and `n_perm` to 40. The script's real defaults are 500
permutations over 5 targets. Two consequences:

1. **The reported p = 0.0244 is exactly 1/41** — the resolution floor of a 40-permutation
   test. It cannot go lower. For the `parity` row (observed 0.964 vs null 0.498 ± 0.006,
   z ≈ 85) quoting p = 0.0244 understates the significance by dozens of orders of magnitude.
2. **Three of the four folded cells have no null at all** (`s2/const`, `s4/axis`, `s4/const`).

And the part that is easy to miss: **line 74 hardcodes `seed_00`.** Even the full
500-permutation run is *n = 1 network per cell*. The multi-seed null the paper needs does not
exist behind the `--quick` flag; it requires a code change to loop over seeds.

The manuscript (main_best.tex:363–366) discloses the single-seed check honestly and promises
"a full multi-seed run is in progress." That promise is unkept, and the script as written
cannot keep it. The residual is also characterised at 0.531 (seed_00) while the n=10 group
mean in the headline table is 0.552 — the paper's residual number comes from a weaker sample
than its own Table 2.

### P1 — A main-text control figure has no generating script

`phase_nonlinear.csv` (columns `hd_mode, seed, linear, knn, mlp`) backs a **main-text** claim
(main_best.tex:315–321): that a kNN decoder and an MLP also sit at chance, so the fold is not
a linear-decoder artefact. This is the direct rebuttal to the most obvious referee objection.

**No script in the repository produces this file.** `run_phase_decoding.py` imports only
`LogisticRegression` and `Ridge`. The `KNeighborsClassifier` hits elsewhere are position
decoders in the compartment analyses, with different columns. The file is read by
`make_paper_figures.py:1060` for `figS_robustness`, so a published figure and a main-text
sentence currently cannot be regenerated. **This is the worst reproducibility hole in the set.**

### P2 — Two robustness claims rest on n = 1 network

- `manifold_robustness.csv` — 4 rows, **no seed column**. Backs "the fold is not a PCA
  artefact" (fold ratio under PCA / Isomap / t-SNE). One network per condition, and both
  Isomap and t-SNE are stochastic and seed-sensitive.
- `diffusion_maps.csv` — 7 rows, **no seed column**. Backs "axis 0.22 vs full 3.79" in
  diffusion distance.

A robustness claim with no seed dimension is not a robustness claim. Both are cheap to fix:
the checkpoints exist.

### P3 — Underpowered cells in the headline set

`phase_s4_c4.csv` is **n = 4** per encoding, and it carries **Table 3** — the C4 result that
elevates the paper from a C2 curiosity to a general quotient law. An exact two-sided
Mann–Whitney at n=4 vs 4 floors at p = 0.0286; nothing in that table can ever be reported
below it. `field_stats.csv` and `map_quality_groupB.csv` are also n=4 in s4.

The s4 arena is systematically the thin one. (The n=8 in the big s4 files is *disclosed* in
the paper — Table 2's caption says so — and is not a defect; the n=4 files are.)

### P4 — Smaller items

- `phase_groupB.csv` is a stale re-analysis of the same networks as `phase_full_n10.csv`:
  100% key overlap, but not one of 48 `phase_acc` values is bit-identical (max Δ = 0.004).
  Figures 2 and 3 therefore plot **slightly different numbers for the same networks**. Its s1
  cells are n=2.
- `tda_topology.csv`: the `variant` column is 100% NaN; `step` mixes ints with the string
  `'final'` and will coerce to object dtype in any `groupby`.
- `horizon_sweep_compartment.csv` is a 3-of-5 sweep (k=5 interrupted, k=10 never started); its
  "no rise with k" Spearman is computed over three x-values.
- `exp0_*_v2.csv`: 15 uncorrected Wilcoxon tests; the `learned` t100+ result at p = 0.049 does
  not survive any correction. The README is candid about this.
- `hockeimer_summary.csv`: 5 rats, but two contribute 165 of 238 fields.

---

## 5. What is gone

- **The raw Hockeimer CA1 dataset.** `hockeimer_reanalysis.py` expects
  `AlleySuperpopDirVisitFiltered.csv`; it is nowhere on the drive. Only the two derived CSVs
  survive. The paper's only real-neuron result **cannot be regenerated** without
  re-downloading from JHU (doi:10.7281/T15HMQD4). Cheapest gap on this list to close.
- **The main trajectory corpus.** `TrajectoryDataset` raises `FileNotFoundError` on an empty
  directory — it does not generate on demand. No trajectories exist for `symmetry`,
  `compartment`, `topology`, `cityblock`, or `hd_invariance`; the only caches on disk are two
  leftovers from the last referee analyses. Regeneration *is* deterministic
  (`default_rng(seed=indices[0])`), **but the per-condition `n_traj` and `T` are not recorded
  anywhere.** Write them down before they are lost.
- **The pod training logs.** `prnn_backup/pod_final/logs/` is an empty directory.

## 6. What is dead weight

- `project3_generalization/`, `project4_topology_before_geometry/`, and repo-root `analysis/`
  are **pure `.pyc` orphans** — zero source files. The source was deliberately removed in
  `ae5cf22a` and is recoverable from git. Safe to delete from disk.
- Repo-root `utils/` is the opposite: **16 project5 files import it**
  (`from utils.Architectures import pRNN_th`). It is a hard dependency, not legacy.

---

## 7. Verdict

Reproducibility is good where it matters: training, trajectory generation, and decoding are
all seeded, so the pipeline is deterministic given the checkpoints. The headline results are
clean, well-powered, and traceable to CSVs that are safely on GitHub.

The exposure is concentrated in three places, in priority order:

1. **The defences, not the claims.** P0–P2 are all robustness checks — the permutation null,
   the nonlinear-decoder control, the PCA-artefact control. Each exists to answer an
   anticipated objection, and each currently fails the follow-up question. All are cheap to
   fix on CPU because the checkpoints exist.
2. **A single exFAT drive** holds 8.8 GB of irreplaceable trained networks with no off-drive
   copy. Everything in §4 that is "cheap to fix because the checkpoints exist" stops being
   fixable the day that drive fails.
3. **Two inputs are already gone** (raw CA1 data, trajectory corpus) and the parameters needed
   to regenerate the second are undocumented.
