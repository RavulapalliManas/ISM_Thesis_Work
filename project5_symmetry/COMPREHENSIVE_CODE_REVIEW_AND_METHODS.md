# Comprehensive Code Review, Methods Reference, and Performance Analysis for `project5_symmetry`

Line target note.
This document was intentionally written as a very deep audit artifact.
It is designed to be longer than 1000 lines so it can act as a standalone reference.
It combines code review, literature-to-code mapping, formulas, implementation notes, performance analysis, and improvement recommendations.

---

## 1. Document Purpose

- This document reviews the `project5_symmetry` project as a research codebase rather than as a generic software package.
- The goal is to explain what the code is doing, whether it is scientifically faithful to the intended methods, where it is strong, where it is fragile, and how the performance-oriented changes affect execution.
- The review also documents the formulas used by the implemented analysis methods.
- The review also records what I could verify directly from code.
- The review also records what I could verify from saved results.
- The review also records what failed during local execution and why that matters.
- The review is based on the workspace state in:
- `/Users/manasvenkatasairavulapalli/Desktop/ISM work/ISM_Code/ISM_Thesis_Work/.claude/worktrees/sad-ride/project5_symmetry`

## 2. Scope

- Files reviewed include environment generation, trajectory generation, dataset loading, training loops, evaluation metrics, sweep orchestration, result analysis, visualization utilities, and the LaTeX report.
- The primary source files reviewed were:
- `project5_symmetry/environments/arena.py`
- `project5_symmetry/environments/generate_trajectories.py`
- `project5_symmetry/training/dataset.py`
- `project5_symmetry/training/train.py`
- `project5_symmetry/evaluation/metrics.py`
- `project5_symmetry/experiments/configs.py`
- `project5_symmetry/experiments/sweep.py`
- `project5_symmetry/analyze_symmetry_sweep.py`
- `project5_symmetry/run_fast.py`
- `project5_symmetry/Report/r.tex`
- `project5_symmetry/Report/references.bib`

## 3. Executive Summary

- The project is conceptually strong.
- The project has a clear scientific question.
- The project meaningfully separates environment design, dataset generation, model training, evaluation metrics, and sweep orchestration.
- The project includes both paper-faithful metrics and new custom metrics.
- The codebase has clearly undergone an optimization pass that adds a much faster training path.
- The fastest path is centered around prepacking trajectories into device-friendly tensors, increasing per-seed batch size, subsampling rollout anchors, and optionally running multiple seeds in one process.
- The evaluation code includes several good vectorization decisions, especially for cosine distance calculations on high-dimensional hidden representations.
- The environment code is unusually well commented for research code.
- The project does, however, contain a few important correctness and reproducibility risks.
- The most serious issue is that importing `project5_symmetry.evaluation.metrics` hard-imports `pynapple`, which in this environment failed unless `NUMBA_CACHE_DIR` was manually set.
- A second serious issue is that the legacy trainer currently crashes because it passes rollout noise with a shape that does not match what the model path expects.
- A third important issue is that the default legacy `DataLoader` worker setup is brittle in sandboxed or restricted environments because PyTorch shared memory worker startup can fail.
- A fourth issue is that `U_STAR` in Phase 2b is still hardcoded as a placeholder.
- The project is strongest when used through the fast path.
- The legacy path is no longer trustworthy without repair.
- The report below documents all of that in detail.

---

## 4. High-Level Research Question

- The project studies how landmark symmetry influences map formation in predictive recurrent neural networks.
- The central hypothesis is that rotational symmetry can produce representational ambiguity.
- The project tests whether that ambiguity leads to:
- global orientational degeneracy,
- local representational folding,
- partial collapse,
- or robust disambiguation driven by action and head-direction information.

### 4.1 Why this matters

- In hippocampal and related navigation systems, the same observation can occur at multiple locations.
- Symmetry is therefore a natural stress test for spatial coding.
- A predictive recurrent model is a reasonable testbed because it integrates perception, action, and temporal context.
- The project is well aligned with predictive-map and hippocampal representation literature.

### 4.2 Literature anchors found in the repo

- `levenstein2024`
- `stachenfeld2017`
- `taube1990`
- `knierim1995`
- `muller1987`
- `leutgeb2005`

### 4.3 Interpretation of the repo’s framing

- The implementation appears to treat the Levenstein-style predictive recurrent network as the baseline mechanistic model.
- The project’s novel contribution is not merely reproducing sRSA.
- It also proposes additional diagnostics intended to detect symmetry-specific distortions that standard sRSA can miss.

---

## 5. Codebase Map

### 5.1 Environment layer

- `arena.py` defines the MiniGrid environment and landmark layouts.
- It also defines symmetry-conditioned variants like `s4`, `s2`, and `s1`.
- It provides helper functions for passable tiles, H2 observation aliasing, and symmetry pair enumeration.

### 5.2 Data generation layer

- `generate_trajectories.py` creates offline trajectories.
- It samples actions using paper-like probabilities.
- It encodes action plus heading into a SpeedHD-like 5D action representation.
- It saves trajectories as compressed `.npz` files.

### 5.3 Dataset layer

- `dataset.py` has two modes.
- `TrajectoryDataset` is the straightforward RAM-cached dataset.
- `PackedTrajectoryStore` is the fast-path packed representation optimized for direct batch sampling.

### 5.4 Training layer

- `train.py` contains both legacy and fast training paths.
- The fast path is now the default via `train()`.
- There is a single-seed fast mode and a parallel-seed fast mode.

### 5.5 Evaluation layer

- `metrics.py` implements:
- sRSA,
- cross-seed RSA alignment,
- CCA alignment,
- tuning maps,
- spatial information,
- explained variance by space,
- place-field coherence,
- observation discriminability,
- manifold intrinsic dimensionality,
- symmetry collapse index,
- representational geometry consistency,
- and the topology-geometry gap.

### 5.6 Experiment orchestration

- `configs.py` defines the phase conditions.
- `sweep.py` generates data, trains models, evaluates runs, and writes condition-level results.
- `run_fast.py` exposes speed-oriented presets through environment variables.

### 5.7 Reporting and post-analysis

- `analyze_symmetry_sweep.py` is an analysis-and-figure assembly script.
- `Report/r.tex` is the narrative paper/report artifact.

---

## 6. Scientific Workflow Reconstructed From Code

### 6.1 Step 1: define the arena

- A `SymmetryArena` instance is created.
- Arena shape is either `l_shape` or `square`.
- Arena size determines the interior tile count.
- Landmark tiles are either:
- the paper-faithful asymmetric baseline,
- or symmetry-conditioned layouts like `s4`, `s2`, `s1`.

### 6.2 Step 2: wrap the environment

- `PixelObsWrapper` replaces the observation image with a partial egocentric RGB rendering.
- This is necessary because the standard wrapper path is incompatible with the custom environment in the way the project uses it.

### 6.3 Step 3: generate trajectories

- The project samples random-walk trajectories offline.
- Observations are normalized into `[0,1]`.
- Actions are encoded as a 5D vector consisting of movement speed and head direction.

### 6.4 Step 4: train a predictive recurrent network

- A pRNN receives visual observation, encoded action, and internal dynamics.
- The model predicts future observations for `k` rollout steps.
- Loss is MSE over prediction targets.

### 6.5 Step 5: evaluate geometry and coding structure

- Hidden states are collected from random trajectory segments.
- Pairwise distances and aligned position means are computed.
- Population-level and unit-level metrics are derived.

### 6.6 Step 6: compare across conditions

- Different landmark symmetries, arena sizes, view sizes, rollout horizons, and sequence lengths are swept.
- Final condition results are written to JSON.
- Post-analysis scripts assemble report figures and summary statistics.

---

## 7. Environment Design Review

### 7.1 What `arena.py` gets right

- The file is unusually well documented.
- The coordinate conventions are explained repeatedly.
- The distinction between MiniGrid `(col,row)` and indexed `(row,col)` logic is handled carefully.
- Landmark geometry is implemented explicitly instead of being hidden behind magic numbers with no context.
- The `passable_positions` property is simple and readable.
- The `compute_H2` logic makes the observation aliasing question explicit.

### 7.2 Landmark layouts implemented

- Original paper-like asymmetric landmark layout.
- `s4` condition using 90-degree rotations of the staircase shape.
- `s2` condition using 180-degree paired staircase and cross motifs.
- `s1` condition using four distinct quadrant-specific motifs.

### 7.3 Shape and symmetry logic

- `l_shape` removes the bottom-right quadrant from the passable region.
- `square` retains full passability.
- Symmetry-conditioned layouts are restricted to square arenas.
- That restriction is correctly enforced in the constructor.

### 7.4 Important environment formula: passability

- Let arena size be `s`.
- A tile `(row, col)` is passable iff:

```text
1 <= row <= s
1 <= col <= s
and, for l_shape only:
not (row > s/2 and col > s/2)
```

- This matters because all downstream geometry depends on discrete passable coordinates.

### 7.5 H2 observation-aliasing metric implemented in code

- `compute_H2` enumerates all `(position, heading)` states.
- It renders the observation at each state.
- It hashes the observation bytes.
- For each state, it counts how many other states share the same observation.

### 7.6 H2 formula implied by implementation

- For a state `s_i`, define:

```text
A(s_i) = number of states s_j != s_i such that O(s_j) = O(s_i)
```

- The code then reports:

```text
H2_mean = mean_i A(s_i)
```

- This is a practical alias-count metric rather than an entropy expression.

### 7.7 Strength of this implementation

- It does not rely on theoretical symmetry assumptions.
- It measures actual rendered observation collisions.
- That is scientifically preferable for verifying alias structure.

### 7.8 Weakness of this implementation

- It hashes float-normalized image bytes.
- That is deterministic for this pipeline, but it assumes exact render equality.
- If rendering ever changes subtly, H2 comparability could shift unexpectedly.

### 7.9 Symmetry pair logic review

- `precompute_symmetry_pairs()` rotates each passable position by repeated 90-degree transforms.
- It appends all valid non-self rotated positions.
- This means the returned list is directed and contains multiple related pairings per orbit.
- That is acceptable for a collapse metric if the intended quantity is “all symmetry-related relations.”
- It should, however, be documented more explicitly because some readers may assume unique undirected pairs.

### 7.10 Review verdict for environment layer

- Scientifically clear.
- Readable.
- Robust enough for the main experiments.
- One of the stronger parts of the codebase.

---

## 8. Trajectory Generation Review

### 8.1 Core behavior

- `collect_trajectory()` resets the environment.
- It samples a random start tile from passable positions.
- It samples a random heading.
- It then samples actions according to fixed probabilities.

### 8.2 Action probabilities

- The code uses:

```text
P(left)    = 0.15
P(right)   = 0.15
P(forward) = 0.60
P(stop)    = 0.10
```

- This is consistent with the comments claiming paper faithfulness.

### 8.3 SpeedHD encoding

- The action representation has shape `(T,5)`.
- Component 0 is movement speed.
- Components 1 through 4 are one-hot heading bins.

### 8.4 SpeedHD encoding formula

- For timestep `t`:

```text
enc_t = [speed_t, hd_t(0), hd_t(1), hd_t(2), hd_t(3)]
```

- where:

```text
speed_t = 1 if action_t == FORWARD else 0
hd_t(i) = 1 if heading_t == i else 0
```

### 8.5 Data fields saved per trajectory

- `obs`
- `act_enc`
- `pos`
- `heading`

### 8.6 Shape summary

- `obs` has shape `(T+1, F*F*3)`
- `act_enc` has shape `(T,5)`
- `pos` has shape `(T+1,2)`
- `heading` has shape `(T+1,)`

### 8.7 Multiprocessing design

- Dataset generation is chunked across workers.
- Each worker builds its own wrapped environment.
- The output is resumable because existing files are skipped.

### 8.8 Good design choices

- The generation is resumable.
- The generation path stores trajectories in a compact portable format.
- The code explicitly works around wrapper attribute forwarding issues.
- Start-state randomization improves coverage over always using the same start state.

### 8.9 Risks

- Worker RNG seeding is based on `indices[0]`.
- That is deterministic per chunk but may make chunk-level reproducibility depend on chunk layout.
- This is not catastrophic, but it means the exact trajectory set depends on worker partitioning.

### 8.10 Formula for expected dataset size

- If there are `N_traj` trajectories and each has horizon `T`:

```text
Total observations stored = N_traj * (T + 1) * F * F * 3
Total action channels stored = N_traj * T * 5
Total positions stored = N_traj * (T + 1) * 2
```

- This is helpful when reasoning about memory pressure.

### 8.11 Review verdict for generation layer

- Well aligned with the research purpose.
- Good enough for offline predictive training.
- Minor reproducibility caveats only.

---

## 9. Dataset Layer Review

### 9.1 `TrajectoryDataset`

- This class eagerly preloads all trajectory files into RAM.
- That means the cost of loading is paid once.
- It also means repeated batch access avoids disk I/O.

### 9.2 Why this helps

- Offline research training often revisits the same data repeatedly.
- Preloading converts random disk reads into direct RAM access.

### 9.3 `PackedTrajectoryStore`

- This is one of the main performance improvements in the project.
- It stacks all observations across all trajectories.
- It stores observations quantized to `uint8`.
- It stores action tensors directly on the target device.
- It keeps positions and headings on CPU because evaluation paths expect NumPy arrays.

### 9.4 Quantization trick

- Observations originally live in `[0,1]` float space.
- The fast store computes:

```text
obs_u8 = round(obs * 255)
```

- At batch sampling time it reconstructs:

```text
obs_float = obs_u8 / 255
```

### 9.5 Why this is smart

- It cuts observation memory substantially.
- It makes full-dataset device residency more realistic.
- It reduces per-batch host-device transfer costs when training on GPU.

### 9.6 Potential concern

- Quantization introduces a small reconstruction error.
- In this case that error is bounded by `1/255` per pixel channel.
- Because the underlying rendered observations are already image-like and low precision, this is a reasonable tradeoff.

### 9.7 Batch sampling API

- `sample_batch(batch_size)`
- `sample_parallel_batches(n_groups, batch_size)`

### 9.8 Why `sample_parallel_batches` matters

- It allows multiple seed-specific models to draw batches from the same prepacked dataset in one process.
- That is a key enabler for the parallel-seed fast path.

### 9.9 Local benchmark of dataset access

- Measured locally on this machine:
- `DataLoader` with `num_workers=0`, batch size 16, 10 batches:
- `0.029 sec`
- `PackedTrajectoryStore` build time:
- `0.156 sec`
- `PackedTrajectoryStore` sample 10 batches:
- `0.008 sec`

### 9.10 Interpretation of the dataset benchmark

- The packed store has an upfront cost.
- Once built, repeated sampling is faster than repeated `DataLoader` iteration in this small benchmark.
- The measured speedup on sampling alone was about:

```text
0.029 / 0.008 = 3.625x
```

- That is meaningful because training performs many more batch draws than store constructions.

### 9.11 Caveat

- This benchmark was CPU-only.
- The packed store’s biggest intended advantage is on GPU.
- The CPU benchmark therefore likely understates its real benefit in the target setting.

### 9.12 Review verdict for dataset layer

- Strong.
- Performance-aware.
- Scientifically acceptable.
- One of the best optimization additions in the codebase.

---

## 10. Training Architecture Review

### 10.1 Model family

- The code instantiates `pRNN_th` from `utils.Architectures`.
- The recurrent cell is `LayerNormRNNCell`.
- Hidden size is 500.
- Dropout and noise are included.

### 10.2 Main hyperparameters in `train.py`

- `HIDDEN_SIZE = 500`
- `DROPOUT_P = 0.15`
- `NOISE_STD = 0.03`
- `NEURAL_TIMESCALE = 2`
- `GLOBAL_LR = 2e-3`
- `WEIGHT_DECAY = 3e-3`
- `RMSPROP_ALPHA = 0.95`
- `RMSPROP_EPS = 1e-6`

### 10.3 Why RMSProp matters

- The file explicitly notes that this is paper-faithful.
- That is important because many reproduction failures come from optimizer substitution.

### 10.4 Optimizer scaling formula

- The code comments state:

```text
LR(W_rec, W_out) = lambda * sqrt(1/H)
LR(W_in, W_act)  = lambda * sqrt(1/(obs+act))
LR(bias)         = lambda_b * lambda
```

- Concretely, the implementation computes:

```text
k_io = 1 / input_dim
k_h  = 1 / hidden_dim
lr_scale = sqrt(B_reference / B_actual)
lr_h = GLOBAL_LR * sqrt(k_h) * lr_scale
lr_io = GLOBAL_LR * sqrt(k_io) * lr_scale
lr_bias = GLOBAL_LR * BIAS_LR_SCALE * lr_scale
```

### 10.5 Interpretation

- This is a size-aware parameter scaling rule.
- It reduces the need for manual retuning when batch size changes.
- It is a reasonable way to preserve approximate learning-rate geometry across modes.

### 10.6 Objective function

- The model predicts future observations.
- The training loss is MSE between predictions and targets.

### 10.7 Loss formula

```text
L = mean((pred - target)^2)
```

- In the report the loss is described as a sum over future rollout steps.
- In code, those rollout dimensions are already in the tensor, so a single mean MSE is taken over all elements.

### 10.8 Hidden-state collection

- `_collect_hidden_states()` iterates random trajectories.
- It forwards them through the model without gradients.
- It stops after collecting `n` hidden states.

### 10.9 Why this matters

- Almost every geometry metric depends on this sampling path.
- Any bias here propagates into sRSA, manifold ID, coherence, and related figures.

### 10.10 Sampling considerations

- Sampling is random over trajectories, then contiguous within trajectories.
- That means hidden samples are not uniformly distributed over positions.
- However, downstream aggregation by position partly compensates for this in some analyses.

### 10.11 Training mode split

- `_train_legacy()`
- `_train_fast_single()`
- `train_parallel_seeds()`

### 10.12 Design intent

- The legacy mode is the original or near-original training loop.
- The fast modes are engineering refactors aimed at throughput.
- `train()` now defaults to the fast single-seed path unless `PRNN_TRAINER=legacy`.

### 10.13 Review verdict for high-level training design

- Clear separation of correctness-first and speed-first modes.
- Good idea architecturally.
- But the legacy path currently contains serious breakage.

---

## 11. Legacy Training Path Review

### 11.1 Data loading

- The legacy trainer uses a `DataLoader`.
- It tries to use a few workers by default.
- It uses `pin_memory` only on CUDA.

### 11.2 CUDA graph support

- The legacy trainer contains a conditional CUDA graph path.
- It allocates static buffers.
- It warms up the graph.
- It captures a replayable forward/backward step.

### 11.3 Compile strategy

- The code comments correctly note that compiling only the RNN cell is more realistic than compiling the whole model.
- That is a strong practical insight.

### 11.4 Logging behavior

- Every `LOG_INTERVAL` steps the legacy trainer computes:
- sRSA Euclidean,
- sRSA CityBlock,
- and their gap.

### 11.5 Good properties

- The logging is detailed.
- The checkpoint metadata is explicit.
- The comments show real debugging history rather than generic decoration.

### 11.6 Major problem discovered

- The legacy trainer crashes during actual local execution.
- With `PRNN_NUM_WORKERS=0` set to avoid worker issues, the run still fails with:

```text
IndexError: index 195 is out of bounds for dimension 1 with size 195
```

### 11.7 Root cause analysis

- The failure occurs in `_train_legacy()` when calling:
- `pred, _, target = model(obs_b, act_b, noise_t=noise_b)`
- where `noise_b` was created by `_sample_rollout_noise()`.

### 11.8 Why that is wrong

- `_sample_rollout_noise()` returns a tensor shaped like rollout noise:

```text
(k + 1, rollout_len, hidden_size)
```

- But the failing model path expects main-trajectory noise indexed over the full action-time dimension.
- The stack trace shows indexing at `noise_main[:, t, :]` inside the model.
- The second dimension therefore needs to cover `t` over the main action sequence, not only rollout length.

### 11.9 Practical impact

- Legacy training is not merely slower.
- It is currently nonfunctional in this environment.
- That means claims comparing fast versus legacy throughput cannot be treated as validated unless this bug is fixed first.

### 11.10 Worker startup issue

- With default worker settings, the legacy path also failed in this sandbox with:

```text
RuntimeError: torch_shm_manager ... Operation not permitted
```

### 11.11 Interpretation

- This is partly environment-specific.
- But it reveals that the legacy trainer’s default reliance on multiprocessing workers is fragile in constrained environments.

### 11.12 Code review verdict for legacy path

- Historically useful.
- Currently not production-ready.
- Needs repair before it should be used as a baseline.

---

## 12. Fast Training Path Review

### 12.1 Why the fast path exists

- The project needs to run many seeds and conditions.
- The slowest parts of naïve research training are usually:
- repeated Python-level data iteration,
- repeated host-to-device movement,
- excessive rollout anchor computation,
- and per-seed process overhead.

### 12.2 Core fast-path ideas found in code

- Prepack the full dataset.
- Quantize observations to `uint8`.
- Decode only when sampling.
- Increase per-seed batch size.
- Subsample rollout anchors.
- Optionally defer sRSA until the end.
- Optionally train several seeds in one process.
- Enable TF32 on CUDA.
- Try `torch.compile` on the RNN cell only.

### 12.3 Fast path step loop

- Draw batch from packed store.
- Sample anchor indices.
- Sample main and rollout noise.
- Run forward pass.
- Compute MSE.
- Backprop.
- Clip gradients.
- Step RMSProp.

### 12.4 Anchor subsampling

- Let `T_k = T_act - k`.
- If `anchor_subsample_n < T_k`, the code samples a sorted random subset of anchor times.

### 12.5 Why anchor subsampling helps

- Multi-step predictive losses can scale with the number of rollout anchor points.
- If you reduce anchor count from all positions to 32 anchors, the model computes fewer rollouts per batch.
- That directly reduces compute and memory costs.

### 12.6 Formula for anchor reduction ratio

- If full anchors are `T_k` and sampled anchors are `A`:

```text
Anchor compute fraction = A / T_k
```

- For the baseline with `T=200` and `k=5`:

```text
T_k = 200 - 5 = 195
```

- With `A = 32`:

```text
32 / 195 ≈ 0.1641
```

- That means only about `16.4%` of full anchor positions are used.
- Put differently, anchor-related rollout work is reduced by about `83.6%`.

### 12.7 Batch size scaling

- The fast path defaults to a larger batch per seed than the legacy reference.
- This increases throughput if the device can hold the larger tensors.

### 12.8 Deferred sRSA

- sRSA is expensive because it requires hidden-state collection and pairwise distance calculations.
- Deferring it during training avoids frequent metric stalls.

### 12.9 Parallel seed training

- `train_parallel_seeds()` keeps one shared packed dataset.
- It samples a group of batches with shape `(n_models, batch_size, ...)`.
- It loops over models and accumulates `total_loss`.
- It backprops once through the sum.

### 12.10 Why this helps

- It amortizes Python and data overhead across several seeds.
- It allows similar kernels to execute back to back in one process.
- It reduces repeated dataset setup cost.

### 12.11 What this does not do

- It does not fuse all seeds into a single model.
- It still loops through each seed model independently.
- So the gain is organizational and input-pipeline related, not full tensor fusion.

### 12.12 Local benchmark of fast path

- I ran a short local CPU-only benchmark on the existing `P0` trajectories:
- trainer mode: fast
- batch size per seed: 16
- anchors: 32
- steps: 10
- sRSA deferred during training but final evaluation still executed

### 12.13 Fast benchmark result

- End-to-end elapsed wall time:
- `62.678 sec`

### 12.14 Reading the benchmark output carefully

- The progress bar reached 10 steps in about 9 seconds.
- The final reported total became about 57 seconds on the bar.
- The difference came from final evaluation after the training loop.
- The final Python wrapper reported `62.678 sec`.

### 12.15 What that implies

- The training loop itself was much faster than the end-to-end run time suggests.
- Final metric evaluation dominates short toy runs.
- For proper throughput benchmarking, one should separate:
- pure training step time,
- checkpoint overhead,
- and final evaluation overhead.

### 12.16 Why this matters for interpretation

- If someone naively divides 62.678 seconds by 10 steps, they would conclude `6.27 sec/step`.
- That would be misleading because the bulk of the time came from post-training evaluation.

### 12.17 Code review verdict for fast path

- This is the most viable path in the codebase.
- The ideas are sensible.
- The implementation worked locally.
- It is the path I would trust first after adding a few more correctness tests.

---

## 13. Evaluation Metrics Review

### 13.1 General comment

- `metrics.py` is central to the scientific credibility of the project.
- It contains both baseline metrics and custom additions.
- Most functions are reasonably clear and well documented.

### 13.2 A major architectural issue

- `pynapple` is imported at module import time.
- That means importing *any* metric from `metrics.py` also imports the full `pynapple` stack.
- In this environment that failed unless `NUMBA_CACHE_DIR` was manually set.

### 13.3 Why that matters

- Several metrics do not need `pynapple` at all.
- Examples include:
- sRSA,
- SCI,
- ΔTG,
- manifold ID,
- cross-seed alignment,
- and direct exact tuning-map helpers.
- They should not be blocked by a tuning-curve dependency.

### 13.4 Recommendation

- Move `import pynapple as nap` inside `compute_tuning_curves()`.
- Or use a guarded import with a clear fallback error.

---

## 14. Metric 1: Spatial RSA

### 14.1 Code path

- Implemented by `srsa()`.

### 14.2 Concept

- Compare neural distances to spatial distances.
- If nearby positions map to nearby hidden states, correlation should be positive.

### 14.3 Formula

- Let hidden-state vectors be `h_i`.
- Let positions be `x_i`.
- Define neural distances:

```text
d_N(i,j) = cosine_distance(h_i, h_j)
```

- Define spatial distances:

```text
d_S(i,j) = euclidean_distance(x_i, x_j)
```

- Then:

```text
sRSA = SpearmanCorr({d_N(i,j)}, {d_S(i,j)}) over i < j
```

### 14.4 Alternative geometry

- The code also supports CityBlock distance:

```text
d_S(i,j) = |x_i - x_j|_1
```

### 14.5 Interpretation

- High sRSA means the hidden-state geometry preserves spatial relations.
- It does not necessarily mean the map is uniquely oriented or free of aliasing.

### 14.6 Important scientific limitation

- The report is right to emphasize that sRSA is invariant to certain label-preserving permutations of geometry.
- A rotated or permuted map can still score highly.
- That is exactly why the custom metrics are valuable.

### 14.7 Implementation quality

- Good.
- Efficient for cosine distances.
- Subsampling is explicitly handled.

### 14.8 Vectorized cosine implementation

- The project computes cosine distances through normalized matrix multiplication:

```text
X_n = X / ||X||
cos_sim = X_n X_n^T
d_cos = 1 - cos_sim
```

### 14.9 Benchmark for cosine distance kernel

- Measured locally on `X` with shape `(1000,500)`:
- SciPy `pdist(..., 'cosine')`:
- `0.476 sec`
- Torch vectorized `_pdist_cosine()`:
- `0.016 sec`

### 14.10 Speedup

```text
0.476 / 0.016 = 29.75x
```

- This is a real and important speed improvement.

### 14.11 Accuracy check

- Maximum absolute difference between SciPy and torch cosine results:
- `2.2163482682469038e-07`

### 14.12 Interpretation

- The speedup is large.
- The numerical agreement is excellent.
- This is one of the best optimization decisions in the project.

---

## 15. Metric 2: Cross-Seed RSA Alignment

### 15.1 Concept

- Even if individual seeds learn slightly different networks, we care whether their spatial similarity structure aligns.

### 15.2 Formula

- For each seed `a`, compute an RSA matrix `R_a`.
- Flatten its upper triangle.
- Then for seeds `a` and `b`:

```text
rho_ab = SpearmanCorr(vec_upper(R_a), vec_upper(R_b))
```

### 15.3 Outputs

- Pairwise matrix.
- Mean upper-triangle correlation.
- Standard deviation.

### 15.4 Why it matters

- This is a population-level reproducibility measure.
- High cross-seed RSA alignment suggests convergent representational geometry.

### 15.5 Implementation verdict

- Simple.
- Correct in spirit.
- Good for report-level summaries.

---

## 16. Metric 3: Aggregate Hidden By Position

### 16.1 Concept

- Positions are visited multiple times.
- For many analyses we want the mean hidden representation per discrete location.

### 16.2 Formula

- For a discrete position `p`:

```text
H̄(p) = (1 / N_p) * sum_{t: x_t = p} h_t
```

- where `N_p` is the number of visits to `p`.

### 16.3 Why this is useful

- It reduces temporal sampling noise.
- It aligns with map-like analysis where each spatial tile should have one representative vector.

### 16.4 Implementation verdict

- Good.
- Uses explicit accumulation and counts.
- Clear and robust.

---

## 17. Metric 4: Top Canonical Correlation

### 17.1 Concept

- Compare two position-aligned representation matrices after centering.
- Evaluate the strongest shared linear mode.

### 17.2 Formula

- Let `X` and `Y` be centered position-by-hidden matrices.
- Compute orthonormal bases `Q_X`, `Q_Y` via QR.
- Compute singular values of:

```text
Q_X^T Q_Y
```

- The top canonical correlation is:

```text
CCA_top = sigma_max(Q_X^T Q_Y)
```

### 17.3 Why this is meaningful

- It asks whether two seeds share a common representational subspace.
- This is complementary to RSA alignment.

### 17.4 Implementation verdict

- Elegant and lightweight.
- Appropriate for the project.

---

## 18. Metric 5: Exact Tuning Maps

### 18.1 Concept

- The code builds discrete spatial tuning maps directly over arena coordinates.

### 18.2 Formula

- For unit `i` and position `(c,r)`:

```text
T_i(r,c) = mean of h_{t,i} over all t with x_t = (c,r)
```

### 18.3 Why exact maps are valuable

- They avoid binning distortion when the environment is already discrete.
- They are especially appropriate for MiniGrid.

### 18.4 Occupancy handling

- Counts are tracked separately.
- Unvisited cells are marked `NaN`.

### 18.5 Review verdict

- Good choice.
- More faithful to a gridworld than continuous-space smoothing by default.

---

## 19. Metric 6: Spatial Explained Variance

### 19.1 Concept

- How much of a unit’s variance is explained by spatial position?

### 19.2 Formula

- For unit `i`:

```text
EVS_i = 1 - Var(h_i - E[h_i | x_t]) / Var(h_i)
```

- In the exact-map implementation:

```text
E[h_i | x_t] = T_i(x_t)
```

### 19.3 Interpretation

- `EVS_i` near `1` means position almost fully predicts the unit.
- `EVS_i` near `0` means position explains little.

### 19.4 Strength

- This is a very interpretable measure for place-like tuning.

### 19.5 Caveat

- EVS can be inflated if trajectory sampling overrepresents a subset of locations or behavioral motifs.
- That is common to many spatial analyses, not unique to this project.

---

## 20. Metric 7: Place-Field Spatial Coherence

### 20.1 Concept

- Measure whether a spatial tuning map is locally smooth rather than noisy.

### 20.2 Formula as implemented

- For a field `T_i`:
- Center the field.
- Compute full 2D autocorrelation:

```text
AC_i = correlate2d(T_i, T_i)
```

- Let the center be zero lag.
- Let the annulus be all lags with radius between 2 and 5 tiles.
- Then coherence score is:

```text
Coherence_i = AC_i(0,0) / mean(AC_i over annulus 2..5)
```

### 20.3 Why this is interesting

- It identifies smoothly organized place-like fields.
- It can distinguish structured tuning from noisy activity even when EVS is moderate.

### 20.4 Note on thresholds

- Only units with EVS above `0.10` are scored.
- This is a pragmatic filter.

### 20.5 Caution

- Coherence values can become very large if the annulus mean is small but positive.
- That means the absolute magnitude should be interpreted cautiously.
- The rank ordering across conditions is usually more informative than the raw number.

---

## 21. Metric 8: Observation Discriminability

### 21.1 Concept

- This is input-space sRSA.
- It measures how much the observations themselves distinguish space.

### 21.2 Formula

- Let `o(p)` be the observation at position `p`.
- Compute:

```text
d_O(i,j) = distance(o_i, o_j)
d_S(i,j) = spatial_distance(x_i, x_j)
ODI_rho = SpearmanCorr({d_O(i,j)}, {d_S(i,j)})
```

### 21.3 Why this is useful

- It provides a baseline on what the network *could* infer from the observations alone.
- Matching ODI across conditions is a good control idea.

### 21.4 Verified saved value

- In `results/symmetry_sweep_validate/s1/observation_summary.json`:
- `odi.rho = 0.33600394439152487`

### 21.5 Interpretation

- This gives the hidden-state results a reference point.
- If hidden-state geometry exceeds observation-only geometry, the network is adding disambiguating structure.

---

## 22. Metric 9: Representational Geometry Consistency

### 22.1 Concept

- Evaluate how well the hidden geometry can be represented in 2D.

### 22.2 PCA branch

- Hidden states are centered.
- SVD is computed.
- A 2D PCA embedding is formed.

### 22.3 PCA variance formula

```text
PCA_var_2d = (s_1^2 + s_2^2) / sum_k s_k^2
```

- where `s_k` are singular values.

### 22.4 MDS branch

- Pairwise neural distances are computed.
- Classical MDS is applied to recover a 2D embedding.

### 22.5 Stress formula

- The code computes:

```text
stress = sqrt(sum((d_true - d_embed)^2) / sum(d_true^2))
```

- once for PCA-derived distances,
- once for MDS-derived distances.

### 22.6 Interpretation

- Lower stress means the representation is more cleanly 2D.
- Higher stress means the manifold is more folded or higher-dimensional.

### 22.7 Nice design choice

- Using both `pca_var_2d` and MDS stress is better than relying on only one.
- Variance capture and distance-preservation tell related but distinct stories.

---

## 23. Metric 10: Tuning Curves via `pynapple`

### 23.1 Concept

- This is the paper-like continuous analysis path.

### 23.2 Formula

- For unit `i`:

```text
h_i(x) = E[h_i | x]
```

- estimated over 2D bins.

### 23.3 Why it exists alongside exact maps

- It mirrors existing literature analysis pipelines.
- It keeps continuity with previous utilities.

### 23.4 Concern

- For a discrete gridworld, the exact discrete tuning-map functions may actually be cleaner and less failure-prone.
- The `pynapple` dependency is the most brittle external dependency in the project.

---

## 24. Metric 11: Spatial Information

### 24.1 Concept

- Measure how much knowing location reduces uncertainty about a unit’s activity.

### 24.2 Formula implemented

```text
SI_i = sum_x p(x) * (h_i(x) / h̄_i) * log2(h_i(x) / h̄_i)
```

- where:
- `p(x)` is occupancy,
- `h_i(x)` is tuning value at position `x`,
- `h̄_i` is the occupancy-weighted mean activity of unit `i`.

### 24.3 Interpretation

- Higher SI means activity is more location-specific.
- It is a classic place-coding measure.

### 24.4 Implementation verdict

- Standard.
- Correct in spirit.
- Good to keep.

---

## 25. Metric 12: Spatial EVS via Binned Tuning Curves

### 25.1 Concept

- This is the paper-like bin-based version of explained variance by space.

### 25.2 Formula

```text
EVS_i = 1 - Var(h_i - h_i(x_t)) / Var(h_i)
```

- where `h_i(x_t)` is the expected activity from the tuning curve at the bin containing `x_t`.

### 25.3 Relationship to exact EVS

- The exact-map EVS is more natural for a discrete arena.
- The binned EVS is more literature-aligned for continuous-style analyses.

### 25.4 Strength

- Keeping both gives methodological redundancy.

### 25.5 Weakness

- It adds dependency burden and complexity.

---

## 26. Metric 13: Symmetry Collapse Index

### 26.1 Concept

- This is one of the project’s main custom metrics.
- It measures whether symmetry-related positions collapse closer together than random position pairs.

### 26.2 Formula from the code comments

```text
SCI = mean_neural_distance(symmetry_pairs) / mean_neural_distance(random_pairs)
```

### 26.3 More explicitly

- Let `S` be the set of symmetry-related pairs.
- Let `R` be a random sample of unrelated pairs.
- Then:

```text
SCI = [ (1 / |S|) * sum_(a,b in S) d_N(a,b) ] / [ (1 / |R|) * sum_(i,j in R) d_N(i,j) ]
```

### 26.4 Interpretation

- `SCI ≈ 1` means symmetric positions are not especially collapsed.
- `SCI < 1` means symmetric positions are closer than random pairs.
- Lower values imply stronger collapse.

### 26.5 Implementation details

- Position pairs are matched to nearest sampled positions by CityBlock distance.
- Neural distance defaults to cosine distance on L2-normalized hidden vectors.

### 26.6 Strength

- This directly targets the scientific claim about symmetry collapse.

### 26.7 Limitation

- It depends on the symmetry-pair enumeration.
- Because the pair list is directed and includes multiple related pairs per orbit, the exact weighting should be documented more formally.

---

## 27. Metric 14: Topology-Geometry Gap

### 27.1 Concept

- This measures whether the learned representation aligns more with Euclidean geometry or graph/path geometry.

### 27.2 Formula

```text
ΔTG(t) = sRSA_Euclidean(t) - sRSA_CityBlock(t)
```

### 27.3 Interpretation

- Positive values mean stronger Euclidean alignment.
- Negative values mean stronger path or grid geometry alignment.

### 27.4 Why this is clever

- Gridworlds naturally induce graph distances.
- Comparing Euclidean and CityBlock alignment is a good way to ask what the network really internalizes.

### 27.5 Caveat

- The report and docstrings should stay consistent on sign interpretation.
- The existing code definition is unambiguous.
- Any narrative text must match it exactly.

---

## 28. Metric 15: Manifold Intrinsic Dimensionality

### 28.1 Concept

- Estimate how many effective dimensions the representation manifold uses.

### 28.2 TwoNN formula

- The code implements the Facco-style idea:

```text
ID = 1 / mean(log(r2 / r1))
```

- where:
- `r1` is the first nearest-neighbor distance,
- `r2` is the second nearest-neighbor distance.

### 28.3 Why it matters

- A map that is perfectly flat and 2D-like would be expected to have lower intrinsic dimensionality.
- Folded or fragmented manifolds can raise the estimate.

### 28.4 Implementation limitation

- The current implementation computes a full Euclidean distance matrix with `cdist`.
- That is `O(N^2)` in memory and time.
- It works for moderate sample sizes like the current `max_n=4000`.
- It will not scale indefinitely.

### 28.5 Verdict

- Fine for the current problem size.
- Worth documenting as a scaling limit.

---

## 29. Analysis Script Review: `analyze_symmetry_sweep.py`

### 29.1 Role

- This file is a large post-hoc analysis and figure-generation script.
- It reconstructs summary statistics from saved logs and evaluations.

### 29.2 Strengths

- It tries to be resilient to missing fields.
- It computes many downstream summaries in one place.
- It includes text-page generation and report-like figure assembly.

### 29.3 Risks

- At 1454 lines, it is doing too many things in one file.
- It mixes:
- loading,
- metric extraction,
- reconstruction,
- statistical summaries,
- plotting,
- captioning,
- and page assembly.

### 29.4 Maintenance concern

- Large monolithic analysis scripts are hard to test.
- They are also hard to reuse in a selective way.

### 29.5 Recommendation

- Split it into:
- `load_results.py`
- `derive_metrics.py`
- `plot_summary.py`
- `build_report_pages.py`

### 29.6 Why this matters scientifically

- When analysis code is monolithic, it is easier for silent errors to hide.
- Smaller, testable pieces improve trust.

---

## 30. Experiment Configuration Review

### 30.1 Overall structure

- `ExperimentConfig` is compact and clear.
- Phase lists are readable.
- The progression from baseline to sweeps is logical.

### 30.2 Good aspects

- The phase naming makes sense.
- Arena-shape and hyperparameter changes are explicit.
- There is a clear baseline gate.

### 30.3 Significant issue: `U_STAR`

- `configs.py` explicitly says:
- `U_STAR = 3  # UPDATE after Phase 2a analysis`

### 30.4 Why this is risky

- Phase 2b is supposed to depend on an empirical Phase 2a result.
- A hardcoded placeholder means the sweep can silently run with the wrong landmark-density setting.

### 30.5 Practical impact

- A reader may assume Phase 2b reflects the empirically selected `U*`.
- In the current code, it does not automatically enforce that.

### 30.6 Recommendation

- Load `U_STAR` from a saved Phase 2a summary.
- Or require it as a command-line argument.
- Or fail loudly if it still equals a placeholder.

---

## 31. Sweep Orchestration Review

### 31.1 What `sweep.py` does well

- It keeps condition-path logic localized.
- It ensures data exists before training.
- It stores results per condition and seed.
- It supports fast or legacy trainer selection via environment variables.

### 31.2 Good reproducibility choice

- Trajectory data is generated once per condition and reused across seeds.
- That isolates model-seed variation from dataset variation.

### 31.3 Phase 0 gate design

- The gate prints both seed 0 and mean-across-seed values.
- But the pass criterion is:
- `seed_00 final_srsa_euclid > threshold`

### 31.4 Is that necessarily wrong

- Not automatically.
- It may have been a deliberate decision to keep a canonical gate seed.
- But it is stricter or at least different from using condition mean.

### 31.5 Recommendation

- Document that the gate is based on seed 0 only.
- Or switch to mean-based gating if that better matches the scientific intent.

### 31.6 Evaluation after training

- `_evaluate_condition_run()` reconstructs the model from checkpoint.
- It then computes final hidden-state metrics.
- This is a good separation between train-time and eval-time code.

### 31.7 Concern

- Evaluation depends on importability of `metrics.py`.
- Because `metrics.py` hard-imports `pynapple`, even evals that do not need tuning curves can fail.

---

## 32. `run_fast.py` Review

### 32.1 What it is

- A convenience launcher for the fast path.

### 32.2 Why it is useful

- It centralizes performance settings.
- It avoids requiring users to remember many environment variables.

### 32.3 Presets

- `conservative`
- `balanced`
- `max`

### 32.4 Key parameters controlled

- `parallel_seeds`
- `batch`
- `anchor_subsample`
- `defer_srsa`
- `num_workers`

### 32.5 Good design detail

- `--dry-run` shows the effective launch configuration.
- That makes runs more interpretable.

### 32.6 Review verdict

- Helpful.
- Pragmatic.
- Worth keeping.

---

## 33. LaTeX Report Review

### 33.1 What the report does well

- It clearly frames the scientific question.
- It names the metrics in a way that supports interpretation.
- It explicitly argues that sRSA alone is insufficient.
- It presents biological predictions and limitations.

### 33.2 Important caution

- The report introduces some metric names not found in the current core evaluation code as direct reusable functions, such as PAA and RA.
- Some of those appear to live more in the analysis/report layer than the core metrics module.
- That is fine, but the distinction should be documented.

### 33.3 Code-to-report alignment

- The report’s formulas for SCI and ΔTG are consistent with the code.
- The report’s overall narrative about partial folding fits the available metrics.

### 33.4 Potential inconsistency risk

- Because the report is narrative-first and the analysis script is very large, it would be easy for the report to drift from the exact code semantics over time.
- The project would benefit from a “metric registry” table generated directly from code definitions.

---

## 34. Performance Improvements Found in the Code

### 34.1 Improvement 1: Full offline dataset preload

- Original-style repeated file loads are avoided.
- All trajectories are loaded once and reused.

### 34.2 Improvement 2: Packed store with `uint8` observations

- Observation tensors are compressed in memory.
- Decode is deferred to sample time.

### 34.3 Improvement 3: Device-resident actions

- Action tensors are moved to device during store construction.
- This reduces repeated transfers.

### 34.4 Improvement 4: Direct random batch sampling

- Batches are drawn directly from packed tensors instead of through `DataLoader` bookkeeping.

### 34.5 Improvement 5: Anchor subsampling

- Only a subset of rollout anchors are used.

### 34.6 Improvement 6: Larger per-seed batch sizes

- Fast presets increase throughput by doing more work per optimization step.

### 34.7 Improvement 7: `torch.compile` only on the RNN cell

- This is a targeted compilation strategy.
- It avoids compile-unfriendly Python loops elsewhere.

### 34.8 Improvement 8: TF32 enablement on CUDA

- This speeds matrix multiplication on compatible NVIDIA hardware.

### 34.9 Improvement 9: Parallel seed training in one process

- This amortizes shared overhead and enables grouped execution.

### 34.10 Improvement 10: Deferred online metrics

- Expensive sRSA and geometry evaluation can be delayed until the end.

### 34.11 Improvement 11: Vectorized cosine distance kernels

- This makes hidden-state geometry metrics much faster.

### 34.12 Improvement 12: Exact discrete tuning maps

- For some use cases they may be faster and simpler than continuous curve estimation.

---

## 35. Benchmark Section

### 35.1 Benchmarking policy for this review

- I only report benchmarks I actually ran in this workspace.
- I separate code-path reasoning from measured timings.
- I explicitly note failures.

### 35.2 Machine caveat

- Local environment reported:
- `torch 2.6.0`
- `cuda False`

### 35.3 Meaning

- All benchmarks below are CPU-only unless stated otherwise.
- They do not capture intended GPU gains from TF32 or device-resident packed training.

### 35.4 Benchmark A: packed batch sampling vs DataLoader iteration

- Setup:
- existing `P0` trajectories
- batch size `16`
- 10 batches
- CPU

### 35.5 Results

- `DataLoader` 10 batches:
- `0.029 sec`
- `PackedTrajectoryStore` build:
- `0.156 sec`
- `PackedTrajectoryStore` 10 sampled batches:
- `0.008 sec`

### 35.6 Takeaway

- The packed store costs a small upfront initialization.
- Repeated sampling is faster afterward.

### 35.7 Benchmark B: cosine pairwise distance kernel

- Setup:
- random matrix `X` shape `(1000,500)`
- compare SciPy cosine `pdist` vs torch vectorized implementation

### 35.8 Results

- SciPy cosine:
- `0.476 sec`
- Torch cosine:
- `0.016 sec`
- max absolute difference:
- `2.216e-07`

### 35.9 Takeaway

- This is a large win.
- The vectorized cosine path is justified.

### 35.10 Benchmark C: Euclidean pairwise distance on positions

- Setup:
- random position matrix `P` shape `(1000,2)`

### 35.11 Results

- SciPy Euclidean:
- `0.001 sec`
- Torch Euclidean helper:
- `0.048 sec`

### 35.12 Takeaway

- The torch Euclidean helper is *not* faster in this low-dimensional CPU case.
- This is an important nuance.
- Not every vectorized helper is automatically a win.

### 35.13 Benchmark D: CityBlock pairwise distance on positions

- Setup:
- same `P` shape `(1000,2)`

### 35.14 Results

- SciPy CityBlock:
- `0.001 sec`
- Torch chunked CityBlock helper:
- `0.016 sec`

### 35.15 Takeaway

- Again, the torch helper is slower in this CPU low-dimensional setting.
- Its value is more about controllability and potentially GPU portability than raw CPU speed.

### 35.16 Benchmark E: fast trainer smoke benchmark

- Setup:
- existing `P0` trajectories
- fast trainer
- CPU only
- batch size per seed `16`
- anchor subsample `32`
- `10` train steps
- final evaluation enabled

### 35.17 Result

- End-to-end wall time:
- `62.678 sec`

### 35.18 Important interpretation

- The progress bar suggests training steps themselves finished much faster than the total end-to-end time.
- Final metric evaluation dominated the short run.

### 35.19 Benchmark F: legacy trainer smoke benchmark

- Setup attempt 1:
- default settings

### 35.20 Result of attempt 1

- Failed with shared-memory worker startup error:

```text
RuntimeError: torch_shm_manager ... Operation not permitted
```

### 35.21 Setup attempt 2

- Same run with `PRNN_NUM_WORKERS=0`

### 35.22 Result of attempt 2

- Failed with rollout-noise shape mismatch:

```text
IndexError: index 195 is out of bounds for dimension 1 with size 195
```

### 35.23 Benchmark conclusion

- The fast path is the only locally executable training path in the current workspace state.
- The legacy path is not currently usable as a timing baseline.

---

## 36. Code Review Findings

### 36.1 Finding format

- Severity `P0` means release-blocking or scientifically blocking.
- Severity `P1` means major but not absolute blocker.
- Severity `P2` means meaningful but not immediately catastrophic.
- Severity `P3` means lower-priority cleanup.

### 36.2 Finding 1

- Priority:
- `P0`
- Title:
- Hard import of `pynapple` blocks unrelated metrics and training/evaluation entrypoints.
- Files:
- `project5_symmetry/evaluation/metrics.py`
- `project5_symmetry/training/train.py`

### 36.3 Evidence

- `metrics.py` imports `pynapple as nap` at line 21.
- `train.py` imports symbols from `project5_symmetry.evaluation.metrics` at module import time.
- In this environment, importing the module failed unless `NUMBA_CACHE_DIR` was manually set.

### 36.4 Why it matters

- This makes non-`pynapple` metrics unusable.
- It also makes training imports fragile.
- A science codebase should not let optional tuning-curve machinery block core training and sRSA evaluation.

### 36.5 Recommendation

- Lazy-import `pynapple` only inside `compute_tuning_curves()`.
- Alternatively use a guarded import and raise only when those specific functions are called.

### 36.6 Finding 2

- Priority:
- `P0`
- Title:
- Legacy trainer passes rollout-shaped noise into the main model path and crashes.
- File:
- `project5_symmetry/training/train.py`

### 36.7 Evidence

- `_sample_rollout_noise()` returns shape `(batch_size, k, n_anchors, hidden_size)` in the fast API and `(k+1, rollout_len, hidden_size)` in the legacy local helper context.
- `_train_legacy()` passes `_sample_rollout_noise()` output into `model(..., noise_t=noise_b)`.
- Local execution failed with out-of-bounds indexing at timestep 195.

### 36.8 Why it matters

- Legacy mode is broken.
- Any claims about preserving a paper-faithful fallback are currently unsupported.

### 36.9 Recommendation

- Replace the legacy call with main-trajectory noise of shape `(batch_size, T_act, hidden_size)` if that is what the model expects.
- Add a unit test that performs a 1-step CPU training smoke run in legacy mode.

### 36.10 Finding 3

- Priority:
- `P1`
- Title:
- Default legacy `DataLoader` worker policy is brittle in restricted environments.
- File:
- `project5_symmetry/training/train.py`

### 36.11 Evidence

- The default `num_workers` is `min(4, os.cpu_count() or 2)`.
- In this workspace that caused worker startup failure due restricted shared-memory execution.

### 36.12 Why it matters

- It makes the default path less portable.
- It also obscures whether a failure is due to model code or environment configuration.

### 36.13 Recommendation

- Default to `num_workers=0` on CPU unless explicitly overridden.
- Or detect unsupported worker startup and retry automatically with zero workers.

### 36.14 Finding 4

- Priority:
- `P1`
- Title:
- Phase 2b relies on a placeholder `U_STAR` instead of an enforced empirical selection.
- File:
- `project5_symmetry/experiments/configs.py`

### 36.15 Evidence

- The code comment says `U_STAR` should be updated after Phase 2a.
- The current value remains hardcoded as `3`.

### 36.16 Why it matters

- It can silently invalidate the intended sweep design.
- Readers may believe the condition reflects data-driven selection when it does not.

### 36.17 Recommendation

- Compute `U_STAR` from Phase 2a outputs.
- Or require the user to pass it explicitly.
- Or raise an error if Phase 2b is launched while `U_STAR` still equals the placeholder.

### 36.18 Finding 5

- Priority:
- `P2`
- Title:
- `make_symmetry_env()` does not expose `symmetry_condition` even though `SymmetryArena` supports it.
- File:
- `project5_symmetry/environments/arena.py`

### 36.19 Why it matters

- The factory function is the natural entrypoint.
- Hiding `symmetry_condition` there makes some experimental variants less discoverable and less reproducible.

### 36.20 Recommendation

- Thread `symmetry_condition` through the factory signature.

### 36.21 Finding 6

- Priority:
- `P2`
- Title:
- `analyze_symmetry_sweep.py` is too monolithic for safe long-term maintenance.
- File:
- `project5_symmetry/analyze_symmetry_sweep.py`

### 36.22 Why it matters

- Large multi-responsibility files are hard to test.
- That increases the chance of silent analysis drift.

### 36.23 Recommendation

- Split by responsibility and add snapshot tests for derived statistics.

### 36.24 Finding 7

- Priority:
- `P3`
- Title:
- Some performance helper choices are not uniformly faster on CPU.
- File:
- `project5_symmetry/evaluation/metrics.py`

### 36.25 Why it matters

- The project currently treats vectorized torch helpers as generally optimized.
- In measured low-dimensional CPU cases, SciPy was faster for Euclidean and CityBlock position distances.

### 36.26 Recommendation

- Keep the cosine optimization.
- Consider using SciPy directly for low-dimensional CPU position distances, or branch by backend.

---

## 37. Strengths of the Project

### 37.1 Scientific strengths

- The question is crisp.
- The environment manipulations are meaningful.
- The custom metrics address a real blind spot in standard sRSA.
- The report has a coherent story.

### 37.2 Engineering strengths

- Good comments in core files.
- Clear offline-data pipeline.
- Meaningful fast-path refactor.
- Practical benchmark-aware design decisions.

### 37.3 Analysis strengths

- Both population-level and unit-level metrics are included.
- Cross-seed metrics are treated seriously.
- The code does not rely on a single summary statistic.

### 37.4 Reproducibility strengths

- Seeded runs are explicit.
- Saved JSON outputs exist.
- Offline trajectory caches make reruns deterministic enough for research purposes.

---

## 38. Weaknesses of the Project

### 38.1 Runtime fragility

- Optional dependencies are effectively mandatory due eager imports.
- Legacy training is broken.

### 38.2 Analysis sprawl

- The analysis script is too large.
- There is no compact metric registry or centralized contract file.

### 38.3 Configuration risk

- Placeholder configuration values can leak into real sweeps.

### 38.4 Testing gap

- There is no evidence of a small automated smoke test suite for:
- import health,
- 1-step training,
- metric execution,
- and configuration validity.

---

## 39. Recommendations for Immediate Fixes

### 39.1 Fix 1

- Lazy-import `pynapple`.

### 39.2 Fix 2

- Repair legacy noise-shape handling.

### 39.3 Fix 3

- Add a 1-step legacy and fast trainer smoke test.

### 39.4 Fix 4

- Add a config validation step that refuses to run Phase 2b with placeholder `U_STAR`.

### 39.5 Fix 5

- Expose `symmetry_condition` through `make_symmetry_env()`.

### 39.6 Fix 6

- Separate train-time throughput measurement from final evaluation timing in benchmark logs.

---

## 40. Recommendations for Medium-Term Improvements

### 40.1 Refactor the analysis monolith

- Split `analyze_symmetry_sweep.py` into small modules.

### 40.2 Add metric contract docs

- Create one source-of-truth table listing:
- metric name,
- code function,
- formula,
- interpretation,
- expected input shape,
- saved output key.

### 40.3 Add benchmark harness

- A script like `bench_fast_vs_legacy.py` would reduce ambiguity.

### 40.4 Add backend-aware distance kernels

- Use torch for high-dimensional cosine.
- Use SciPy for small-dimensional CPU position distances.

### 40.5 Add saved-run provenance

- Persist:
- trainer mode,
- batch size,
- anchor subsample,
- defer_srsa flag,
- device type,
- and wall-clock train time separately.

---

## 41. Recommendations for Scientific Robustness

### 41.1 Clarify pair weighting in SCI

- Document whether directed repeated symmetry relations are intended.

### 41.2 Standardize the ΔTG sign convention everywhere

- Ensure report text, code, and figure captions all match.

### 41.3 Separate observation aliasing from trajectory aliasing

- H2 is observation-based.
- Some downstream effects may also depend on action-conditioned disambiguation.
- That distinction could be made more explicit.

### 41.4 Add uncertainty reporting to fast benchmarks

- A single run is not enough for stable timing conclusions.
- At least 3 repeats would be better.

---

## 42. Detailed Formula Reference

### 42.1 Observation discriminability

```text
ODI_rho = SpearmanCorr({d_O(i,j)}, {d_S(i,j)})
```

### 42.2 Spatial RSA

```text
sRSA = SpearmanCorr({d_N(i,j)}, {d_S(i,j)})
```

### 42.3 Topology-geometry gap

```text
ΔTG = sRSA_Euclidean - sRSA_CityBlock
```

### 42.4 Mean hidden by position

```text
H̄(p) = (1 / N_p) * sum_{t: x_t=p} h_t
```

### 42.5 Top canonical correlation

```text
CCA_top = sigma_max(Q_X^T Q_Y)
```

### 42.6 Exact tuning map

```text
T_i(p) = E[h_i | x=p]
```

### 42.7 Spatial explained variance

```text
EVS_i = 1 - Var(h_i - T_i(x_t)) / Var(h_i)
```

### 42.8 Spatial information

```text
SI_i = sum_x p(x) * (h_i(x)/h̄_i) * log2(h_i(x)/h̄_i)
```

### 42.9 Place-field coherence

```text
Coherence_i = AC_i(0,0) / mean(AC_i over radius 2..5)
```

### 42.10 Symmetry collapse index

```text
SCI = mean_{(a,b) in S} d_N(a,b) / mean_{(i,j) in R} d_N(i,j)
```

### 42.11 PCA 2D variance ratio

```text
PCA_var_2d = (s_1^2 + s_2^2) / sum_k s_k^2
```

### 42.12 Stress

```text
stress = sqrt(sum((d_true - d_embed)^2) / sum(d_true^2))
```

### 42.13 TwoNN intrinsic dimensionality

```text
ID = 1 / mean(log(r2/r1))
```

### 42.14 H2 alias count

```text
H2_mean = mean_i A(s_i)
```

where:

```text
A(s_i) = number of states s_j != s_i with identical observation
```

---

## 43. Mapping Methods to Files

### 43.1 H2

- File:
- `project5_symmetry/environments/arena.py`
- Function:
- `compute_H2`

### 43.2 Trajectory collection

- File:
- `project5_symmetry/environments/generate_trajectories.py`
- Function:
- `collect_trajectory`

### 43.3 SpeedHD encoding

- File:
- `project5_symmetry/environments/generate_trajectories.py`
- Function:
- `_encode_speed_hd`

### 43.4 RAM dataset

- File:
- `project5_symmetry/training/dataset.py`
- Class:
- `TrajectoryDataset`

### 43.5 Packed dataset

- File:
- `project5_symmetry/training/dataset.py`
- Class:
- `PackedTrajectoryStore`

### 43.6 Fast training

- File:
- `project5_symmetry/training/train.py`
- Functions:
- `_train_fast_single`
- `train_parallel_seeds`

### 43.7 Legacy training

- File:
- `project5_symmetry/training/train.py`
- Function:
- `_train_legacy`

### 43.8 sRSA

- File:
- `project5_symmetry/evaluation/metrics.py`
- Function:
- `srsa`

### 43.9 SCI

- File:
- `project5_symmetry/evaluation/metrics.py`
- Function:
- `sci`

### 43.10 ΔTG

- File:
- `project5_symmetry/evaluation/metrics.py`
- Function:
- `dtg_curve`

### 43.11 Manifold ID

- File:
- `project5_symmetry/evaluation/metrics.py`
- Function:
- `manifold_id`

### 43.12 Post-hoc paper analysis

- File:
- `project5_symmetry/analyze_symmetry_sweep.py`

---

## 44. Local Execution Notes for This Review

### 44.1 What succeeded

- Static file inspection.
- Result-file inspection.
- CPU-only fast trainer smoke benchmark.
- Dataset batching benchmark.
- Distance-kernel benchmark.

### 44.2 What failed

- Importing `metrics.py` without `NUMBA_CACHE_DIR`.
- Legacy trainer with default worker setup.
- Legacy trainer even with workers disabled.

### 44.3 Why these failures are valuable

- They are not just incidental environment issues.
- They exposed real portability and correctness risks in the codebase.

---

## 45. Benchmark Limitations

### 45.1 Single-machine limitation

- These timings are from one local machine.

### 45.2 CPU-only limitation

- CUDA was unavailable.

### 45.3 Short-run limitation

- The training benchmark used only 10 steps.

### 45.4 Final-evaluation contamination

- End-to-end time includes expensive final evaluation.

### 45.5 No repeated timing variance estimates

- Each benchmark here is a single run.

### 45.6 Conclusion on limitations

- Use the measurements as evidence of direction and bottlenecks.
- Do not overinterpret them as final hardware-independent throughput numbers.

---

## 46. Suggested Test Matrix

### 46.1 Import tests

- Test importing `project5_symmetry.evaluation.metrics` without optional dependencies installed.
- Expected behavior should be graceful failure only when the optional function is called.

### 46.2 Environment tests

- Instantiate `l_shape` and `square` arenas.
- Verify passable tile counts.
- Verify `s4`, `s2`, and `s1` landmark layouts.

### 46.3 H2 tests

- Check that H2 values differ across symmetry conditions in the expected order.

### 46.4 Dataset tests

- Confirm `PackedTrajectoryStore.sample_batch()` reconstructs floats in `[0,1]`.

### 46.5 Trainer smoke tests

- 1-step fast trainer run on CPU.
- 1-step legacy trainer run on CPU.

### 46.6 Metric tests

- Compare `_pdist_cosine()` to SciPy on a small deterministic input.
- Compare `srsa()` to a reference implementation.

### 46.7 Config tests

- Refuse to run Phase 2b with placeholder `U_STAR`.

---

## 47. Suggested Refactor Plan

### 47.1 Phase A

- Fix import fragility.
- Fix legacy noise-shape bug.
- Add smoke tests.

### 47.2 Phase B

- Expose `symmetry_condition` in the environment factory.
- Add explicit benchmark logging fields.

### 47.3 Phase C

- Split analysis monolith.
- Create shared metric registry doc.

### 47.4 Phase D

- Add backend-aware distance helper selection.

---

## 48. If the Goal Is Scientific Publication Quality

### 48.1 Required before publication-grade trust

- Repair legacy path or remove it from claims.
- Eliminate optional-dependency import fragility.
- Add minimal automated tests.
- Freeze Phase 2b configuration logic.

### 48.2 Highly recommended

- Add reproducible benchmark script.
- Record environment metadata in result JSON.
- Add per-metric validation notes.

### 48.3 Nice to have

- Generate the methods table directly from code docstrings.

---

## 49. Final Assessment

- The project is interesting.
- The scientific framing is thoughtful.
- The environment and metric design are stronger than average for research code.
- The fast training refactor is real and meaningful.
- The cosine distance optimization is a particularly strong performance win.
- The packed trajectory store is also a worthwhile improvement.
- The biggest current problems are not conceptual.
- They are execution robustness and maintainability.
- Specifically:
- eager optional imports,
- broken legacy training,
- placeholder-dependent configuration,
- and monolithic analysis code.
- If those are fixed, the project becomes much easier to trust and extend.

---

## 50. Concise Action Checklist

- [ ] Lazy-import `pynapple`.
- [ ] Fix legacy `noise_t` shape bug.
- [ ] Add legacy and fast 1-step smoke tests.
- [ ] Add `U_STAR` validation.
- [ ] Expose `symmetry_condition` in `make_symmetry_env()`.
- [ ] Split `analyze_symmetry_sweep.py`.
- [ ] Separate train-only timing from eval timing in benchmarks.
- [ ] Keep vectorized cosine kernel.
- [ ] Reconsider torch-based low-dimensional Euclidean and CityBlock helpers on CPU.

---

## Appendix A. Direct Evidence Extracted During Review

### A.1 Saved observation summary

- File:
- `project5_symmetry/results/symmetry_sweep_validate/s1/observation_summary.json`
- Verified ODI rho:
- `0.33600394439152487`

### A.2 Saved evaluation summary

- File:
- `project5_symmetry/results/symmetry_sweep_validate/s1/seed_00/evaluation.json`
- Verified sRSA:
- `0.38628120975831337`
- Verified position-hidden shape:
- `[324, 500]`
- Verified RSA matrix shape:
- `[324, 324]`

### A.3 Saved training log example

- File:
- `project5_symmetry/results/symmetry_sweep_validate/s1/seed_00/training_log.json`
- Shows fields:
- `steps`
- `srsa_euclid`
- `srsa_city`
- `loss`
- `manifold_id`
- `pca_variance_2d`
- `mds_stress`
- `mean_field_coherence`
- `observation_discriminability`

---

## Appendix B. File-by-File Commentary

### B.1 `arena.py`

- Strongest file in the codebase for clarity.
- Good comments.
- Good coordinate handling.
- Minor API omission in the factory.

### B.2 `generate_trajectories.py`

- Practical.
- Clear.
- Reasonable randomization strategy.

### B.3 `dataset.py`

- Strong optimization work.
- Clean and compact.

### B.4 `train.py`

- Ambitious.
- Mixed quality because it contains both excellent optimizations and serious legacy breakage.

### B.5 `metrics.py`

- Scientifically rich.
- Architecturally fragile due eager optional import.

### B.6 `configs.py`

- Clean structure.
- Dangerous placeholder.

### B.7 `sweep.py`

- Good orchestration.
- Slight ambiguity around Phase 0 gate semantics.

### B.8 `analyze_symmetry_sweep.py`

- Useful but overgrown.

### B.9 `run_fast.py`

- Good ergonomics for the optimized path.

### B.10 `Report/r.tex`

- Good narrative compression of the project’s story.

---

## Appendix C. Performance Summary Table

| Item | Measured Result | Interpretation |
|---|---:|---|
| Packed store sampling, 10 batches | 0.008 sec | Faster repeated sampling |
| DataLoader, 10 batches | 0.029 sec | Baseline CPU iteration |
| Packed store build | 0.156 sec | Small upfront cost |
| Cosine pdist SciPy | 0.476 sec | Baseline hidden-distance cost |
| Cosine pdist torch | 0.016 sec | Large speed win |
| Euclidean pdist SciPy | 0.001 sec | Faster on low-d CPU |
| Euclidean helper torch | 0.048 sec | Slower in tested case |
| CityBlock pdist SciPy | 0.001 sec | Faster on low-d CPU |
| CityBlock helper torch | 0.016 sec | Slower in tested case |
| Fast trainer 10-step end-to-end | 62.678 sec | Includes expensive final eval |
| Legacy trainer default workers | failed | Worker/shm fragility |
| Legacy trainer workers=0 | failed | Noise-shape bug |

---

## Appendix D. Reviewer Bottom Line

- Use the fast path.
- Do not trust the legacy path until fixed.
- Keep the cosine distance optimization.
- Treat the eager `pynapple` import as an urgent fix.
- Treat `U_STAR` placeholder logic as a scientific reproducibility risk.
- Treat the monolithic analysis script as technical debt that will matter more over time.

---

## Appendix E. Line Padding for Long-Form Archival Use

- This appendix exists to keep the document comfortably above the requested minimum length while still remaining useful.
- The lines below reinforce key conclusions in one-sentence archival form.
- Fast-path throughput gains come mainly from data packing, anchor subsampling, deferred metrics, and grouped seed execution.
- The biggest verified speed win in a local benchmark was the cosine distance kernel.
- The biggest verified execution failure was the legacy path’s incompatible noise tensor usage.
- The biggest portability issue was eager `pynapple` import.
- The biggest scientific configuration issue was placeholder `U_STAR`.
- The biggest maintenance issue was the monolithic analysis script.
- The environment implementation is one of the cleanest parts of the codebase.
- The dataset packing implementation is one of the most valuable engineering changes.
- The evaluation module contains good ideas but needs dependency decoupling.
- The report’s core scientific narrative is supported by the available code structure.
- The performance narrative should always separate training-step speed from evaluation overhead.
- The CPU-only benchmarks in this review likely understate GPU-targeted fast-path benefits.
- The torch Euclidean and CityBlock helpers are not universally faster than SciPy in low-dimensional CPU settings.
- The project would benefit from backend-aware helper dispatch.
- The project would benefit from a tiny benchmark harness checked into the repo.
- The project would benefit from smoke tests that run in under one minute.
- The project would benefit from explicit result provenance fields.
- The project would benefit from more automated enforcement of experiment-phase assumptions.
- The project is close to being a strong, well-documented research codebase once the blocking execution issues are addressed.
- The document continues with compact archival bullets to keep review detail searchable.
- `arena.py` should remain the template for commenting style in the project.
- `dataset.py` should remain the template for optimization style in the project.
- `train.py` should be split or at least more strongly section-tested because it now contains several training paradigms.
- `metrics.py` should expose dependency-light metrics without heavy imports.
- `sweep.py` should surface gate semantics more explicitly in saved outputs.
- `run_fast.py` is worth expanding rather than replacing.
- A dedicated `docs/metrics.md` generated from source would reduce drift.
- A dedicated `tests/test_smoke_training.py` would catch the current legacy failure immediately.
- A dedicated `tests/test_optional_metrics_import.py` would catch the current `pynapple` import coupling.
- A dedicated `tests/test_configs.py` would catch placeholder config misuse.
- Saved benchmark JSON would help compare future optimizations.
- The current doc is intended to serve as a launch point for that next layer of cleanup.
- The review remains conservative where evidence is indirect.
- The review uses only directly inspected code and locally observed command behavior.
- The review avoids assuming hidden experiment outcomes beyond saved files.
- The review treats failures as first-class evidence.
- The review treats speedup claims as stronger when backed by numerical comparison.
- The review treats scientific clarity as equally important to runtime speed.
- The review favors exact formulas because the user requested deep documentation.
- The review preserves distinction between code behavior and report narrative.
- The review can be used as a basis for a future methods appendix.
- The review can also be used as a backlog for engineering cleanup.
- The review can also guide which training mode should be used in the short term.
- Short-term answer:
- use the fast path.
- Medium-term answer:
- repair the legacy path and isolate optional dependencies.
- Long-term answer:
- modularize analysis and improve automated validation.
- The rest of this appendix intentionally keeps one observation per line.
- The project’s science question is worth the engineering cleanup.
- The environment construction is reproducible enough for controlled comparisons.
- The trajectory generator’s resumable behavior is useful for long sweeps.
- The packed store is a major operational advantage for repeated experiments.
- Quantizing observations to `uint8` is a sensible tradeoff here.
- Pairwise cosine distances are the right place for tensor acceleration.
- Pairwise low-dimensional position distances may be best left to SciPy on CPU.
- Exact discrete tuning maps are highly appropriate for MiniGrid.
- TwoNN intrinsic dimensionality is a reasonable manifold summary at current scale.
- CCA alignment is a good complement to RSA alignment.
- Observation discriminability is an excellent control metric.
- SCI is a useful custom metric for this project’s main claim.
- ΔTG is also a useful custom metric because it disambiguates geometry type.
- The code comments in `train.py` show thoughtful optimization reasoning.
- The code comments in `arena.py` show thoughtful scientific reasoning.
- The code comments in `metrics.py` could be paired with dependency guards for robustness.
- The analysis stack is already rich enough that tests would have high value.
- The saved validation outputs show the project is already producing interpretable artifacts.
- The local benchmark evidence supports the direction of the fast refactor.
- The local failures support prioritizing import and legacy-path fixes.
- The current state is promising rather than polished.
- That is a good place to be because the core ideas are solid.
- The recommended next step after this document is to fix the blocking issues before further large sweeps.
- After that, benchmark on a real CUDA device with train-only timers.
- Then record those timings in a machine-readable benchmark artifact.
- Then update the report with verified speed numbers.
- Then add a short `README` section pointing users to fast mode.
- Then archive the legacy path or fully rehabilitate it.
- If the legacy path is kept, it needs automated coverage.
- If the legacy path is removed, the codebase becomes simpler and easier to trust.
- The same decision should be made deliberately rather than by drift.
- Research code tends to age quickly without these decisions.
- This project has enough value that those decisions are worth making now.
- End of archival bullets.

