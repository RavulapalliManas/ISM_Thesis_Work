# Predictive Learning and Cognitive Map Formation in Recurrent Neural Networks

**Manas Venkata Sai Ravulapalli** — ISM Thesis Work

This repository contains the thesis work investigating how predictive recurrent neural networks learn spatial representations, how those representations generalize across environments of varied geometry and topology, and how landmark symmetry shapes the emergent structure of hippocampal-like codes.

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Repository Organization](#2-repository-organization)
3. [Project 5: Landmark Symmetry and Cognitive Map Formation](#3-project-5-landmark-symmetry-and-cognitive-map-formation) *(thesis focus)*
4. [Project 4: Topology Before Geometry](#4-project-4-topology-before-geometry)
5. [Project 3: Generalization Across 2D and 3D Arenas](#5-project-3-generalization-across-2d-and-3d-arenas)
6. [Original Levenstein Codebase](#6-original-levenstein-codebase)
7. [Shared Dependencies](#7-shared-dependencies)
8. [Hardware Requirements](#8-hardware-requirements)

---

## 1. Scientific Background

### Predictive Learning Framework

This work builds on the framework established by Levenstein et al. (2024): *Sequential predictive learning is a unifying theory for hippocampal representation and replay* ([bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1)). The core idea is that a recurrent network trained to predict future observations must maintain latent spatial state variables that support path integration. Spatial structure — place fields, head-direction tuning, grid-like representations — can emerge spontaneously from this objective without coordinate supervision.

### Successor Representation Interpretation

The Successor Representation (SR) provides the theoretical link between predictive learning and spatial cognition (Stachenfeld et al. 2017). Under the SR interpretation, the neural representation at a position encodes the expected discounted future occupancy of the agent. Formally, the SR matrix **M** has entries `M[i,j]` = expected discounted visits to position *j* starting from position *i*. This means the geometry of the representation reflects the transition structure of the environment — topologically similar environments produce similar SR matrices, enabling generalization.

### Head-Direction Anchoring and the Symmetry Problem

Animals navigate environments where multiple locations generate identical sensory observations. The head-direction (HD) system tracks global orientation through path integration while landmark anchoring corrects drift via visual cues. However, symmetric landmarks force these two mechanisms into direct conflict: the same landmark view appears at multiple rotations, so the HD state and the landmark signal become discordant.

This creates two distinct failure modes:

- **Global degeneracy**: Different training runs settle on incompatible global orientations (the map "flips" between seeds)
- **Local folding**: Symmetric positions within a single run collapse onto each other (multi-field place cells)

The double-rotation literature documents that place cells increase their number of firing fields in symmetric arenas (Muller et al. 1987, Knierim et al. 1995), with multi-field fraction scaling with symmetry order. Yet no computational account existed within the predictive learning framework for how symmetry drives these representational changes.

### Why Standard sRSA Is Insufficient

Seed-to-seed RSA (sRSA) correlates pairwise neural distances with pairwise spatial distances. However, this metric has a critical vulnerability: row permutations of the neural population vector preserve all pairwise distances, so rotated maps yield identical sRSA values even though they represent opposite global orientations. sRSA is therefore **formally blind to both global rotations and local folding**.

Three purpose-built metrics address this gap:

- **PAA (Permutation-Aware Alignment)**: Tests whether a rotated cross-seed alignment improves over identity alignment, directly detecting orientational divergence
- **RA (Rotational Autocorrelation)**: Measures whether individual units respond identically at rotationally equivalent positions, capturing unit-level aliasing
- **C2 Contrast**: Compares neural similarity between 180°-related and 90°-related quadrant pairs, detecting group-theoretic structure in the population code

---

## 2. Repository Organization

```
ISM_Thesis_Work/
│
├── utils/                        # Core predictive RNN framework (shared by all projects)
│   ├── Architectures.py           # pRNN variants: vRNN, thRNN, thetaRNN, Autoencoders
│   ├── thetaRNN.py                # Theta cycling RNN cell implementation
│   ├── predictiveNet.py          # PredictiveNet wrapper class (1244 lines)
│   ├── agent.py                   # RandomActionAgent, RandomHDAgent
│   ├── env.py                     # make_env() — MiniGrid environment factory
│   ├── lossFuns.py                # predMSE, LPLLoss
│   ├── ActionEncodings.py         # OneHotHD, SpeedHD, Velocities, etc.
│   ├── LinearDecoder.py           # Linear decoder for spatial representations
│   ├── figures.py                 # TrainingFigure, SpontTrajectoryFigure
│   ├── general.py                 # General utility functions (saveFig, savePkl, etc.)
│   ├── serialization.py           # Unified pathlib-based save/load (pickle, JSON)
│   ├── data_schema.py             # Trajectory format constants + validation
│   ├── data_store.py              # ResultStore — unified seed-dir loader
│   └── gpu_manager.py             # Standalone interactive GPU experiment manager
│
├── analysis/                      # Offline analysis pipeline (shared by all projects)
│   ├── OfflineTrajectoryAnalysis.py   # Main trajectory analysis (806 lines)
│   ├── SpatialTuningAnalysis.py       # Place field and spatial tuning
│   ├── RepresentationalGeometryAnalysis.py  # RGA (RSA, RDM)
│   ├── DiffusionReplayAnalysis.py     # Diffusion/replay analysis
│   └── ExperienceReplayAnalysis.py   # Experience replay metrics
│
├── tests/                         # Test suite (pytest)
│   ├── conftest.py                # Shared fixtures, repo-root sys.path
│   ├── utils/                     # Tests for serialization, data_schema
│   ├── project5_symmetry/         # Tests for dataset, arena
│   └── reasoning_geometry/        # Tests for data loaders
│
├── FigureScripts/                # 22 Jupyter notebooks for original paper figures
│
├── project5_symmetry/            # THESIS PROJECT — Landmark symmetry study
│
├── project4_topology_before_geometry/  # Supporting project — Topology vs. geometry
│
├── project3_generalization/      # Supporting project — Cross-environment generalization
│
├── reasoning_geometry/           # Exploratory work — Geometric reasoning in RNNs
│
├── configs/                      # Configuration files (visual_rnn example configs)
├── BashScripts_ClusterTraining/  # HPC cluster training scripts
├── nets/                         # Trained network checkpoints and output figures
└── outputs/                      # General results storage
```

### How the Projects Relate

All three projects use the same core predictive RNN architecture from `utils/`. Each project extends the framework in a different direction:

| Project | Scientific Question | Environment Backend | Primary Metric |
|---------|-------------------|-------------------|----------------|
| **Project 5** (thesis) | How does landmark symmetry affect map structure? | MiniGrid (L-shape, square) | sRSA, RA, PAA, C2 Contrast |
| **Project 4** | Do topological representations emerge before geometric ones? | MiniGrid + RatInABox | T_topology, T_geometry, gap |
| **Project 3** | How does learning transfer across environments of varied geometry? | RatInABox + MiniGrid + 3D | sRSA, SR error, transfer efficiency |

---

## 3. Project 5: Landmark Symmetry and Cognitive Map Formation

> **Primary thesis project.** Investigates whether landmark symmetry causes global map degeneracy, local representational folding, or both, in predictive recurrent networks.

**Report:** `project5_symmetry/Report/r_fixed.tex` (compiled PDF: `project5_symmetry/Report/r_fixed.pdf`)
Also at root: `report_revised.tex` (older version)

### 3.1 Scientific Question

Animals navigate environments where multiple locations generate identical sensory observations. The head-direction system tracks global orientation through path integration while landmark anchoring corrects drift via visual cues. When landmarks are rotationally symmetric, these two mechanisms enter direct conflict: the same landmark view appears at multiple rotations, so the HD state and the landmark signal become discordant.

The study asks: does landmark symmetry cause **global map degeneracy** (different seeds learn incompatible global orientations), **local representational folding** (symmetric positions collapse within a single run), or **both**? A standard metric (seed-to-seed RSA) cannot answer this question because row permutations preserve pairwise distances.

### 3.2 Key Findings

1. **Head-direction input prevents global degeneracy.** Even in C4-symmetric arenas, oriented transition statistics (from head-direction-tagged actions) anchor the global frame. Seeds do not settle on incompatible orientations.

2. **Local precision degrades monotonically with symmetry order.** RA scales as S4 (0.223) > S2 (0.116) > S1 (0.098), confirming unit-level aliasing increases with symmetry.

3. **Multi-field place cells emerge naturally in symmetric arenas.** The fraction of units with multiple spatially-tuned fields increases with symmetry order, consistent with the double-rotation literature.

4. **Standard sRSA is blind to map distortion under observation aliasing.** Both global rotation and local folding yield identical sRSA values, confirming the need for purpose-built metrics.

5. **Distance geometry shifts toward path-dependence under high symmetry.** DTG (delta topology-geometry gap) becomes more negative in S4 (DTG_S4 = −0.052) than S1 (DTG_S1 = −0.020), indicating the neural geometry increasingly reflects action history rather than pure spatial proximity.

6. **C2 symmetry group structure is encoded in the population code.** The C2 Contrast metric detects statistically significant encoding of the symmetry group relation, confirming the network distinguishes 180°-related positions.

### 3.3 Arena and Model Configuration

| Parameter | Value |
|-----------|-------|
| Arena | 18×18 MiniGrid, L-shape (cut bottom-right quadrant) or square |
| Symmetry conditions | S4 (C4), S2 (C2), S1 (asymmetric) |
| Observational Discriminability (ODI) | 0.336 (held fixed across conditions) |
| View field | F=7 (7×7×3 egocentric RGB patches) |
| Head-direction input | 12-bin one-hot |
| Model | NormReLU pRNN, N=500 |
| Rollout horizon | k=5 steps |
| Training steps | 80,000 |
| Optimizer | RMSProp (α=0.95, ε=1e-6), LR=2e-3 |
| Seeds | S4: n=5, S2: n=3, S1: n=3 |

### 3.4 Evaluation Metrics

| Metric | Full Name | What It Measures | Normal Range |
|--------|-----------|-----------------|--------------|
| `sRSA` | Spatial RSA | Linear decodeability of position from neural distances (Euclidean and CityBlock spatial distances) | > 0.40 for stable geometry |
| `RA` | Rotational Autocorrelation | Unit-level aliasing at symmetric positions (r between unit's patterns at rotationally equivalent positions) | 0 (no aliasing) to 1 (full aliasing) |
| `PAA` | Permutation-Aware Alignment | Cross-seed global orientational divergence (rotated vs. identity RSA alignment gain) | < 0.05 = no degeneracy |
| `C2 Contrast` | C2 Symmetry Contrast | Population-level encoding of 180°-related vs. 90°-related position pairs | Positive = encodes C2 structure |
| `SCI` | Symmetry Collapse Index | Mean neural distance of symmetric pairs / mean neural distance of random pairs | 1 = fully collapsed; > 1 = preserved |
| `DTG` | Delta Topology-Geometry Gap | sRSA_Euclidean − sRSA_CityBlock over training | Positive = Euclidean geometry dominant |
| `manifold_id` | Two-NN Manifold ID | Intrinsic dimensionality of the neural manifold via TwoNN estimator | ~2-4 for spatial representations |
| `decode_error` | Decoding Error | Ridge regression error from hidden states to 2D position | Lower = better position decodability |
| `frac_tuned` | Fraction Spatially Tuned | Fraction of units with significant spatial selectivity (spatial info > threshold) | 0 to 1 |
| `EVS` | Explained Variance Spatial | Fraction of unit variance explained by spatial position | 0 to 1 |

### 3.5 Experiment Phases

| Phase | Description | Key Variable |
|-------|-------------|--------------|
| **PHASE0** | Baseline gate | L-shape 18×18, F=7, U=3, k=5, T=200. Must reach sRSA > 0.40 for seed 0 before proceeding |
| **PHASE1** | Arena scaling | L-shape + 4 square sizes (12, 18, 24, 30) to test size invariance |
| **PHASE2A** | Landmark density sweep | U=0,1,2,3,4 (number of landmark classes), square 18×18 |
| **PHASE2B** | View size sweep | F=3,5,7, fixed U=U* |
| **PHASE4A** | Rollout horizon sweep | k=1,3,5 × L-shape/square |
| **PHASE4B** | Sequence length sweep | T=50,200,600 × L-shape/square |

### 3.6 Quick-Start Commands

```bash
# Set PYTHONPATH to include project root
set PYTHONPATH=.

# ── Training ──────────────────────────────────────────────────────────────────

# Phase 0: Baseline gate (must pass sRSA > 0.40 for seed 0)
python -m project5_symmetry.run_fast --phase 0

# Or use the full sweep launcher:
python project5_symmetry/experiments/run.py --phase 0

# Resume Phase 0 from step 40k → 80k (validates loss before committing)
python project5_symmetry/training/resume_p0.py

# Run full sweep (all conditions, multiple seeds)
python -m project5_symmetry.run_fast --phase all

# ── Visualization ────────────────────────────────────────────────────────────
# Render arenas, landmarks, agent POV, H2 aliasing heatmaps
python project5_symmetry/visualize_environments.py

# ── Analysis ─────────────────────────────────────────────────────────────────
# Part 1: Load data, compute metrics → master_metrics.csv
python project5_symmetry/full_analysis_part1.py

# Part 2: Main publication figures (Fig 1–6)
python project5_symmetry/full_analysis_part2.py

# Part 3: Supplementary figures (Fig S1–S6)
python project5_symmetry/full_analysis_part3_supp.py

# Standalone analysis + PDF report
python project5_symmetry/analyze_symmetry_sweep.py
```

### 3.7 Project Structure

```
project5_symmetry/
├── run_fast.py                      # Main training launcher (LaunchPreset presets)
├── experiments/
│   ├── configs.py                   # SymmetryExperimentConfig dataclass, phase definitions
│   ├── sweep.py                     # Outer sweep loop, gate, evaluation
│   ├── run.py                       # Entry-point launcher (delegates to sweep)
│   ├── run_sweep.py                 # Standalone symmetry sweep runner
│   └── run_hd_ablation.py           # Head-direction ablation experiment
├── analysis/
│   └── pipeline.py                  # Analysis pipeline (statistics + figures)
├── training/
│   ├── train.py                     # Core training (pRNN_th, RMSProp, 80k steps)
│   ├── dataset.py                   # TrajectoryDataset, PackedTrajectoryStore
│   ├── train_single_run.py          # Single-run forward-pass debug script
│   └── resume_p0.py                 # Resume from step 40k → 80k
├── environments/
│   ├── arena.py                     # SymmetryArena (MiniGrid), PixelObsWrapper, H2 compute
│   └── generate_trajectories.py     # Offline trajectory collection (multiprocess)
├── evaluation/
│   └── metrics.py                   # sRSA, RA, PAA, C2 Contrast, SCI, DTG, manifold_id, EVS
├── figures/
│   ├── rebuild_figures.py           # Rebuild all manuscript figures
│   ├── plot_rate_maps.py            # Rate map visualization
│   └── plot_training_curves.py      # Training curve visualization
├── full_analysis_part1.py           # Data loading + metric computation → master_metrics.csv
├── full_analysis_part2.py           # Publication figures (Fig 1–6)
├── full_analysis_part3_supp.py      # Supplementary figures (Fig S1–S6)
├── analyze_symmetry_sweep.py        # Standalone analysis + PDF report generator
├── visualize_environments.py        # Arena/landmark/POV/H2 visualization
├── visualize_phase0.py              # Phase 0 arena visualiser
├── visualize_new_arenas.py          # New arena visualiser
└── Report/
    ├── r_fixed.tex                  # Main thesis report (LaTeX source)
    ├── r_fixed.pdf                  # Compiled PDF
    ├── r.tex                        # Original version
    ├── references.bib
    ├── stats_results (3).csv
    └── images/                      # ~90 figure files (PDF, PNG)
```

---

### 3.8 Experiment Pipeline Overview

Project 5 has **two parallel experiment tracks** that serve different scientific
questions. Both use the same model architecture, training loop, and evaluation
metrics, but vary different parameters.

#### Track A — Phase-Based Parameter Sweep

**Use when:** Varying arena size, landmark density, view field, rollout horizon,
or sequence length to study how these parameters affect spatial representations.

**Entry points:**
```bash
# Fast GPU trainer (recommended):
python -m project5_symmetry.run_fast --phase 0

# Full sweep launcher (more options):
python project5_symmetry/experiments/run.py --phase all
```

**Flow:**
```
run.py / run_fast.py
  └─► sweep.py  (iterates over condition × seed)
        ├─► generate_trajectories.py  (writes .npz files)
        ├─► train.py                  (trains pRNN, saves .pt + logs)
        └─► metrics.py                (computes sRSA, SCI, DTG, etc.)
              └─► metrics.json        (per seed)
              └─► all_results_*.json  (aggregated)
```

**Output directory structure:**
```
project5_symmetry/results/
  <condition_id>/                    e.g. P1-B, P2a-U3
    trajectories/                    traj_00000.npz … traj_09999.npz
    arena_meta.json
    seed_00/
      ckpt_final.pt                  trained model weights
      training_log.json              loss + sRSA curves over training steps
      metrics.json                   final evaluation metrics
      tb/                            TensorBoard event files
    seed_01/
      ...
  all_results_phase1.json            aggregated across all seeds in phase
```

**Config source:** `configs.py` — `SymmetryExperimentConfig` dataclass +
pre-built phase lists (PHASE0 through PHASE4B).

---

#### Track B — Symmetry Group Sweep (S4/S2/S1)

**Use when:** Comparing how different levels of landmark symmetry (C4, C2,
asymmetric) affect representational geometry, aliasing, and manifold structure.
This is the primary track for the thesis findings.

**Entry point:**
```bash
python project5_symmetry/experiments/run_sweep.py
```

**Flow:**
```
run_sweep.py
  ├─► Phase 0 validation gate (optional)
  └─► for condition in ['s4', 's2', 's1']:
        ├─► SymmetryArena(symmetry_condition=condition)
        ├─► generate_trajectories()
        ├─► train pRNN
        └─► compute metrics (sRSA, SCI, DTG, manifold_id)
              └─► symmetry_sweep_raw.pkl  (aggregated dict)
```

**Output:**
```
project5_symmetry/results/symmetry_sweep/
  symmetry_sweep_raw.pkl          dict[condition → list[metrics_per_seed]]
  s4/
    seed_00/
      ckpt_final.pt               model weights
      training_log.json           training curves
  s2/
    seed_00/
      ...
  s1/
    seed_00/
      ...
```

**Config source:** Inline in `run_sweep.py` — conditions defined at module top.

---

#### Analysis Pipeline

**Entry point:**
```bash
python project5_symmetry/analysis/pipeline.py
```

The analysis pipeline (`analysis/pipeline.py`) is designed for **Track B**
(symmetry group sweep). It reads `symmetry_sweep_raw.pkl` or per-seed
`metrics.json` files and produces summary statistics + figures.

**Output:**
```
results/symmetry_summary_statistics.csv
results/symmetry_condition_comparisons.csv
figures/srsa_by_symmetry.png
```

---

#### Which Pipeline Should I Use?

| Goal | Use Track |
|------|-----------|
| Reproduce thesis sRSA-by-symmetry figures | **B** — `run_sweep.py`, then `pipeline.py` |
| Vary landmark density or view field | **A** — `run.py --phase 2a` |
| Test arena size scaling | **A** — `run.py --phase 1` |
| Sweep rollout horizon or sequence length | **A** — `run.py --phase 4a/4b` |
| Quick sanity check / gate test | **A** — `run.py --phase 0` or B's `--validate` |

---

## 4. Project 4: Topology Before Geometry

> Investigates whether topological representations (Betti numbers) emerge before geometric representations (spatial metrics like sRSA) in predictive RNNs trained on navigation tasks.

### 4.1 Scientific Question

A fundamental question in spatial cognition is whether the brain first learns the topological structure of the environment (which places are connected, regardless of metric distances) before learning the geometric details (exact distances and angles). This project tests the hypothesis that predictive RNNs exhibit the same bias: topological representations — detectable via persistent homology (Betti numbers) — should converge before geometric ones (detectable via sRSA).

The **gap** metric (T_geometry − T_topology) is the key scientific output: a positive gap supports the hypothesis.

### 4.2 Key Concepts

- **Betti-0**: Number of connected components (should be 1 for connected environments)
- **Betti-1**: Number of loops/cycles (0 for simply-connected, 1 for a single loop, 2 for figure-8, etc.)
- **T_topology**: First training step at which Betti numbers are correct for 3 consecutive evaluations
- **T_geometry**: First training step at which sRSA ≥ 0.4 for 3 consecutive evaluations
- **gap** = T_geometry − T_topology (positive = topology converges first)

### 4.3 Aliasing-Controlled Environment System

The core innovation is a parameterized environment system that independently controls geometry (shape) and aliasing (observation symmetry). The canonical naming scheme:

```
{geometry}_tau={τ}_lambda={λ}_H={H}_omega={ω}
```

| Parameter | Meaning | Range |
|-----------|---------|-------|
| **τ (tau)** | Tile pattern periodicity (observation wavelength) | 0 (no pattern) → high (repeating stripes) |
| **λ (lambda)** | Landmark density | 0 (none) → 1 (maximal) |
| **H (heading)** | Heading reference frame distinctiveness | 0 (all directions identical) → 1 (all distinct) |
| **ω (omega)** | Overall aliasing level | Composite of tau, lambda, H |

**Alias presets**: `zero_alias`, `low_alias`, `medium_alias`, `high_alias`, `maximum_alias`

### 4.4 Environment Taxonomy

**MiniGrid discrete environments** (20+ registered):
- Convex: `square_low_alias`, `square_high_alias`, `rectangle_wide`, `rectangle_narrow`
- Non-convex: `l_shape_standard`, `l_shape_large`, `u_shape_mask`, `t_maze_mask`
- Loops: `two_room_corridor`, `hairpin_maze`, `figure8_mask`, `double_loop_mask`
- Challenging: `maze_simple_mask`, `maze_medium_mask`, `spiral_maze_mask`, `dead_end_maze_mask`
- Aliasing tests: `symmetry_trap`, `long_corridor_alias`, `uniform_box`, `checkerboard_large_period`

**RatInABox continuous environments**:
- `annulus_approx` (with tunable inner_radius) — Betti-1 = 1
- `figure8_env` — Betti-1 = 2
- `cylinder_env` (periodic x) — Betti-1 = 1

**Note**: `maze_medium` and `spiral_maze` produce disconnected traversable regions and should not be used.

### 4.5 Convergence Metrics

| Metric | Full Name | Convergence Criterion |
|--------|-----------|----------------------|
| `betti_correct` | Betti Correct | Betti-0 and Betti-1 match ground truth AND persistence gap ratio > 2.0 |
| `persistence_gap_ratio_dim1` | H1 Gap Ratio | Ratio of dominant H1 lifetime to second-largest H1 lifetime |
| `srsa_euclidean` | sRSA (Euclidean) | ≥ 0.40 for 3 consecutive evaluations |
| `srsa_geodesic` | sRSA (Geodesic) | ≥ 0.40 for 3 consecutive evaluations |
| `T_topology` | Topology Convergence Step | First step with 3 consecutive betti_correct |
| `T_geometry` | Geometry Convergence Step | First step with 3 consecutive srsa_euclidean ≥ 0.40 |
| `gap` | Convergence Gap | T_geometry − T_topology |

### 4.6 Quick-Start Commands

```bash
# ── Local prototyping (CPU, fast) ────────────────────────────────────────────
python project4_topology_before_geometry/scripts/run_local.py

# ── Production training (GPU, RTX 4070 target) ───────────────────────────────
python project4_topology_before_geometry/scripts/run_remote.py

# ── Benchmark sweeps ─────────────────────────────────────────────────────────
python project4_topology_before_geometry/scripts/run_benchmark.py

# ── Verify environments ──────────────────────────────────────────────────────
# Single environment aliasing diagnostics
python project4_topology_before_geometry/scripts/verify_aliasing.py --env l_shape_standard

# Batch validation of all ~49 prebuilt environments
python project4_topology_before_geometry/scripts/batch_verify_aliasing.py

# ── Visualize environments ───────────────────────────────────────────────────
python project4_topology_before_geometry/scripts/visualize_environments.py

# ── Configuration ────────────────────────────────────────────────────────────
# Edit configs/local_config.yaml  (CPU, 5,000 trials, 3 envs)
# Edit configs/remote_config.yaml (GPU, 50,000 updates, torch.compile, bf16)
```

### 4.7 Project Structure

```
project4_topology_before_geometry/
├── models/
│   ├── prnn.py                      # RolloutPRNN wrapper (201 lines)
│   └── objectives.py               # 6 loss functions + LossFactory (165 lines)
├── sensory/
│   ├── action_encoder.py           # MiniGrid (5-D) and RatInABox (13-D) action encoding
│   └── aliasing_control.py         # Tile pattern generation, aliasing scoring (250 lines)
├── environments/
│   ├── base_env.py                  # Abstract BaseTopologyEnv, RolloutBatch (138 lines)
│   ├── minigrid_envs.py             # Paper-faithful discrete MiniGrid envs (263 lines)
│   ├── aliasing_controlled_envs.py  # Full aliasing-controlled system (718 lines)
│   ├── rib_envs.py                  # RatInABox continuous envs (340 lines)
│   ├── env_factory.py               # get_env() factory + list_environments()
│   └── topology_labels.py           # Ground-truth Betti labels for all envs
├── training/
│   └── trainer.py                   # Trainer with AsyncLogger (277 lines)
├── evaluation/
│   ├── convergence_tracker.py       # Central metrics computation (421 lines)
│   ├── topological_metrics.py       # Betti number computation + convergence criteria (139 lines)
│   ├── geometric_metrics.py         # sRSA, decoding, spatial info, EV, PR (292 lines)
│   ├── drift_tracker.py             # RDM delta for representational drift
│   ├── replay_decoder.py            # Sleep replay trajectory analysis
│   └── persistence_analysis.py      # Persistence diagram evolution helpers
├── analysis/
│   └── phase_transition.py         # Convergence + gap plots (137 lines)
├── scripts/
│   ├── run_local.py                # Local prototype entry point (110 lines)
│   ├── run_remote.py               # Production training entry point (168 lines)
│   ├── run_benchmark.py            # Sweep orchestration (110 lines)
│   ├── aliasing_sweep.py           # Sweep definitions (4 sweeps)
│   ├── verify_aliasing.py          # Single-env diagnostics (125 lines)
│   ├── batch_verify_aliasing.py    # Batch validation → batch_summary.csv
│   └── visualize_environments.py  # Render all environments (56 lines)
└── configs/
    ├── local_config.yaml
    ├── remote_config.yaml
    ├── test_config.yaml
    └── test_benchmark_config.yaml
```

---

## 5. Project 3: Generalization Across 2D and 3D Arenas

> Extends the predictive-RNN framework to investigate how spatial representations transfer across environments of varied geometry, topology, and dimensionality.

### 5.1 Scientific Question

How does a predictive RNN's learned representation generalize when placed in novel environments? Specifically: does training in a curriculum of environments with similar topology enable zero-shot transfer to geometrically novel environments? What is the role of the cortical module as a slow-learning prior? How do topological representations (SR matrices) predict representational transfer quality?

### 5.2 Two-Environment Suites

#### 2D Environment Suite (`environments/suite_2d.py`)

18 predefined arenas across 4 categories:

**Category A — Simple convex:**
- A1_square, A2_large_square, A3_circle, A4_rectangle

**Category B — Non-convex / compartmentalized:**
- B1_l_shape, B2_t_maze, B3_hairpin_maze, B4_compartmentalized

**Category C — Annotated / morphing:**
- C1_reward_zone_layout, C2_barrier_with_gap, C3_morph_series

**Category D — Topologically nontrivial:**
- D1_annulus, D2_figure8_style

**Observation modes:**
- `"bvc_hd"`: Boundary Vector Cells + Head Direction Cells (from RatInABox)
- `"visual"`: Egocentric RGB patches from tile-map renderer

#### 3D Environment Scaffold (`environments/suite_3d.py`)

Lightweight 3D framework for exploratory experiments:
- Surface types: `volume`, `lattice`, `tilted_lattice`, `platform`
- Navigators: `SurfaceNavigator3D`, `VolumetricNavigator3D` (OU velocity dynamics)
- Feature generators: `PlaceCells3D` (Gaussian), `HeadDirectionCells3D` (spherical), `BoundaryVectorCells3D`

### 5.3 Model Architecture: Two-Module System

```
HippocampalPredictiveRNN          CorticalRNNPrior (slow-learning prior)
  └─ _MLPObservationAdapter         └─ GRU-based prior over hippocampal h_t
  └─ AMP + gradient checkpointing     └─ Low-rank recurrent decoder
  └─ Layerwise LR scaling (RMSProp)    └─ Methods: encode_sequence, infer_recurrent_matrix
  └─ EWC regularization support        └─ initialize_hippocampus (blends into recurrent matrix)
  └─ Recurrence scaling (ablations)
  └─ Spontaneous replay generation
```

### 5.4 Evaluation Metrics

| Metric | Full Name | Category |
|--------|-----------|----------|
| `BG1_trials_to_criterion` | Trials to Criterion | Behavioral |
| `BG2_zero_shot_accuracy` | Zero-Shot Accuracy | Behavioral |
| `RG1_sRSA` | Spatial RSA | Representational Geometry |
| `RG2_CERA` | Procrustes Alignment Error | Representational Geometry |
| `RG3_CKA` | Centered Kernel Alignment | Representational Geometry |
| `RG4_betti_numbers` | Betti Numbers | Representational Geometry |
| `SG1_SR_error` | SR Error | Successor/Structure |
| `SG2_transfer_vs_similarity` | Transfer vs. SR Similarity | Successor/Structure |
| `SG3_eigenspectrum_overlap` | Eigenspectrum Overlap | Successor/Structure |
| `GG1_elongation_index` | Elongation Index | Geometry/Manifold |
| `GG2_field_size_anisotropy` | Field Size Anisotropy | Geometry/Manifold |
| `GG3_topological_remapping_index` | Topological Remapping Index | Geometry/Manifold |

### 5.5 Training Pipelines

| Pipeline | File | Description |
|----------|------|-------------|
| **Single-environment baseline** | `training/single_env.py` | Train in one environment, evaluate sRSA, spatial tuning, replay quality |
| **Curriculum** | `training/curriculum.py` | Sequential training across environments with greedy similarity ordering; supports EWC |
| **Ablation** | `training/ablations.py` | Recurrence-strength sweep (single_env or curriculum mode) |
| **Two-module** | `training/curriculum.py` | Same as curriculum but with cortical prior enabled |
| **Visual-input** | `visual_rnn/train.py` | Independent pipeline for egocentric RGB patches using TileMap + CNN encoder |

### 5.6 Quick-Start Commands

```bash
# ── Baseline training ───────────────────────────────────────────────────────
python -m project3_generalization.experiments.run_baselines

# ── Curriculum training ──────────────────────────────────────────────────────
python -m project3_generalization.experiments.run_curriculum

# ── Two-module (cortical + hippocampal) ──────────────────────────────────────
python -m project3_generalization.experiments.run_two_module

# ── Ablation sweep ───────────────────────────────────────────────────────────
python -m project3_generalization.experiments.run_ablation

# ── Hardware-constrained runs (most operational) ──────────────────────────────
python -m project3_generalization.experiments.run_hardware_constrained

# ── 3D experiments ──────────────────────────────────────────────────────────
python -m project3_generalization.experiments.run_3d

# ── Visual RNN experiments ───────────────────────────────────────────────────
# (requires project3_generalization installed or PYTHONPATH set)
from project3_generalization.visual_rnn.train import run_single_experiment
```

### 5.7 Project Structure

```
project3_generalization/
├── experiments/
│   ├── run_baselines.py             # Single-environment baseline launcher
│   ├── run_curriculum.py            # Multi-environment curriculum launcher
│   ├── run_two_module.py            # Cortical+hippocampal curriculum launcher
│   ├── run_ablation.py              # Recurrence-strength ablation launcher
│   ├── run_hardware_constrained.py  # Hardware-budgeted runs (most operational)
│   └── run_3d.py                    # 3D simulation launcher
├── training/
│   ├── single_env.py               # Single-environment training loop
│   ├── curriculum.py               # Curriculum + EWC + cortical prior
│   └── ablating.py                  # Ablation harness
├── models/
│   ├── hippocampal_module.py       # HippocampalPredictiveRNN wrapper (AMP, EWC, etc.)
│   └── cortical_module.py          # CorticalRNNPrior (slow-learning prior)
├── environments/
│   ├── suite_2d.py                 # 18 2D arenas + RatInABox/BVC+HD observation
│   ├── suite_3d.py                 # 3D scaffold + navigators + feature generators
│   └── similarity.py              # SR computation + pairwise similarity matrices
├── evaluation/
│   ├── metrics.py                  # All core metrics (BG/RG/SG/GG prefixed)
│   └── topology.py                 # Persistent homology (ripser wrapper)
├── analysis/
│   ├── figures.py                  # Similarity matrix, transfer, learning curves
│   └── stats.py                    # Cohen's d, Pearson r, FDR (Benjamini-Hochberg)
├── visual_rnn/
│   ├── train.py                    # Visual-input single-run pipeline
│   ├── model.py                   # build_visual_model_config() (CNN encoder, hidden=384)
│   ├── renderer.py                # TileMap → egocentric RGB patches
│   └── analysis.py                # Post-run visualization (UMAP, tuning curves, etc.)
├── hardware.py                     # Hardware-aware config dataclasses + PhaseLogger
└── tests/
    └── test1.py                     # Ad hoc environment visualization sanity check
```

---

## 6. Original Levenstein Codebase

The `utils/` and `analysis/` directories form the foundational codebase shared by all three projects. They implement the predictive learning framework from Levenstein et al. (2024).

### 6.1 Core Architecture (`utils/Architectures.py`)

Defines all pRNN variants — over 1271 lines of architecture definitions:

| Architecture | Description |
|-------------|-------------|
| `vRNN_*_win` | Vanilla RNN with various prediction window sizes (0–10) |
| `thRNN_*_win` | Theta-RNN with theta cycling (5-win is paper standard) |
| `AutoencoderPred_*` | Autoencoder variants with predictive heads |
| `thcycRNN_*_win` | Theta-cycling RNN variants |

The **thetaRNN** (`utils/thetaRNN.py`) is the core RNN cell: it implements theta-coupling where the input and recurrent contributions are active on different phases of the theta cycle (every 2 timesteps), matching hippocampal theta rhythm dynamics.

### 6.2 Main Model (`utils/predictiveNet.py`)

The `PredictiveNet` class (1244 lines) wraps architectures and training. Key methods:

- `train()` — Full training loop with RMSProp, LR scaling per layer
- `evaluate()` — Compute sRSA and other metrics
- `spontaneous_activity()` — Generate replay trajectories from spontaneous dynamics

### 6.3 Offline Analysis (`analysis/`)

| Module | Purpose |
|--------|---------|
| `OfflineTrajectoryAnalysis.py` | Main analysis class (806 lines): spatial representations, diffusion fits, transition maps |
| `SpatialTuningAnalysis.py` | Place field computation, spatial information, stability |
| `RepresentationalGeometryAnalysis.py` | RSA, RDM computation (RGA.calculateRSA_space) |
| `DiffusionReplayAnalysis.py` | Replay detection and characterization |
| `ExperienceReplayAnalysis.py` | Experience replay metrics |
| `decodeAnalysis.py` | Decoding from neural activity |
| `RepresentationalConnectivityAnalysis.py` | Connectivity-based analysis |

### 6.4 Figure Scripts (`FigureScripts/`)

22 Jupyter notebooks for reproducing the original paper figures:

- `Figure1_PredictiveLearningSpatialRep.ipynb` — Main Figure 1
- `Figure2_MultiMaskCogMap.ipynb` — Main Figure 2
- `Figure3_Replay.ipynb` — Main Figure 3
- `Figure4_ThetapRNN.ipynb` — Main Figure 4
- `FigureS2.ipynb` through `FigureS21.ipynb` — Supplementary figures

Each notebook loads trained networks from `nets/replicate_fig1/` and generates figures using `analysis.*` modules and `utils.predictiveNet.PredictiveNet`.

---

## 7. Shared Dependencies

All three projects share the core `utils/` and `analysis/` packages. Additional dependencies by functionality:

### Core ML
```
torch
numpy
scipy
matplotlib
scikit-learn
pandas
```

### Environment Simulation
```
minigrid          # or gymnasium (MiniGrid discrete navigation)
ratinabox         # Continuous 2D animal navigation with BVC/HDCell agents
shapely           # Polygon geometry operations
```

### Geometry and Topology
```
ripser            # Persistent homology computation
gudhi             # Topological data analysis (optional)
networkx          # Graph operations for geodesic connectivity
```

### Analysis and Visualization
```
pynapple          # Tuning curve computation (nap.compute_2d_tuning_curves_continuous)
umap-learn        # UMAP for manifold visualization (fallback: PCA)
tensorboard       # Training logs
wandb             # Optional dashboard (Weights & Biases)
tqdm              # Progress bars
statsmodels       # Multiple testing correction (Benjamini-Hochberg)
```

### Training Infrastructure
```
PyYAML            # Configuration files
rich              # Rich terminal output
psutil            # Resource monitoring
```

---

## 8. Hardware Requirements

| Project | Recommended | Minimum | Notes |
|---------|-------------|---------|-------|
| **Project 5** (symmetry) | GPU (8GB+ VRAM) | CPU (prototyping only) | `torch.compile` + CUDA Graphs used for speed; CPU works for debugging |
| **Project 4** (topology) | GPU (RTX 4070 target) | CPU (test_config.yaml) | 50k-step full runs require GPU; async topology worker (ripser) runs on CPU |
| **Project 3** (generalization) | GPU for visual RNN; CPU OK for 2D baselines | CPU for 2D suite | `ratinabox` is CPU-bound for rollout generation; GPU needed for visual CNN+RNN |

### GPU Configuration

Projects 4 and 5 support hardware-aware configurations:

```bash
# Project 4
# Edit: project4_topology_before_geometry/configs/remote_config.yaml
# Key settings: hidden_size=512, batch_size=1, accumulation=32, torch.compile, bf16 mixed precision

# Project 3
# Edit: project3_generalization/hardware.py HardwareConfig
# OOM-adaptive resource back-off: batch_size, sequence_length, hidden state storage
```

### CUDA Requirements

- `torch.compile` used in Projects 4 and 5 (not supported on Windows via `run_remote.py` — `maybe_compile()` checks `sys.platform != 'win32'`)
- CUDA Graphs for speed in Project 5 training
- Multi-GPU: not currently used; single-GPU training is standard

---

## Quick Reference: Entry Points by Project

### Project 5 (Symmetry — thesis focus)

```bash
# Training
python -m project5_symmetry.run_fast --phase 0        # Gate (fast GPU trainer)
python project5_symmetry/experiments/run.py --phase 0 # Gate (full sweep launcher)
python -m project5_symmetry.run_fast --phase all      # Full sweep
python project5_symmetry/experiments/run_hd_ablation.py  # HD ablation experiment

# Analysis
python project5_symmetry/full_analysis_part1.py        # Metrics computation
python project5_symmetry/full_analysis_part2.py        # Main figures
python project5_symmetry/full_analysis_part3_supp.py  # Supplementary figures
python project5_symmetry/analysis/pipeline.py          # Unified analysis pipeline
python project5_symmetry/analyze_symmetry_sweep.py     # Standalone analysis + PDF

# Report
# project5_symmetry/Report/r_fixed.tex + r_fixed.pdf
```

### Project 4 (Topology Before Geometry)

```bash
# Training
python project4_topology_before_geometry/scripts/run_local.py    # Local prototype
python project4_topology_before_geometry/scripts/run_remote.py   # Production GPU

# Environment validation
python project4_topology_before_geometry/scripts/verify_aliasing.py --env l_shape_standard
python project4_topology_before_geometry/scripts/batch_verify_aliasing.py

# Visualization
python project4_topology_before_geometry/scripts/visualize_environments.py
```

### Project 3 (Generalization)

```bash
# Training
python -m project3_generalization.experiments.run_baselines             # Single-env baseline
python -m project3_generalization.experiments.run_curriculum            # Curriculum
python -m project3_generalization.experiments.run_two_module           # Two-module
python -m project3_generalization.experiments.run_ablation             # Ablation sweep
python -m project3_generalization.experiments.run_hardware_constrained  # Resource-budgeted

# Analysis
python -m project3_generalization.experiments.run_3d                    # 3D experiments

# Visual RNN
# from project3_generalization.visual_rnn.train import run_single_experiment
```

### Original Levenstein Reproduction

```bash
# Training (legacy)
python project3_generalization/scripts/train_net.py

# Figure notebooks
# Open FigureScripts/Figure*.ipynb

# Analysis
python project3_generalization/scripts/run_analysis.py
```

---

## 9. Testing

Test suite location: `tests/` (pytest framework).

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/utils/ -v
python -m pytest tests/project5_symmetry/ -v
```

### What's covered

| Test module | What it validates |
|-------------|-------------------|
| `tests/utils/test_serialization.py` | Pickle/JSON roundtrips, numpy handling, auto-mkdir |
| `tests/utils/test_data_schema.py` | Trajectory filename format, key validation, dtype checks |
| `tests/project5_symmetry/test_dataset.py` | TrajectoryDataset loading, PackedTrajectoryStore, error handling |
| `tests/reasoning_geometry/` | *(placeholder)* Data loader tests |

---

*Last updated: May 2026. Main thesis report: `project5_symmetry/Report/r_fixed.tex`*