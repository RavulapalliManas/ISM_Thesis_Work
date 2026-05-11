# ISM_Thesis_Work

This repository contains the thesis work of **Manas Venkata Sai Ravulapalli** on predictive learning and cognitive map formation in recurrent neural networks.

The work spans three inter-related projects investigating how predictive RNNs learn spatial representations, how those representations generalize across environments, and how landmark symmetry shapes the emergent geometry of hippocampal-like codes.

---

## Background

This repository started as a reproduction workspace for:

Levenstein D, Efremov A, Henha Eyono R, Peyrache A, Richards BA. Sequential predictive learning is a unifying theory for hippocampal representation and replay.
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1)

The original codebase contains the predictive RNN architectures, training utilities, analyses, and figure notebooks used for the Levenstein-style single-environment experiments. These remain available under `FigureScripts/` and alongside the new project packages.

---

## Project 5: Landmark Symmetry and Cognitive Map Formation

**Primary thesis project.** This study asks whether landmark symmetry causes global map degeneracy, local representational folding, or both in predictive recurrent networks.

**Key question:** Animals navigate environments where multiple locations generate identical sensory observations. How does rotational symmetry in landmarks affect the emergence and structure of cognitive maps?

**Main findings:**
- Head-direction input anchors global orientation through transition statistics, preventing degeneracy even under high observation symmetry
- Local precision degrades monotonically with symmetry order (RA scales as S4 > S2 > S1)
- Multi-field place cells emerge as a natural consequence of symmetric arenas
- Standard sRSA is blind to map distortion under partial observation symmetry — new metrics (PAA, RA, C2 Contrast) are required

**Report:** `project5_symmetry/Report/r_fixed.tex` (compiled PDF: `project5_symmetry/Report/r_fixed.pdf`)

**Key entry points:**
- `project5_symmetry/experiments/` — experiment scripts
- `project5_symmetry/evaluation/` — metric implementations (PAA, RA, C2 Contrast, SCI)
- `project5_symmetry/analysis/figures.py` — figure generation

---

## Project 3: Generalization Across 2D and 3D Arenas

Extends the predictive-RNN framework to investigate how spatial representations transfer across environments of varied geometry and topology.

### What Was Added

#### 1. 2D environment suite

`project3_generalization/environments/suite_2d.py` — configurable 2D environment library including:
- Symmetric open arenas: square, large square, circle, rectangle
- Non-convex arenas: L-shape, T-maze, hairpin maze, compartmentalized arena
- Functional/landmark arenas: reward-zone layouts, barrier-with-gap, morph series
- Topology-focused arenas: annulus and figure-8 style environment
- RatInABox-based observation generation using boundary-vector and head-direction signals

#### 2. Structural similarity / successor representation pipeline

`project3_generalization/environments/similarity.py` — computes:
- Discretized transition matrices
- Successor representations
- Pairwise structural similarity matrices across environments

#### 3. Predictive model wrappers

- `project3_generalization/models/hippocampal_module.py`
- `project3_generalization/models/cortical_module.py`

Thin wrappers over the existing Levenstein predictive RNN architectures with recurrence scaling support for ablations and a cortical prior module for two-module transfer experiments.

#### 4. Unified evaluation metrics

- `project3_generalization/evaluation/metrics.py`
- `project3_generalization/evaluation/topology.py`

Implementations for sRSA reuse, fraction of spatially tuned cells, participation ratio, replay quality, CERA / CKA, SR error and transfer-vs-similarity summaries, elongation and remapping metrics, and Betti-number / persistent-homology helpers.

#### 5. Training pipelines

- `project3_generalization/training/single_env.py` — single-environment baseline training
- `project3_generalization/training/curriculum.py` — curriculum training
- `project3_generalization/training/ablations.py` — EWC-based forgetting control, frozen-readout transfer control, recurrence-strength ablation

#### 6. 3D scaffolding

`project3_generalization/environments/suite_3d.py` — lightweight 3D framework with:
- 3D environment specs
- Surface and volumetric navigators
- Simple 3D place/head-direction/boundary-vector feature generators
- Simulation utilities for future 3D predictive-RNN experiments

#### 7. Experiment entry points

- `project3_generalization/experiments/run_baselines.py`
- `project3_generalization/experiments/run_curriculum.py`
- `project3_generalization/experiments/run_two_module.py`
- `project3_generalization/experiments/run_ablation.py`
- `project3_generalization/experiments/run_3d.py`

#### 8. Analysis helpers

- `project3_generalization/analysis/figures.py`
- `project3_generalization/analysis/stats.py`

### Package Layout

```text
project3_generalization/
├── analysis/
├── environments/
├── evaluation/
├── experiments/
├── models/
└── training/
```

### Dependencies

The original repo dependencies are still relevant. For the new `project3_generalization` modules, the important additional packages are:

- `ratinabox`
- `shapely`
- `ripser`
- `gudhi`

`torch`, `numpy`, `scipy`, `matplotlib`, and `scikit-learn` are also used by the new modules.

### Validation Status

The following have been checked locally:

- The new package compiles successfully with `python3 -m compileall project3_generalization`
- 2D environment construction and rollout collection work
- 2D similarity matrix generation works on a small subset
- 3D environment simulation works for the lightweight navigator path

Full Torch-based training smoke tests, baseline checkpoint validation, and multi-seed experiment runs across curriculum, two-module, and ablation conditions still need completion.

---

## Project 4: Topology Before Geometry

This package extends the existing predictive-RNN codebase with a reuse-first layer investigating whether topological structure of the environment is learned before geometric details.

**Design priorities:**
1. Reuse the legacy rollout predictive RNN through `project4_topology_before_geometry.models.prnn.RolloutPRNN`
2. Keep MiniGrid as the paper-faithful primary backend and RatInABox as the fallback / non-trivial-topology backend
3. Track geometry and topology convergence separately through `ConvergenceTracker`

**Key entry points:**
- `project4_topology_before_geometry/scripts/run_local.py`
- `project4_topology_before_geometry/scripts/run_remote.py`
- `project4_topology_before_geometry/environments/env_factory.py`

**Scientific requirement:** Validate the `l_shape_standard` rollout baseline before novel experiments. The current code includes the baseline path and a passing smoke test, but the full `8e4`-trial replication run has not been completed yet.

---

## Original Figure Reproduction

The original figure notebooks and training scripts remain in the repository.
Jupyter notebooks for the original Levenstein et al. work are under `FigureScripts/`, and the older training/analysis workflow remains available alongside the new Project 3, 4, and 5 packages.