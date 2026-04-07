# LevensteinEtAl2024_PredictiveLearning

This repository started as a reproduction workspace for:

Levenstein D, Efremov A, Henha Eyono R, Peyrache A, Richards BA. Sequential predictive learning is a unifying theory for hippocampal representation and replay.  
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1)

The original codebase in this repo still contains the predictive RNN architectures, training utilities, analyses, and figure notebooks used for the Levenstein-style single-environment experiments.

## Project 3 Update

This repo now also includes a new package for:

**Generalization of Cognitive Map Formation in Predictive Recurrent Neural Networks Across 2D and 3D Arenas of Varied Geometry**

The new work lives under:

`project3_generalization/`

It is designed as an extension layer around the existing predictive-RNN code rather than a rewrite of the original repository.

## What Was Added

### 1. 2D environment suite

Added a configurable 2D environment library in:

`project3_generalization/environments/suite_2d.py`

This includes:

- Symmetric open arenas: square, large square, circle, rectangle
- Non-convex arenas: L-shape, T-maze, hairpin maze, compartmentalized arena
- Functional/landmark arenas: reward-zone layouts, barrier-with-gap, morph series
- Topology-focused arenas: annulus and figure-8 style environment

Also added:

- random-walk rollout generation
- simple environment validation helpers
- RatInABox-based observation generation using boundary-vector and head-direction signals

### 2. Structural similarity / successor representation pipeline

Added:

`project3_generalization/environments/similarity.py`

This module computes:

- discretized transition matrices
- successor representations
- pairwise structural similarity matrices across environments

### 3. Predictive model wrappers

Added:

- `project3_generalization/models/hippocampal_module.py`
- `project3_generalization/models/cortical_module.py`

These provide:

- a thin wrapper over the existing Levenstein predictive RNN architectures
- recurrence scaling support for ablations
- a cortical prior module for the two-module transfer experiments

### 4. Unified evaluation metrics

Added:

- `project3_generalization/evaluation/metrics.py`
- `project3_generalization/evaluation/topology.py`

This includes implementations or scaffolds for:

- sRSA reuse
- fraction of spatially tuned cells
- participation ratio
- replay quality
- CERA / CKA
- SR error and transfer-vs-similarity summaries
- elongation and remapping metrics
- Betti-number / persistent-homology helpers

### 5. Training pipelines

Added:

- `project3_generalization/training/single_env.py`
- `project3_generalization/training/curriculum.py`
- `project3_generalization/training/ablations.py`

These cover:

- single-environment baseline training
- curriculum training
- EWC-based forgetting control
- frozen-readout transfer control
- recurrence-strength ablation support

### 6. 3D scaffolding

Added:

`project3_generalization/environments/suite_3d.py`

This currently provides a lightweight 3D framework with:

- 3D environment specs
- surface and volumetric navigators
- simple 3D place/head-direction/boundary-vector feature generators
- simulation utilities for future 3D predictive-RNN experiments

### 7. Experiment entry points

Added runnable entry scripts in:

- `project3_generalization/experiments/run_baselines.py`
- `project3_generalization/experiments/run_curriculum.py`
- `project3_generalization/experiments/run_two_module.py`
- `project3_generalization/experiments/run_ablation.py`
- `project3_generalization/experiments/run_3d.py`

### 8. Analysis helpers

Added:

- `project3_generalization/analysis/figures.py`
- `project3_generalization/analysis/stats.py`

These provide lightweight figure and statistics helpers for transfer/similarity analyses.

## New Package Layout

```text
project3_generalization/
├── analysis/
├── environments/
├── evaluation/
├── experiments/
├── models/
└── training/
```

## Dependencies Needed For The New Work

The original repo dependencies are still relevant.  
For the new `project3_generalization` modules, the important additional packages are:

- `ratinabox`
- `shapely`
- `ripser`
- `gudhi`

`torch`, `numpy`, `scipy`, `matplotlib`, and `scikit-learn` are also used by the new modules.

## Current Validation Status

The following have been checked locally:

- the new package compiles successfully with `python3 -m compileall project3_generalization`
- 2D environment construction and rollout collection work
- 2D similarity matrix generation works on a small subset
- 3D environment simulation works for the lightweight navigator path

The following still need full experiment validation:

- full Torch-based training smoke tests in this environment
- the required baseline checkpoint that the L-shape reaches `sRSA > 0.4`
- full curriculum, two-module, and ablation experiment runs across multiple seeds

## Important Scientific Checkpoint

Before using the curriculum or transfer results, the intended baseline requirement remains:

**Confirm that the L-shaped environment baseline reaches `sRSA > 0.4`.**

That should be treated as the gating validation step before interpreting any multi-environment transfer effects.

## Original Figure Reproduction

The original figure notebooks and training scripts remain in the repository.  
Jupyter notebooks for the original work are under `FigureScripts/`, and the older training/analysis workflow remains available alongside the new Project 3 package.
