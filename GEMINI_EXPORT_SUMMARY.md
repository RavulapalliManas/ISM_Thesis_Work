# Gemini Handoff Summary

## Context

This conversation focused on extending the existing Levenstein predictive-RNN repository into a new multi-environment research package for:

**"Generalization of Cognitive Map Formation in Predictive Recurrent Neural Networks Across 2D and 3D Arenas of Varied Geometry."**

The original repository is a Levenstein-style predictive learning codebase. The new work was added as a separate package rather than rewriting the old code.

## What Was Implemented

### New package

A new package was created at:

- `project3_generalization/`

It currently contains:

- `project3_generalization/environments/suite_2d.py`
- `project3_generalization/environments/similarity.py`
- `project3_generalization/environments/suite_3d.py`
- `project3_generalization/models/hippocampal_module.py`
- `project3_generalization/models/cortical_module.py`
- `project3_generalization/training/single_env.py`
- `project3_generalization/training/curriculum.py`
- `project3_generalization/training/ablations.py`
- `project3_generalization/evaluation/metrics.py`
- `project3_generalization/evaluation/topology.py`
- `project3_generalization/experiments/run_baselines.py`
- `project3_generalization/experiments/run_curriculum.py`
- `project3_generalization/experiments/run_two_module.py`
- `project3_generalization/experiments/run_ablation.py`
- `project3_generalization/experiments/run_3d.py`
- `project3_generalization/analysis/figures.py`
- `project3_generalization/analysis/stats.py`

### 2D environment suite

The 2D module now includes:

- open arenas: square, large square, circle, rectangle
- non-convex arenas: L-shape, T-maze, hairpin, compartmentalized arena
- reward/barrier environments
- square-to-circle morph series
- annulus and figure-8 style topologies

### Structural similarity

`project3_generalization/environments/similarity.py` was added to compute:

- discretized transition matrices
- successor representations
- pairwise structural similarity matrices

### Model wrappers

`project3_generalization/models/hippocampal_module.py` wraps the existing Levenstein predictive-RNN architecture and provides:

- `HippocampalConfig`
- `HippocampalPredictiveRNN`
- RMSprop optimizer setup
- recurrence scaling support
- batch prediction and training helpers

`project3_generalization/models/cortical_module.py` adds a cortical prior module for the two-module experiments.

### Training pipelines

The following training paths were scaffolded:

- single-environment baselines
- curriculum training
- EWC support
- frozen-readout control
- recurrence ablation

### Metrics

The following metrics/helpers were added or scaffolded:

- sRSA reuse
- fraction of spatially tuned cells
- participation ratio
- replay quality
- CERA
- CKA
- SR error
- transfer-vs-similarity summary
- eigenspectrum overlap
- elongation index
- topological remapping
- Betti number / persistent homology utilities

### 3D scaffolding

A lightweight 3D framework was added with:

- 3D environment specs
- surface and volumetric navigators
- simple 3D place/head-direction/boundary-vector feature generators
- basic 3D rollout support

### README update

The repository README was rewritten to document the new `project3_generalization` package and its current validation state.

File:

- `README.md`

## What Was Verified

The following checks were completed successfully:

- `python3 -m compileall project3_generalization`
- 2D environment construction
- 2D rollout collection
- small-subset similarity matrix generation
- lightweight 3D simulation

## What Was Not Fully Verified

Full Torch-based smoke tests could not be completed in the sandbox because importing/running `torch` hit an OpenMP shared-memory error:

- `OMP: Error #179: Function Can't open SHM2 failed`

An outside-sandbox smoke test was requested, but approval was rejected, so the following remain unverified in practice:

- actual end-to-end training runs
- the required L-shape baseline checkpoint
- curriculum and two-module experiments
- recurrence ablation runs

## Important Scientific Checkpoint

A major requirement from the project brief is still outstanding:

- confirm the L-shape baseline reaches `sRSA > 0.4`

This has **not** yet been run to completion.

## Most Important Conceptual Issue Discovered

During the conversation, a key mismatch was identified between the current implementation and the intended scientific setup.

### What is implemented right now

The current RatInABox rollout pipeline in:

- `project3_generalization/environments/suite_2d.py`

builds each observation vector as:

- `BoundaryVectorCells` firing rates
- `HeadDirectionCells` firing rates

and builds the action input from:

- `vx`
- `vy`
- rotational velocity

So, the RNN is currently trained on **BVC + HD sensory vectors**, not on vision-like sensory patches.

### Why this is a mismatch

The user pointed out an important distinction:

- In the original Levenstein framework, a major part of the coherence/emergent-map story comes from predicting an egocentric sensory input like a `7x7x3` patch in front of the agent.
- Using BVCs as the main sensory input may "bake in" spatial structure too directly and may undermine the claim that the representation is emerging from predictive learning.

This concern is valid and was explicitly raised by the user.

### Current status of that issue

This was **identified and explained**, but **not yet fixed in code**.

So the present implementation should be treated as:

- a structural scaffold for the project
- **not yet a faithful implementation of the desired sensory-prediction regime**

## How the current training loop works

At a high level:

1. RatInABox simulates an agent moving in a continuous arena.
2. At each timestep, sensory neurons are sampled.
3. Those sensory values form the observation vector.
4. Motion variables form the action input.
5. The predictive RNN is trained to predict future observations over a sequential masked rollout.

Relevant files:

- `project3_generalization/environments/suite_2d.py`
- `project3_generalization/models/hippocampal_module.py`
- `project3_generalization/training/single_env.py`
- `utils/Architectures.py`
- `utils/lossFuns.py`

## User's Final Position on Sensory Input

The user explicitly said:

- they do **not** want BVC input as the main sensory input
- they care about preserving emergent properties
- they want the setup to stay aligned with the Levenstein logic where predictive learning over sensory structure drives map formation

This means the next revision should likely replace the current BVC-based observation model with something closer to:

- egocentric local visual input
- a front-facing sensory patch
- or another minimally structured sensory encoding that does not directly hand the network spatial features

## Recommended Next Steps For Gemini

### Highest priority

Rework the sensory pipeline so that the predictive target is no longer BVC-dominated.

Most likely path:

1. Replace the current `BoundaryVectorCells + HeadDirectionCells` observation vector in `project3_generalization/environments/suite_2d.py`.
2. Introduce a sensory representation more faithful to Levenstein's setup:
   - egocentric local view
   - front-of-agent patch
   - or another explicitly non-spatially-predecoded sensory stream
3. Keep head direction as a separate side input if needed.
4. Re-check input/output dimensionality in `HippocampalConfig`.
5. Re-run the L-shape baseline and verify `sRSA > 0.4`.

### Secondary

After the sensory mismatch is fixed:

1. validate the L-shape baseline first
2. then run single-environment baselines
3. then run curriculum experiments
4. then run two-module experiments
5. then run recurrence ablations

## Repo Changes Made In This Conversation

### Added

- `project3_generalization/`
- `GEMINI_EXPORT_SUMMARY.md`

### Modified

- `README.md`

## Bottom line

This conversation produced a substantial project scaffold and integrated it into the repo, but it also uncovered a scientifically important mismatch:

**the current implementation uses BVC-based sensory input, whereas the user wants a more Levenstein-faithful sensory prediction setup that better preserves the possibility of emergent representations.**

That mismatch is the most important thing Gemini should address next.
