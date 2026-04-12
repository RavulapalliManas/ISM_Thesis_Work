# Handoff Document

This document is for Gemini, Claude, or any other follow-on agent picking up work in `project4_topology_before_geometry`.

It is intentionally concrete. It covers:

- the active scientific framing
- the latest prompt/context
- what has already been implemented
- what was validated
- what is still incomplete
- what to do next, in order
- files that matter
- known pitfalls

## Workspace

- Repo root: `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work`
- Main package: `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry`

## Mission Summary

This codebase extends the Levenstein et al. 2024 predictive hippocampal RNN work into a larger environment/topology/aliasing study.

There are now two active layers of requirements:

1. The original `topology-before-geometry` specification.
2. A newer extension requiring a large MiniGrid-only environment library with explicit geometry x aliasing disentanglement.

The scientific core has not changed:

- Reuse the existing legacy pRNN architecture and training logic where possible.
- Do not rewrite `thetaRNN.py`, `Architectures.py`, or the core loss logic.
- Baseline replication on `l_shape_standard` remains the credibility gate before claiming novel science.

## Active Prompt Context

### Foundational Prompt: Topology Before Geometry

The project was originally driven by a long unified spec based on Levenstein et al. 2024. The most important parts are:

- Build on top of the existing predictive RNN code, do not reimplement the backbone.
- Replicate the paper baseline on `l_shape_standard` before pushing novel claims.
- Add environment diversity, topological measurements, persistent homology, geodesic sRSA, replay analysis, and convergence-gap analysis.
- Keep MiniGrid as the primary backend and RatInABox as a fallback for non-grid topologies.
- Treat rollout-MSE as the main figure loss. Loss ablations are supplementary.
- Enforce the stricter topology criterion using Betti correctness plus persistence-gap logic.

Important standing constraints from that spec:

- Do not rewrite the pRNN architecture.
- Do not rewrite the loss logic.
- Do not rewrite metric semantics.
- Do not rewrite environment physics/rendering engines.
- Reuse existing code when available.

### Current Prompt: Extended MiniGrid Environment Library + GridWorld-Only Constraint

The latest user prompt adds a second requirement layer:

- Build a comprehensive MiniGrid/GridWorld-only environment framework.
- Add many prebuilt environments while keeping aliasing control explicit.
- Support geometry x aliasing disentanglement.
- Use only MiniGrid/GridWorld primitives. No custom rendering engine. No continuous physics for this new extension.

Key definitions from the current prompt:

- Aliasing metric:
  - `A = E[cos(patch(x1,y1,theta), patch(x2,y2,theta))]` for geodesically distant states.
- Hypothesis:
  - high aliasing breaks map formation
  - low aliasing enables map formation
  - the system should support finding the phase transition threshold

New required environment categories:

- Base parametric aliasing envs:
  - `zero_alias`, `low_alias`, `medium_alias`, `high_alias`, `maximum_alias`
- Geometric envs:
  - `square`, `rectangle`, `circle_approx`, `l_shape`, `u_shape`, `corridor`, `t_maze`, `plus_maze`
  - `two_room`, `three_room`, `four_room`, `bottleneck_room`, `maze_simple`, `maze_medium`
  - `loop_corridor`, `spiral_maze`, `dead_end_maze`, `branching_tree`
  - `figure_8`, `double_loop`, `nested_rooms`, `room_with_island`
- Aliasing-stress envs:
  - `uniform_box`, `repeating_stripes`, `checkerboard_large_period`, `symmetry_trap`
  - `long_corridor_alias`, `ambiguous_junctions`, `perceptual_alias_maze`
  - `no_landmarks`, `boundary_only_landmarks`, `center_only_landmark`, `sparse_random_landmarks`
- Combined envs:
  - `two_room_low_alias`, `two_room_high_alias`
  - `l_shape_low_alias`, `l_shape_high_alias`
  - `maze_low_alias`, `maze_high_alias`

Current naming requirement from the prompt:

- canonical form should encode geometry and alias parameters
- prompt example used Greek symbols:
  - `{geometry}_τ={}_λ={}_H={}_ω={}`

Implementation note:

- The repo now uses an ASCII-safe internal canonical name to avoid Windows/PowerShell encoding failures:
  - `{geometry}_tau={}_lambda={}_H={}_omega={}`
- A separate `canonical_display_name` preserves the Greek-symbol display version.

## What Has Already Been Done

### Earlier Engineering Work Already In Place

These were implemented before the latest environment-library extension:

- Rollout pRNN package scaffold built in `project4_topology_before_geometry`.
- `tqdm` progress bars added.
- Gradient accumulation added:
  - progress now advances by optimizer update, not raw trajectory
  - `5000` trajectories with `n_accumulate=8` means `625` update steps
- Remote-performance optimizations added:
  - cached geodesic distances
  - async topology worker
  - rollout prefetch
  - async CSV logging
  - mixed precision support
  - `torch.compile` helper
- `run_remote.py` fixed so `--env` is optional and config defaults are usable.
- Direct-file execution path bootstrap added to `run_local.py` and `run_remote.py`.

Main files for that work:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\evaluation\geometric_metrics.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\evaluation\convergence_tracker.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\training\trainer.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\models\prnn.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\run_local.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\run_remote.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\configs\remote_config.yaml`

### Latest Environment-Library Extension Implemented

The following new MiniGrid-only aliasing-controlled framework is now present.

#### 1. New environment library

File:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\environments\aliasing_controlled_envs.py`

Implemented:

- `AliasingControlledEnv`
- `AliasingControlledMiniGridCore`
- `build_layout(layout_type)`
- `make_environment(...)`
- `list_prebuilt_environments()`
- `is_aliasing_controlled_name(...)`

Supported request styles:

- string names:
  - `square`
  - `two_room_high_alias`
  - `uniform_box`
  - `square_low_alias`
- dict specs:
  - `{"geometry": "two_room", "tile_period": 3, "landmark_density": 0.0, "wall_entropy": 0.2, "gradient_weight": 0.0}`
- explicit parameterized names:
  - `two_room_tau=3_lambda=0_H=0.2_omega=0`

Supported geometry families currently implemented in the layout builder:

- `square`
- `rectangle`
- `circle_approx`
- `l_shape`
- `u_shape`
- `corridor`
- `t_maze`
- `plus_maze`
- `two_room`
- `three_room`
- `four_room`
- `bottleneck_room`
- `maze_simple`
- `maze_medium`
- `loop_corridor`
- `spiral_maze`
- `dead_end_maze`
- `branching_tree`
- `figure_8`
- `double_loop`
- `nested_rooms`
- `room_with_island`

Supported stress/combined names currently mapped:

- `uniform_box`
- `repeating_stripes`
- `checkerboard_large_period`
- `symmetry_trap`
- `long_corridor_alias`
- `ambiguous_junctions`
- `perceptual_alias_maze`
- `no_landmarks`
- `boundary_only_landmarks`
- `center_only_landmark`
- `sparse_random_landmarks`
- `square_low_alias`
- `square_high_alias`
- `two_room_low_alias`
- `two_room_high_alias`
- `l_shape_low_alias`
- `l_shape_high_alias`
- `maze_low_alias`
- `maze_high_alias`

#### 2. Extended aliasing controls

File:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\sensory\aliasing_control.py`

Implemented:

- alias presets:
  - `zero_alias`
  - `low_alias`
  - `medium_alias`
  - `high_alias`
  - `maximum_alias`
- helpers:
  - `preset_to_params(...)`
  - `alias_level_from_params(...)`
- richer tile generation modes:
  - `uniform`
  - `periodic`
  - `clustered`
  - `sparse_random`
  - `stripes`
  - `repeating_stripes`
  - `checkerboard`
  - `symmetry`
  - `boundary_only`
  - `center_only`
  - `gradient`

The existing metrics stayed in place:

- `compute_aliasing_score(...)`
- `compute_geo_euclidean_discrepancy(...)`

#### 3. Dynamic topology label resolution

File:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\environments\topology_labels.py`

Implemented:

- `get_topology_label(env_name)`
- dynamic parsing of new environment names
- support for old hard-coded envs and new alias-controlled envs

This was necessary because the old `TOPOLOGY_LABELS[env_name]` pattern breaks for generated names.

#### 4. Factory integration

File:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\environments\env_factory.py`

Implemented:

- `get_env()` now routes both legacy envs and the new MiniGrid alias-controlled envs
- `list_environments()` now includes the new environment library
- factory bug fixed so `seed` is not passed twice into `make_environment(...)`

#### 5. Sweep helper

File:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\aliasing_sweep.py`

Implemented:

- `base_alias`
- `geometry_sweep`
- `geometry_alias_cross`
- `failure_modes`

This is still a helper script, not a full experiment runner.

#### 6. Validation helper

File:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\verify_aliasing.py`

Implemented diagnostics:

- connectivity visualization
- bottleneck highlighting
- symmetry-axis detection
- duplicate-patch warning
- disconnected-region warning

Implemented checks/warnings for the new prompt requirements:

- identical patches across more than 20% of sampled positions
- symmetry without disambiguating cues
- disconnected regions

## What Was Validated

### Compile/import sanity

`compileall` over `project4_topology_before_geometry` completed successfully after the latest edits.

### Environment creation smoke tests

Successfully instantiated:

- `square`
- `two_room_high_alias`
- `uniform_box`
- `square_low_alias`
- legacy `l_shape_standard`
- dict spec for a `two_room` env with explicit alias params

`list_environments()` reported the expanded library correctly.

### Aliasing verification smoke tests

`verify_aliasing.run_checks("symmetry_trap", {"seed": 42})` produced:

- aliasing score around `0.649`
- duplicate patch fraction around `0.94`
- symmetry axes detected
- expected warnings emitted

`verify_aliasing.run_checks("long_corridor_alias", {"seed": 42})` produced:

- aliasing score around `0.638`
- duplicate patch fraction around `0.99`
- expected warnings emitted
- connectivity figure saved

Generated artifact:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\figures\aliasing_diagnostics\long_corridor_alias_connectivity.png`

### Important encoding fix validated

The original attempt to use Greek symbols directly in canonical environment names caused Windows/PowerShell encoding issues.

This is now handled by:

- ASCII-safe internal canonical names
- Greek-symbol display names kept separately

This prevents `UnicodeEncodeError` during normal use on Windows.

## What Is Still Incomplete

This section is the most important one for the next model.

### 1. No full scientific run has been completed on the new environment library

The new MiniGrid alias-controlled envs are wired into the factory, but there is not yet:

- a full training sweep over the new envs
- a phase-transition analysis over aliasing levels
- a geometry x aliasing benchmark report
- a completed result table

### 2. Baseline replication gate is still the scientific blocker

The project still has not completed the full required baseline replication on `l_shape_standard` at the paper target scale.

This means:

- infrastructure is increasingly solid
- but novel scientific claims should still be treated as provisional until baseline replication is confirmed

Replication gate from the original spec:

- `sRSA_euclidean > 0.6`
- `SW_dist < 0.1`
- `fraction_tuned > 0.15`

### 3. The new environment library is functional, but not yet fully integrated into experiment configs

What is missing:

- dedicated config entries for the expanded MiniGrid library
- sweep configs for systematic geometry x aliasing runs
- a clean benchmark script that consumes `aliasing_sweep.py`
- result aggregation across seeds

### 4. The prompt’s alias-threshold science has not yet been executed

The latest prompt specifically wants phase-transition analysis around aliasing thresholds.

Not done yet:

- estimate `P(map formation)` as a function of aliasing
- find the threshold where success probability is about `0.5`
- demonstrate failure modes by environment class
- compare topology and geometry formation timing as aliasing varies

### 5. Geometry diagnostics exist, but publication-level analysis does not

`verify_aliasing.py` is a good start, but missing:

- summary report generation
- automated batch validation across all prebuilt envs
- standardized plots for symmetry, bottlenecks, and duplicate-patch rates

### 6. Some environment names overlap conceptually with legacy envs

Examples:

- `l_shape_standard` is still the original legacy environment path
- `l_shape_low_alias` is part of the new alias-controlled library

This is fine, but it needs to remain explicit in future analysis to avoid mixing incomparable conditions.

### 7. Generated artifacts and caches are in the worktree

The repo currently has:

- logs
- checkpoints
- figures
- `__pycache__`

These were not cleaned up because no destructive cleanup was requested.

## Current Git/Worktree State

At the time of this handoff, the working tree includes modifications and new files related to the environment-library extension.

Important tracked/edited files:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\environments\env_factory.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\environments\topology_labels.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\run_local.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\run_remote.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\sensory\aliasing_control.py`

Important new files:

- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\environments\aliasing_controlled_envs.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\aliasing_sweep.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\scripts\verify_aliasing.py`
- `C:\Users\91984\Desktop\ISM work\ISM_Thesis_Work\project4_topology_before_geometry\HANDOFF_GEMINI_CLAUDE.md`

There are also generated figures/logs/checkpoints and `__pycache__` artifacts in the worktree.

## Recommended Next Steps

This is the recommended order for Gemini/Claude or the next agent.

### Priority 1: Preserve the scientific gate

1. Confirm the baseline replication path on `l_shape_standard`.
2. Do not present the new environment-library science as final until baseline replication is either passed or clearly documented as pending.

### Priority 2: Turn the new library into a runnable benchmark

1. Create a config or benchmark runner that consumes `aliasing_sweep.py`.
2. Add seed handling and per-env output naming.
3. Log:
   - aliasing score
   - sRSA
   - SW distance
   - tuned fraction
   - convergence gap
   - topology correctness
4. Save a compact summary CSV across envs and seeds.

### Priority 3: Execute the aliasing-threshold study

1. For each base geometry, sweep alias strength systematically.
2. Estimate success/failure rates for map formation.
3. Fit or at least report the approximate phase-transition threshold.
4. Compare failure modes across:
   - symmetry
   - corridor ambiguity
   - sparse landmarks

### Priority 4: Strengthen environment validation

1. Batch-run `verify_aliasing.py` across all prebuilt envs.
2. Save a validation report including:
   - aliasing score
   - duplicate patch fraction
   - symmetry axes
   - bottlenecks
   - disconnected flag
3. Flag envs that are invalid or too similar to one another.

### Priority 5: Keep naming/reporting disciplined

1. Clearly distinguish legacy envs from new alias-controlled envs.
2. Use ASCII canonical names in file/log paths.
3. Use `canonical_display_name` only for figures or human-readable reporting.

## Practical Commands To Run Next

These are reasonable next commands for the next model/operator.

### Inspect the new environment library

```powershell
python -c "from project4_topology_before_geometry.environments.env_factory import list_environments; print(list_environments())"
```

### Print the planned sweep definitions

```powershell
python project4_topology_before_geometry\scripts\aliasing_sweep.py
```

### Run aliasing diagnostics on a stress-test env

```powershell
python -c "from project4_topology_before_geometry.scripts.verify_aliasing import run_checks; print(run_checks('symmetry_trap', {'seed': 42}))"
```

### Run aliasing diagnostics on a corridor stress env

```powershell
python -c "from project4_topology_before_geometry.scripts.verify_aliasing import run_checks; print(run_checks('long_corridor_alias', {'seed': 42}))"
```

### Run the local training script

```powershell
python project4_topology_before_geometry\scripts\run_local.py
```

### Run the remote training script with explicit env

```powershell
python project4_topology_before_geometry\scripts\run_remote.py --env l_shape_standard --seed 42
```

## Known Gotchas

### Update count vs trial count

Training progress bars currently advance in optimizer updates, not raw trajectories, because of gradient accumulation.

Example:

- `n_trials = 5000`
- `n_accumulate = 8`
- progress bar total becomes `625`

This is expected.

### Aliasing score interpretation

The current aliasing score is mean cosine similarity between distant egocentric observations.

Important:

- it is not a percentage
- it is not literally “64% of positions are aliased”
- values closer to `1` mean distant observations look very similar

### Windows encoding

Do not rely on Greek-symbol environment names as internal file identifiers on Windows.

Use:

- `canonical_name` for files/logs
- `canonical_display_name` for display

### Do not touch these unless explicitly intended

Per project constraints, avoid modifying:

- `thetaRNN.py`
- `Architectures.py`
- core loss semantics
- metric semantics
- environment physics engines

## Suggested Handoff Message

If another model needs the shortest possible orientation, use this:

> The repo now has a working MiniGrid-only aliasing-controlled environment library integrated into `env_factory.py`, along with sweep and verification helpers. The main remaining work is not framework plumbing but scientific execution: baseline replication on `l_shape_standard`, then systematic geometry x aliasing sweeps and phase-transition analysis. Use the new envs through `make_environment(...)` / `get_env(...)`, keep internal names ASCII-safe, and do not modify the legacy pRNN core unless baseline replication proves the architecture is wrong.

## Final Status

The codebase is meaningfully further along than before:

- infrastructure exists
- the new environment library exists
- validation helpers exist
- the factory routing exists

What does not exist yet is the finished scientific result set for the new prompt.

That is the next stage.

## Appendix A: Latest User Prompt (Verbatim)

```text
Here is a clean, extended version of your original prompt with the requested modification:

* Keeps your scientific framing intact
* Adds a large library of prebuilt MiniGrid-style environments
* Ensures everything still flows through the aliasing-controlled framework
* Explicitly enforces reuse of GridWorld/MiniGrid primitives only (no custom engines)

You are extending a computational neuroscience codebase that studies cognitive map formation in predictive recurrent neural networks (pRNNs).

The scientific goal is to systematically vary the ALIASING PROPERTIES of 2D MiniGrid-style environments to find the phase transition threshold below which cognitive maps reliably form.

Aliasing Definition

A = E[cos(patch(x1,y1,theta), patch(x2,y2,theta))] for d_geodesic > threshold

- High aliasing (A > 0.5) -> network fails (stuck basin)
- Low aliasing (A < 0.3) -> map formation succeeds (hypothesis)
- Phase transition = P(map formation) = 0.5

Current Problem

- square_low_alias actually has A ~= 0.639
- Caused by periodic floor tiling (tau = 3)
- Network saturates at sRSA ~= 0.30

Objective

Build a comprehensive environment framework that:

1. Provides parametric aliasing control
2. Adds many prebuilt environments for benchmarking
3. Uses ONLY MiniGrid/GridWorld primitives
4. Allows geometry x aliasing disentanglement

Four primary aliasing controls:

- tau: floor tile period
- lambda: landmark density
- H: wall entropy
- omega: gradient weight

New requirement: Large environment library

The factory should support:

1. Base parametric environments:
   - zero_alias
   - low_alias
   - medium_alias
   - high_alias
   - maximum_alias

2. Geometric environments:
   - square
   - rectangle
   - circle_approx
   - l_shape
   - u_shape
   - corridor
   - t_maze
   - plus_maze
   - two_room
   - three_room
   - four_room
   - bottleneck_room
   - maze_simple
   - maze_medium
   - loop_corridor
   - spiral_maze
   - dead_end_maze
   - branching_tree
   - figure_8
   - double_loop
   - nested_rooms
   - room_with_island

3. Aliasing-stress environments:
   - uniform_box
   - repeating_stripes
   - checkerboard_large_period
   - symmetry_trap
   - long_corridor_alias
   - ambiguous_junctions
   - perceptual_alias_maze
   - no_landmarks
   - boundary_only_landmarks
   - center_only_landmark
   - sparse_random_landmarks

4. Combined geometry x aliasing environments:
   - two_room_low_alias
   - two_room_high_alias
   - l_shape_low_alias
   - l_shape_high_alias
   - maze_low_alias
   - maze_high_alias

Strict constraint:

- Use MiniGrid / GridWorld primitives only
- No custom rendering engines
- No OpenCV environments
- No continuous physics

Allowed:

- Walls
- Doors
- Empty cells
- Objects (landmarks)

Implementation asks:

- Extend class AliasingControlledEnv
- Add build_layout(self, layout_type)
- Extend make_environment(env_type, **kwargs)
- Support both:
  - make_environment("two_room_high_alias")
  - make_environment({"geometry": "two_room", "tile_period": 3, "n_landmarks": 0})
- Naming convention:
  - {geometry}_tau={}_lambda={}_H={}_omega={}

Scientific requirements:

- Geometry != aliasing
- Must test same geometry with different aliasing
- Must test same aliasing with different geometry
- Must expose symmetry-induced aliasing, corridor aliasing, junction ambiguity
- Must support the hypothesis:
  "Topology emerges before geometry when aliasing is below threshold"

Add sweep support:

- geometry_sweep
- geometry_alias_cross

Extend verify_aliasing with:

- connectivity graph visualization
- bottleneck highlighting
- symmetry-axis detection

Warn on:

- identical patches across >20% of positions
- symmetry without disambiguating cues
- disconnected regions
```

## Appendix B: Immediate To-Do Shortlist

If a new model needs a very short action list, do these next:

1. Run the baseline `l_shape_standard` validation and record whether the replication gate passes.
2. Batch-validate the new MiniGrid library with `verify_aliasing.py`.
3. Turn `aliasing_sweep.py` into an executable benchmark pipeline with configs and CSV summaries.
4. Run geometry x aliasing sweeps and estimate the phase transition threshold.
5. Keep all internal env names ASCII-safe and do not modify the legacy pRNN core unless the baseline fails.
