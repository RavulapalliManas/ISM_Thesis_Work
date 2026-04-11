# Project 3 Architecture

## High-level Overview

`project3_generalization/` is the newer, modular experiment stack layered on
top of the older predictive-RNN repository. Its purpose is to study how a
predictive recurrent neural network forms and transfers cognitive maps across
multiple 2-D and 3-D environments.

At a high level, the package is organized into five layers:

1. `environments/`
   - defines the tasks and arena geometry
   - generates trajectories and observations
   - computes environment-to-environment structural similarity

2. `models/`
   - wraps the legacy predictive-RNN architectures in a cleaner API
   - optionally adds a cortical prior module for transfer experiments

3. `training/`
   - implements reusable optimization loops
   - supports single-environment baselines, curricula, and ablations

4. `evaluation/`
   - computes neuroscience- and ML-oriented metrics such as sRSA, spatial
     tuning, replay quality, SR error, and topology summaries

5. `experiments/`
   - thin command-line entry points that assemble configs and call the shared
     training code

The `visual_rnn/` subpackage is an extension branch that swaps the older
hand-crafted sensory code for egocentric RGB patches rendered from the same 2-D
environment geometry.

## Core Data Flow

### Baseline 2-D training

`experiments/run_baselines.py`
-> `training/single_env.py`
-> `environments/suite_2d.py`
-> `models/hippocampal_module.py`
-> `evaluation/metrics.py`

1. An environment spec is selected from `build_suite_2d()`.
2. `collect_rollout_2d()` samples a trajectory and observation/action stream.
3. `HippocampalPredictiveRNN` predicts the next observation from the current
   observation and action sequence.
4. Periodic evaluation computes representational metrics from hidden states.
5. Results and checkpoints are written to disk.

### Curriculum training

`experiments/run_curriculum.py`
-> `environments/similarity.py`
-> `training/curriculum.py`
-> `training/single_env.py`

1. A set of environments is chosen.
2. `compute_similarity_matrix()` estimates structural similarity using
   transition matrices and successor representations.
3. A curriculum order is selected, either greedily by similarity or randomly.
4. The same model is trained sequentially across environments.
5. Transfer is measured from zero-shot performance or relative learning speed.
6. Re-exposure to the first environment quantifies recovery after interference.

### Visual-input training

`train_single_run.py`
-> `visual_rnn/train.py`
-> `visual_rnn/renderer.py`
-> `models/hippocampal_module.py`

1. A 2-D environment is rasterized into a `TileMap`.
2. `collect_rollout_2d(..., observation_mode="visual")` renders egocentric
   `7x7x3` patches along the trajectory.
3. The hippocampal model uses an observation adapter, typically a CNN, to
   encode patches before the recurrent core.
4. Training logs, checkpoints, plots, and post-run summaries are written to a
   dedicated run directory.

## Key Abstractions

### `EnvironmentSpec2D`
Immutable geometric description of a 2-D arena.

### `SimulationRollout2D`
Container for one rollout, including observations, actions, positions, and
optionally rendered visual patches.

### `HippocampalConfig`
Dataclass holding the full model/training-facing configuration for the
hippocampal predictive RNN wrapper.

### `HippocampalPredictiveRNN`
Main Project 3 model class. It wraps legacy recurrent architectures, handles
encoding/decoding, computes losses, and exposes checkpointing and batch APIs.

### `SingleEnvironmentConfig` / `CurriculumConfig`
Dataclasses that define the optimization schedule and evaluation cadence for
baseline and curriculum experiments.

### `TransitionEstimate`
Container for an environment's estimated transition matrix, occupancy map, and
successor representation.

## External Dependencies

- `torch`: model definition and optimization
- `numpy`: array computation
- `scipy`: sparse linear algebra and statistics
- `scikit-learn`: decoding, PCA, and similarity helpers
- `matplotlib`: plotting
- `shapely`: 2-D geometry construction and rasterization support
- `ratinabox`: 2-D environment simulation and sensory rollouts
- `ripser`: persistent-homology calculations
- `umap-learn`: optional hidden-state embeddings for visualization

## How To Read The Code Quickly

If you only have 30 to 60 minutes:

1. `ARCHITECTURE.md`
2. `environments/suite_2d.py`
3. `models/hippocampal_module.py`
4. `training/single_env.py`
5. `training/curriculum.py`
6. `evaluation/metrics.py`
7. `experiments/run_hardware_constrained.py`

If you care specifically about the visual branch:

1. `visual_rnn/renderer.py`
2. `visual_rnn/model.py`
3. `visual_rnn/train.py`
4. `visual_rnn/analysis.py`
