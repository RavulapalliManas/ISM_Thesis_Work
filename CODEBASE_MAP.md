# Codebase Map

This repository currently contains two partially overlapping research codebases:

1. `legacy predictive-RNN workflow`
   - centered on `PredictiveNet`
   - used for the original Levenstein-style single-environment experiments
   - training, analysis, and figure generation are split across `utils/`, `analysis/`, `FigureScripts/`, and top-level scripts

2. `Project 3 generalization workflow`
   - centered on `project3_generalization/`
   - adds modular 2D/3D environment suites, curriculum experiments, structural-similarity analysis, ablations, and a newer visual-input pipeline

## Mental Model

If you want the shortest way to think about the repo:

- `utils/` = old shared engine room
- `analysis/` = old post-hoc neuroscience analyses
- `trainNet.py` = old training entry point
- `run_analysis.py` = old spatial tuning demo / figure reproduction helper
- `project3_generalization/` = newer, cleaner package for Project 3
- `project3_generalization/experiments/` = new experiment launchers
- `project3_generalization/training/` = reusable new training loops
- `project3_generalization/models/` = new model wrappers around the old architecture family
- `project3_generalization/environments/` = new task/environment definitions
- `project3_generalization/evaluation/` = new metrics
- `project3_generalization/visual_rnn/` = newest visual-observation branch

## Main Execution Paths

### Legacy path

`trainNet.py` -> `utils.predictiveNet.PredictiveNet` -> `utils.Architectures` / `utils.thetaRNN` -> `analysis/*` and `utils/figures.py`

This is the older monolithic path. Training, metrics, decoding, spontaneous replay analysis, and saved plots are all driven from the `PredictiveNet` object.

### Project 3 path

`project3_generalization/experiments/run_*.py`
-> `project3_generalization/training/*`
-> `project3_generalization/models/*`
-> `project3_generalization/environments/*`
-> `project3_generalization/evaluation/*`

This path is more modular. Each experiment script assembles configs, builds environment specs, calls a training loop, and serializes a summary.

### Visual RNN path

`train_single_run.py`
-> `project3_generalization.visual_rnn.train`
-> `project3_generalization.visual_rnn.renderer`
-> `project3_generalization.models.hippocampal_module`

This is the newest path for image-like egocentric visual observations rather than the older hand-crafted BVC/head-direction observation stream.

## Top-Level Files And Folders

### Root files

- `README.md`: high-level description of the old repo plus the Project 3 extension.
- `updates.md`: notes from the recent chat-driven additions, mainly the visual-RNN path.
- `config_hardware.yaml`: hardware-budget config for Project 3 experiments. Despite the `.yaml` suffix, the file currently contains JSON.
- `requirements.txt`: base dependencies.
- `requirements_visual_rnn.txt`: dependencies for the newer visual-RNN path.
- `run_single_instance.sh`: shell helper for one legacy training run.
- `trainNet.py`: legacy training entry point for one predictive-RNN experiment.
- `run_analysis.py`: legacy analysis/reproduction script that loads one saved network and runs spatial tuning analysis.
- `train_single_run.py`: new entry point for one visual predictive-RNN run from a JSON config.

### Root folders

- `analysis/`: legacy analysis modules built around saved `PredictiveNet` objects and decoded trajectories.
- `BashScripts_ClusterTraining/`: legacy cluster job launchers and one Python hyperparameter panel generator.
- `configs/`: JSON config examples for the visual pipeline.
- `FigureScripts/`: Jupyter notebooks for manuscript figure reproduction.
- `nets/`: saved model checkpoints, training figures, and legacy outputs.
- `project3_generalization/`: newer modular package for Project 3.
- `utils/`: legacy core library for models, environments, action encodings, decoders, losses, and plotting.
- `.venv/`: local virtual environment, not part of the scientific code.

## Folder-By-Folder Map

### `analysis/`

Legacy neuroscience analyses that assume a trained `PredictiveNet` and then ask representational questions about wake activity, spontaneous replay, geometry, connectivity, and object memory.

- `decodeAnalysis.py`: lightweight decoder-analysis wrapper for wake sequences.
- `DiffusionReplayAnalysis.py`: studies replay trajectories through diffusion-style fits, decoder-based sleep trajectory analysis, and noise-level sweeps.
- `ExperienceReplayAnalysis.py`: evaluates replay of stored experience snippets, overlap with experience, replay scores, and replay trial statistics.
- `ObjectMemoryTask.py`: constructs an object-memory variant of the task, trains a decoder, and quantifies learning of novel objects.
- `OfflineActivityAnalysis.py`: broader spontaneous-activity analysis module with transition maps, coverage, wake-vs-sleep comparison, and summary figures.
- `OfflineTrajectoryAnalysis.py`: main legacy sleep/replay trajectory analysis class; decodes spontaneous activity, computes continuity, coverage, coherence, view similarity, and diffusion fits.
- `RepresentationalConnectivityAnalysis.py`: links tuning/reliability properties to weight structure and representational distance.
- `representationalGeometryAnalysis.py`: computes RSA-style geometry for action, space, observations, and head direction, with Isomap helpers.
- `SpatialTuningAnalysis.py`: place-field / spatial-information analysis, tuning-curve reliability, and recurrence-ablation controls.
- `trajectoryAnalysis.py`: low-level coverage and continuity calculations reused by replay analyses.

### `BashScripts_ClusterTraining/`

Legacy batch-launch support for running many old-style trainings on a cluster.

- `hyperparm_panel.py`: generates shell commands over a hyperparameter grid.
- `Autoencoder_sparse_panel.sh`: launch panel for sparse autoencoder experiments.
- `hyperparm_panel_masked.sh`: masked-network batch launcher.
- `hyperparm_panel_masked1.sh`: alternate masked-network batch launcher.
- `hyperparm_panel_sparse.sh`: sparse-network batch launcher.
- `hyperparm_panel_theta.sh`: theta-RNN batch launcher.
- `maskedk_panel.sh`: masked-architecture sweep launcher.

### `configs/`

Configuration examples for the newest single-run visual pipeline.

- `visual_single_run.example.json`: example config for `train_single_run.py`.

### `FigureScripts/`

Notebook-only folder for manuscript figure reproduction and exploratory figure generation. These notebooks sit on top of the legacy code rather than define reusable library code.

- `Figure1_PredictiveLearningSpatialRep.ipynb` ... `FigureS21.ipynb`: figure-specific notebooks for the original paper/supplement workflows.

### `nets/`

Output storage for legacy saved models and generated figures.

- `replicate_fig1/`: saved example network and figure-reproduction artifacts.

### `project3_generalization/`

The newer modular package. This is the cleanest place to continue development if the goal is to consolidate experiments.

- `__init__.py`: package marker and top-level Project 3 description.
- `hardware.py`: centralized experiment budget/config definitions, runtime phase logging, memory snapshots, and result-directory creation.

#### `project3_generalization/analysis/`

Small plotting and stats helpers for Project 3 summaries.

- `__init__.py`: package marker.
- `figures.py`: plots similarity matrices, transfer-vs-similarity summaries, and learning curves.
- `stats.py`: small stats helpers such as Cohen's d, Pearson correlation, and FDR correction.

#### `project3_generalization/environments/`

Environment definitions and structural-similarity machinery.

- `__init__.py`: package marker.
- `similarity.py`: estimates transition matrices, successor representations, and cross-environment structural similarity.
- `suite_2d.py`: defines the 2D arena library, rollout collection, random walks, validation, and optional visual-observation generation.
- `suite_3d.py`: lightweight 3D arena/navigator/sensory scaffolding for future 3D experiments.

#### `project3_generalization/evaluation/`

Metric code for Project 3.

- `__init__.py`: package marker.
- `metrics.py`: main metrics library, including sRSA, tuning fractions, replay quality, CKA/CERA, SR error, transfer-vs-similarity, and geometry/topology-related measures.
- `topology.py`: persistent-homology and Betti-number helpers.

#### `project3_generalization/experiments/`

Thin CLI launchers. These are the scripts a human runs; they mainly parse args, build configs, call training/evaluation code, and write JSON summaries.

- `run_baselines.py`: single-environment baseline runs for selected 2D arenas.
- `run_curriculum.py`: curriculum training over multiple environments, with similarity-based or random ordering and optional EWC/readout-only controls.
- `run_two_module.py`: curriculum experiment that adds the cortical prior module on top of the hippocampal model.
- `run_ablation.py`: recurrence-strength ablation sweeps in single-environment or curriculum mode.
- `run_3d.py`: lightweight 3D navigator simulation and anisotropy analysis rather than full training.
- `run_hardware_constrained.py`: all-in-one Project 3 launcher that packages baseline/curriculum experiments around a hardware budget config.

#### `project3_generalization/models/`

Model wrappers that bridge the older architecture library into the newer Project 3 API.

- `__init__.py`: package marker.
- `hippocampal_module.py`: main predictive-RNN wrapper with configs, optional visual encoders, rollout loss, device/AMP handling, and batch prediction/training methods.
- `cortical_module.py`: cortical prior module for transfer/two-module experiments; can infer/init recurrent structure from hidden-state sequences.

#### `project3_generalization/training/`

Reusable training loops.

- `__init__.py`: package marker.
- `single_env.py`: baseline training loop, evaluation loop, resource adaptation on OOM/slow batches, and baseline summarization.
- `curriculum.py`: sequential multi-environment training, curriculum ordering, EWC regularization, transfer bookkeeping, and re-exposure analysis.
- `ablations.py`: recurrence-scale ablation harness that reuses baseline/curriculum training code.

#### `project3_generalization/visual_rnn/`

Newest branch for visual observations and richer run logging.

- `__init__.py`: package marker.
- `renderer.py`: tile-map renderer that converts 2D environments into egocentric RGB patches.
- `model.py`: helper to build a visual-friendly `HippocampalConfig`.
- `train.py`: full single-run training pipeline with dashboards, checkpoints, decoding metrics, embeddings, and post-run artifacts.
- `analysis.py`: plotting and post-run report generation for the visual training path.

#### `project3_generalization/results/`

Result storage for the new package.

- `hardware_constrained/`: saved baseline/curriculum outputs, summary JSONs, phase timings, and checkpoints from hardware-budgeted runs.

### `utils/`

Legacy shared library. This folder is still the backbone of the old pipeline, and parts of it are reused by Project 3.

- `_init_.py`: appears intended to be a package init file, but it is misspelled and therefore does not function as the standard `__init__.py`.
- `ActionEncodings.py`: action and head-direction encoding helpers used before feeding actions to the RNN.
- `agent.py`: random-action agents and observation collection utilities for the older environments.
- `Architectures.py`: very large catalog of predictive-RNN architectures and variants; effectively the legacy model zoo.
- `CANNNet.py`: continuous-attractor-network variants and utilities layered onto the predictive-RNN code.
- `CANNtools.py`: helper math for CANN kernels and connectivity matrices.
- `env.py`: old environment creation and viewpoint helpers.
- `figures.py`: figure-generation utilities used by the legacy pipeline.
- `general.py`: generic utility functions for saving/loading, statistics, delay-distance calculations, and conversion helpers.
- `LayerNormRNN.py`: layer-normalized RNN cell implementations.
- `LinearDecoder.py`: linear decoder used for position/head-direction decoding from hidden states.
- `lossFuns.py`: predictive losses and regularization losses such as LPL and MSE variants.
- `predictiveNet.py`: central legacy wrapper class combining model, optimizer, environment library, training, decoding, saving, and analysis hooks.
- `pytorchInits.py`: weight initialization helpers, including CANN-oriented initialization.
- `thetaRNN.py`: theta-cycle-aware recurrent cells and recurrent-layer implementations.

## Which Files Are The Real Hubs?

If you want the smallest set of files that define most of the repo's behavior:

- `utils/predictiveNet.py`: legacy hub
- `utils/Architectures.py`: legacy architecture zoo
- `analysis/OfflineTrajectoryAnalysis.py`: legacy replay/sleep analysis hub
- `analysis/SpatialTuningAnalysis.py`: legacy tuning-analysis hub
- `project3_generalization/environments/suite_2d.py`: new environment hub
- `project3_generalization/models/hippocampal_module.py`: new model hub
- `project3_generalization/training/single_env.py`: new baseline-training hub
- `project3_generalization/training/curriculum.py`: new curriculum hub
- `project3_generalization/evaluation/metrics.py`: new metrics hub
- `project3_generalization/visual_rnn/train.py`: newest visual training hub

## Why The Repo Feels Spread Out

The main readability problem is structural, not scientific:

1. the repo preserves the old monolithic workflow
2. the new Project 3 package was added alongside it rather than replacing it
3. the newest visual-RNN path was then added inside Project 3 as a third execution branch
4. result folders, figure notebooks, shell launchers, and reusable code are all mixed in one repository root

So there are really three experiment-launch styles coexisting:

- old monolithic script-based training: `trainNet.py`
- modular Project 3 CLI experiments: `project3_generalization/experiments/run_*.py`
- visual single-run JSON-config pipeline: `train_single_run.py`

## Suggested Read Order

If you are onboarding a new reader, this is the cleanest path:

1. `README.md`
2. `CODEBASE_MAP.md`
3. `project3_generalization/experiments/run_hardware_constrained.py`
4. `project3_generalization/training/single_env.py`
5. `project3_generalization/training/curriculum.py`
6. `project3_generalization/models/hippocampal_module.py`
7. `project3_generalization/environments/suite_2d.py`
8. `project3_generalization/evaluation/metrics.py`
9. only then go back to `utils/predictiveNet.py` and legacy `analysis/`

If you are reproducing the older paper instead:

1. `trainNet.py`
2. `utils/predictiveNet.py`
3. `utils/Architectures.py`
4. `analysis/SpatialTuningAnalysis.py`
5. `analysis/OfflineTrajectoryAnalysis.py`
6. `FigureScripts/`
