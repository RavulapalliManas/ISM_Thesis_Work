# Topology Before Geometry

This package extends the existing predictive-RNN codebase with a reuse-first `project4_topology_before_geometry` layer.

Current implementation priorities:

1. Reuse the legacy rollout predictive RNN through `project4_topology_before_geometry.models.prnn.RolloutPRNN`.
2. Keep MiniGrid as the paper-faithful primary backend and RatInABox as the fallback / non-trivial-topology backend.
3. Track geometry and topology convergence separately through `ConvergenceTracker`.

Key entry points:

- `project4_topology_before_geometry/scripts/run_local.py`
- `project4_topology_before_geometry/scripts/run_remote.py`
- `project4_topology_before_geometry/environments/env_factory.py`

Important note:

The scientific requirement is to validate the `l_shape_standard` rollout baseline before novel experiments. The current code includes the baseline path and a passing smoke test, but the full `8e4`-trial replication run has not been completed yet in this session.
