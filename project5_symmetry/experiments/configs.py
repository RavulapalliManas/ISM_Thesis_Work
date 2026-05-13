"""
Experiment configuration definitions for project5_symmetry.

Defines SymmetryExperimentConfig dataclass and pre-built experiment
phase lists (PHASE0–PHASE4B) for the phase-based parameter sweep
(run via project5_symmetry/experiments/run.py or run_fast.py).

For symmetry-group-order experiments (s4/s2/s1), use the separate
entry point at project5_symmetry/experiments/run_sweep.py.

Phase summary:
    PHASE0   — Baseline gate (L-shape 18×18, F=7, U=3, k=5, T=200)
    PHASE1   — Arena scaling (L-shape + square 12/18/24/30)
    PHASE2A  — Landmark density sweep (U=0..4)
    PHASE2B  — View field size (F=3,5,7 at optimal U)
    PHASE4A  — Rollout horizon sweep (k=1,3,5)
    PHASE4B  — Sequence length sweep (T=50,200,600)

Note: Phase 3 was reserved but unused. Numbers jump 2→4.
"""

from dataclasses import dataclass, field


@dataclass
class SymmetryExperimentConfig:
    condition_id: str
    arena_shape: str    # 'square' | 'l_shape'
    arena_size: int     # grid edge length
    F: int              # visual field edge (3, 5, 7)
    U: int              # landmark colour classes (0-4)
    k: int              # rollout steps
    T: int              # sequence length
    n_seeds: int = 9
    n_traj: int = 10000
    B: int = 8
    n_steps: int = 80000


# ---------------------------------------------------------------------------
# Phase 0 — Baseline gate (must reach sRSA_euclid > 0.4 before any sweep)
# ---------------------------------------------------------------------------
PHASE0 = [
    SymmetryExperimentConfig('P0',     'l_shape', 18, F=7, U=3, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 1 — Symmetry & arena scaling (F=7, U=3, T=200, k=5)
# ---------------------------------------------------------------------------
PHASE1 = [
    SymmetryExperimentConfig('P1-ctrl', 'l_shape', 18, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P1-A',    'square',  12, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P1-B',    'square',  18, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P1-C',    'square',  24, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P1-D',    'square',  30, F=7, U=3, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 2a — Landmark density sweep (square 18x18, F=7, T=200, k=5)
# ---------------------------------------------------------------------------
PHASE2A = [
    SymmetryExperimentConfig('P2a-U0',  'square', 18, F=7, U=0, k=5, T=200),
    SymmetryExperimentConfig('P2a-U1',  'square', 18, F=7, U=1, k=5, T=200),
    SymmetryExperimentConfig('P2a-U2',  'square', 18, F=7, U=2, k=5, T=200),
    SymmetryExperimentConfig('P2a-U3',  'square', 18, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P2a-U4',  'square', 18, F=7, U=4, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 2b — View size sweep (square 18x18, U=U*, T=200, k=5)
# U* is determined empirically after Phase 2a; placeholder = 3 until then.
# ---------------------------------------------------------------------------
U_STAR = 3  # UPDATE after Phase 2a analysis

PHASE2B = [
    SymmetryExperimentConfig('P2b-F3',  'square', 18, F=3, U=U_STAR, k=5, T=200),
    SymmetryExperimentConfig('P2b-F5',  'square', 18, F=5, U=U_STAR, k=5, T=200),
    SymmetryExperimentConfig('P2b-F7',  'square', 18, F=7, U=U_STAR, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 4a — Rollout k sweep (k ∈ {1,3,5} × {L-shape, near-transition square})
# ---------------------------------------------------------------------------
PHASE4A = [
    SymmetryExperimentConfig('P4a-Lk1', 'l_shape', 18, F=7, U=3, k=1, T=200),
    SymmetryExperimentConfig('P4a-Lk3', 'l_shape', 18, F=7, U=3, k=3, T=200),
    SymmetryExperimentConfig('P4a-Lk5', 'l_shape', 18, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P4a-Sk1', 'square',  18, F=7, U=3, k=1, T=200),
    SymmetryExperimentConfig('P4a-Sk3', 'square',  18, F=7, U=3, k=3, T=200),
    SymmetryExperimentConfig('P4a-Sk5', 'square',  18, F=7, U=3, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 4b — Sequence length T sweep (T ∈ {50,200,600} × {L-shape, square})
# ---------------------------------------------------------------------------
PHASE4B = [
    SymmetryExperimentConfig('P4b-LT50',  'l_shape', 18, F=7, U=3, k=5, T=50),
    SymmetryExperimentConfig('P4b-LT200', 'l_shape', 18, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P4b-LT600', 'l_shape', 18, F=7, U=3, k=5, T=600),
    SymmetryExperimentConfig('P4b-ST50',  'square',  18, F=7, U=3, k=5, T=50),
    SymmetryExperimentConfig('P4b-ST200', 'square',  18, F=7, U=3, k=5, T=200),
    SymmetryExperimentConfig('P4b-ST600', 'square',  18, F=7, U=3, k=5, T=600),
]

# ---------------------------------------------------------------------------
# Full list (Phase 0 not included — gate must pass first)
# ---------------------------------------------------------------------------
ALL_CONDITIONS = PHASE1 + PHASE2A + PHASE2B + PHASE4A + PHASE4B

# Index by condition_id for quick lookup
CONDITION_MAP: dict[str, SymmetryExperimentConfig] = {c.condition_id: c for c in PHASE0 + ALL_CONDITIONS}
