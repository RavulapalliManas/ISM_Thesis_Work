from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
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
    ExperimentConfig('P0',     'l_shape', 18, F=7, U=3, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 1 — Symmetry & arena scaling (F=7, U=3, T=200, k=5)
# ---------------------------------------------------------------------------
PHASE1 = [
    ExperimentConfig('P1-ctrl', 'l_shape', 18, F=7, U=3, k=5, T=200),
    ExperimentConfig('P1-A',    'square',  12, F=7, U=3, k=5, T=200),
    ExperimentConfig('P1-B',    'square',  18, F=7, U=3, k=5, T=200),
    ExperimentConfig('P1-C',    'square',  24, F=7, U=3, k=5, T=200),
    ExperimentConfig('P1-D',    'square',  30, F=7, U=3, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 2a — Landmark density sweep (square 18x18, F=7, T=200, k=5)
# ---------------------------------------------------------------------------
PHASE2A = [
    ExperimentConfig('P2a-U0',  'square', 18, F=7, U=0, k=5, T=200),
    ExperimentConfig('P2a-U1',  'square', 18, F=7, U=1, k=5, T=200),
    ExperimentConfig('P2a-U2',  'square', 18, F=7, U=2, k=5, T=200),
    ExperimentConfig('P2a-U3',  'square', 18, F=7, U=3, k=5, T=200),
    ExperimentConfig('P2a-U4',  'square', 18, F=7, U=4, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 2b — View size sweep (square 18x18, U=U*, T=200, k=5)
# U* is determined empirically after Phase 2a; placeholder = 3 until then.
# ---------------------------------------------------------------------------
U_STAR = 3  # UPDATE after Phase 2a analysis

PHASE2B = [
    ExperimentConfig('P2b-F3',  'square', 18, F=3, U=U_STAR, k=5, T=200),
    ExperimentConfig('P2b-F5',  'square', 18, F=5, U=U_STAR, k=5, T=200),
    ExperimentConfig('P2b-F7',  'square', 18, F=7, U=U_STAR, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 4a — Rollout k sweep (k ∈ {1,3,5} × {L-shape, near-transition square})
# "Near-transition square" ~ P1-B (18x18) based on expected Phase 1 results.
# ---------------------------------------------------------------------------
PHASE4A = [
    ExperimentConfig('P4a-Lk1', 'l_shape', 18, F=7, U=3, k=1, T=200),
    ExperimentConfig('P4a-Lk3', 'l_shape', 18, F=7, U=3, k=3, T=200),
    ExperimentConfig('P4a-Lk5', 'l_shape', 18, F=7, U=3, k=5, T=200),
    ExperimentConfig('P4a-Sk1', 'square',  18, F=7, U=3, k=1, T=200),
    ExperimentConfig('P4a-Sk3', 'square',  18, F=7, U=3, k=3, T=200),
    ExperimentConfig('P4a-Sk5', 'square',  18, F=7, U=3, k=5, T=200),
]

# ---------------------------------------------------------------------------
# Phase 4b — Sequence length T sweep (T ∈ {50,200,600} × {L-shape, square})
# ---------------------------------------------------------------------------
PHASE4B = [
    ExperimentConfig('P4b-LT50',  'l_shape', 18, F=7, U=3, k=5, T=50),
    ExperimentConfig('P4b-LT200', 'l_shape', 18, F=7, U=3, k=5, T=200),
    ExperimentConfig('P4b-LT600', 'l_shape', 18, F=7, U=3, k=5, T=600),
    ExperimentConfig('P4b-ST50',  'square',  18, F=7, U=3, k=5, T=50),
    ExperimentConfig('P4b-ST200', 'square',  18, F=7, U=3, k=5, T=200),
    ExperimentConfig('P4b-ST600', 'square',  18, F=7, U=3, k=5, T=600),
]

# ---------------------------------------------------------------------------
# Full list (Phase 0 not included — gate must pass first)
# ---------------------------------------------------------------------------
ALL_CONDITIONS = PHASE1 + PHASE2A + PHASE2B + PHASE4A + PHASE4B

# Index by condition_id for quick lookup
CONDITION_MAP = {c.condition_id: c for c in PHASE0 + ALL_CONDITIONS}
