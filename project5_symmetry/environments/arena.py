"""
MiniGrid-based arena for project5_symmetry.

Faithful to Levenstein et al. 2024 (Adrien1.pdf, Methods p.14):
  - 18x18 L-shape (or NxN square) interior, outer boundary walls
  - Floor tiles: grey + up to 4 landmark colour types (red/blue/yellow/green)
  - Observation: 7x7x3 (or FxFx3) egocentric image, tile_size=1
  - Action encoding: SpeedHD 5-dim (speed + 4-way HD one-hot)
"""

import hashlib
import json
import numpy as np

import gym
from gym_minigrid.minigrid import MiniGridEnv, Grid, Floor, Wall, MissionSpace
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

# MiniGrid action integers
MG_LEFT, MG_RIGHT, MG_FORWARD = 0, 1, 2

# Paper landmark colours (all map to existing gym_minigrid COLORS)
# Rendered as COLORS[name]/2, approximating the paper's 0-1 values
# grey→[0.196,0.196,0.196] (plain floor), red→[0.5,0,0], blue→[0,0,0.5],
# yellow→[0.5,0.5,0], green→[0,0.5,0]
_LANDMARK_COLORS = ['grey', 'red', 'blue', 'yellow', 'green']

# Action indices (gym_minigrid)
LEFT, RIGHT, FORWARD = 0, 1, 2
# Action 3 (pickup) used as no-op "stop" — has no effect on plain floor
STOP = 3

# Paper action probabilities (Methods, p.14)
PAPER_ACTION_PROBS = [0.15, 0.15, 0.60, 0.10, 0.0, 0.0, 0.0]


class SymmetryArena(MiniGridEnv):
    """
    Custom MiniGrid environment for the project5_symmetry sweep.

    Parameters
    ----------
    shape : 'l_shape' | 'square'
    size  : int  — interior tile count per side (e.g. 18 for the 18x18 L-shape)
    U     : int  — number of landmark colour types (0 = uniform grey floor)
    F     : int  — agent_view_size (obs = F×F×3 with tile_size=1)
    seed  : int  — controls random landmark placement
    """

    def __init__(self, shape: str, size: int, U: int, F: int = 7, seed: int = 0):
        self.arena_shape = shape
        self.arena_size = size
        self.U = U
        self._landmark_seed = seed

        # Build the landmark assignment before super().__init__ calls _gen_grid
        self._landmark_map = self._build_landmark_map()

        mission_space = MissionSpace(mission_func=lambda: "explore")
        super().__init__(
            mission_space=mission_space,
            width=size + 2,    # +2 for outer boundary walls
            height=size + 2,
            max_steps=size * size * 100,
            agent_view_size=F,
            # render_mode omitted: RGBImgPartialObsWrapper uses gen_obs() directly,
            # not render(). Passing 'rgb_array' here triggers Pyglet init and hangs.
        )

    # ------------------------------------------------------------------
    # Landmark map (pre-computed, deterministic per seed)
    # ------------------------------------------------------------------

    def _build_landmark_map(self) -> np.ndarray:
        """
        Return an (size+2, size+2) int array: -1=wall/impassable, 0..U-1=landmark type.
        Uses (row, col) indexing where interior rows/cols are 1..size.
        """
        s = self.arena_size
        lmap = np.full((s + 2, s + 2), -1, dtype=np.int32)

        rng = np.random.default_rng(self._landmark_seed)
        for r in range(1, s + 1):
            for c in range(1, s + 1):
                if self._is_passable(r, c):
                    lmap[r, c] = int(rng.integers(0, max(1, self.U)))
        return lmap

    def _is_passable(self, row: int, col: int) -> bool:
        """Row, col are 1-indexed interior positions."""
        s = self.arena_size
        if not (1 <= row <= s and 1 <= col <= s):
            return False
        if self.arena_shape == 'l_shape':
            cut = s // 2
            # Remove top-right quadrant: rows 1..cut, cols cut+1..s
            if row <= cut and col > cut:
                return False
        return True

    # ------------------------------------------------------------------
    # MiniGrid grid generation
    # ------------------------------------------------------------------

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Outer boundary walls
        self.grid.wall_rect(0, 0, width, height)

        s = self.arena_size
        for r in range(1, s + 1):
            for c in range(1, s + 1):
                if self._is_passable(r, c):
                    ltype = self._landmark_map[r, c]
                    if self.U == 0 or ltype == 0:
                        color = 'grey'
                    else:
                        color = _LANDMARK_COLORS[ltype]   # 1→red, 2→blue, 3→yellow, 4→green
                    # MiniGrid uses (col, row) = (x, y) ordering
                    self.grid.set(c, r, Floor(color))
                else:
                    # Interior walls (L-shape cutout)
                    self.grid.set(c, r, Wall())

        # Set agent to first passable tile deterministically.
        # We do NOT call self.place_agent() here because it uses
        # self.np_random which is only initialised after super().reset()
        # returns — calling it from _gen_grid causes an infinite loop.
        first_r, first_c = next(
            (r, c) for r in range(1, s + 1) for c in range(1, s + 1)
            if self._is_passable(r, c)
        )
        self.agent_pos = np.array([first_c, first_r])  # MiniGrid (x,y) = (col, row)
        self.agent_dir = 0

    # ------------------------------------------------------------------
    # Passable positions list (for trajectory generation and H2)
    # ------------------------------------------------------------------

    @property
    def passable_positions(self) -> list:
        """List of (col, row) tuples — MiniGrid (x, y) coords — for passable tiles."""
        s = self.arena_size
        return [(c, r) for r in range(1, s + 1) for c in range(1, s + 1)
                if self._is_passable(r, c)]

    # ------------------------------------------------------------------
    # Symmetry pair precomputation (C4 for squares)
    # ------------------------------------------------------------------

    def precompute_symmetry_pairs(self) -> list:
        """
        C4 symmetry group (squares only): return all (pos, R*pos) pairs
        where R is 90° CW rotation.  Both endpoints must be passable.

        Returns list of ((c1,r1), (c2,r2)) tuples in MiniGrid (x,y) coords.
        """
        if self.arena_shape != 'square':
            return []

        s = self.arena_size
        # C4 rotation on interior (1-indexed): (c, r) → (s+1-r, c)
        def rot_cw(c, r):
            return s + 1 - r, c

        passable_set = set(self.passable_positions)
        pairs = []
        for pos in self.passable_positions:
            rotated = pos
            for _ in range(3):
                rotated = rot_cw(*rotated)
                if rotated in passable_set and rotated != pos:
                    pairs.append((pos, rotated))
        return pairs


# ------------------------------------------------------------------
# Factory — returns a wrapped env ready for RandomActionAgent
# ------------------------------------------------------------------

def make_symmetry_env(shape: str, size: int, U: int, F: int = 7, seed: int = 0):
    """
    Create and wrap a SymmetryArena as expected by utils/agent.py.

    Returns RGBImgPartialObsWrapper(arena, tile_size=1).
    Observation: dict with 'image' key, shape (F, F, 3), dtype uint8.
    Flattened obs_size = F * F * 3.
    """
    arena = SymmetryArena(shape=shape, size=size, U=U, F=F, seed=seed)
    wrapped = RGBImgPartialObsWrapper(arena, tile_size=1)
    return wrapped


def get_obs_at(wrapped_env, pos_xy: tuple, heading: int) -> np.ndarray:
    """
    Return the flattened float32 observation (obs_size,) at (col, row), heading.
    heading: 0=right(East), 1=down(South), 2=left(West), 3=up(North) — MiniGrid convention.

    Used for H2 and SCI computation without needing a full episode rollout.
    """
    inner = wrapped_env.env   # SymmetryArena (1 wrap level with direct instantiation)
    inner.agent_pos = np.array(pos_xy)
    inner.agent_dir = heading
    raw_obs = inner.gen_obs()
    obs_dict = wrapped_env.observation(raw_obs)
    img = obs_dict['image']                       # (F, F, 3) uint8
    return (img.reshape(-1).astype(np.float32) / 255.0)


def compute_H2(wrapped_env) -> dict:
    """
    For each (pos, heading) state, count aliased states with identical observation.

    Returns dict: 'mean', 'distribution' (list), 'n_passable_tiles', 'n_states'.
    """
    inner = wrapped_env.env
    inner.reset()

    states = []
    for pos in inner.passable_positions:
        for h in range(4):
            states.append((pos, h))

    hashes = [
        hashlib.md5(get_obs_at(wrapped_env, pos, h).tobytes()).hexdigest()
        for pos, h in states
    ]

    hash_to_indices = {}
    for i, hsh in enumerate(hashes):
        hash_to_indices.setdefault(hsh, []).append(i)

    counts = np.zeros(len(states), dtype=np.int32)
    for indices in hash_to_indices.values():
        c = len(indices) - 1
        for i in indices:
            counts[i] = c

    return {
        'mean': float(counts.mean()),
        'distribution': counts.tolist(),
        'n_passable_tiles': len(inner.passable_positions),
        'n_states': len(states),
    }


def save_arena_metadata(wrapped_env, path: str):
    """Serialize arena geometry, H2, and symmetry pairs to JSON."""
    inner = wrapped_env.env
    h2 = compute_H2(wrapped_env)
    sym_pairs = inner.precompute_symmetry_pairs()
    meta = {
        'shape': inner.arena_shape,
        'size': inner.arena_size,
        'U': inner.U,
        'F': inner.agent_view_size,
        'seed': inner._landmark_seed,
        'n_passable': len(inner.passable_positions),
        'H2': h2,
        'symmetry_pairs': [list(a) + list(b) for a, b in sym_pairs],
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
