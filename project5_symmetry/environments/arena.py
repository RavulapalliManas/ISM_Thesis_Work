"""
MiniGrid-based arena for project5_symmetry.

Faithful to Levenstein et al. 2024 (Adrien1.pdf, Methods p.14):
  - 18x18 L-shape (or NxN square) interior, outer boundary walls
  - Floor tiles: grey + up to 4 landmark colour types (red/blue/yellow/green)
  - Observation: 7x7x3 (or FxFx3) egocentric image, tile_size=1
  - Action encoding: SpeedHD 5-dim (speed + 4-way HD one-hot)

Note on gymnasium wrapper compatibility:
  gymnasium.Wrapper does NOT forward __getattr__ to self.env, so
  RGBImgPartialObsWrapper's self.get_frame() call fails on custom envs.
  We use our own PixelObsWrapper that explicitly calls self.env.get_frame().
"""

import hashlib
import json
import numpy as np

import gymnasium
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from minigrid.core.world_object import Floor, Wall
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

# MiniGrid action integers
MG_LEFT, MG_RIGHT, MG_FORWARD = 0, 1, 2

# Landmark colour names for random-U mode (legacy, kept for compatibility)
# MiniGrid renders Floor(name) as COLORS[name]/2 (uint8 in 0–127 range)
_LANDMARK_COLORS = ['grey', 'red', 'blue', 'yellow', 'green']

# ── Paper-faithful landmark colours (Figure 1A, Levenstein et al. 2024) ──────
# Maps normalised RGB [0–1] → MiniGrid Floor color name.
# MiniGrid renders Floor(name) as COLORS[name]/2, so:
#   'blue'   → [0, 0, 255]/2 → ≈ [0,   0,   0.50]  ≈ target [0,    0,    0.45]
#   'red'    → [255, 0, 0]/2 → ≈ [0.50, 0,  0]     ≈ target [0.45, 0,    0   ]
#   'yellow' → [255,255,0]/2 → ≈ [0.50,0.50,0]     ≈ target [0.45, 0.45, 0   ]
#   'grey'   → [100,100,100]/2 → ≈ [0.20,0.20,0.20] (base floor)
_LANDMARK_RGB_TO_COLOR: dict[tuple, str] = {
    (0.0,  0.0,  0.45): 'blue',
    (0.45, 0.0,  0.0):  'red',
    (0.45, 0.45, 0.0):  'yellow',
}

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
    shape         : 'l_shape' | 'square'
    size          : int  — interior tile count per side (18 for the paper baseline)
    U             : int  — number of landmark colour types for random mode (legacy)
    F             : int  — agent_view_size (obs = F×F×3 with tile_size=1)
    seed          : int  — controls random landmark placement (random mode only)
    use_landmarks : bool — if True (default), place the three fixed paper-faithful
                           landmark regions from Figure 1A of Levenstein et al. 2024.
                           Set to False for ablation / U=0 (uniform grey floor).
    """

    def __init__(self, shape: str, size: int, U: int, F: int = 7, seed: int = 0,
                 use_landmarks: bool = True):
        self.arena_shape    = shape
        self.arena_size     = size
        self.U              = U
        self._landmark_seed = seed
        self.use_landmarks  = use_landmarks

        # Build the legacy random landmark map (kept for compatibility)
        self._landmark_map = self._build_landmark_map()

        # ── Validate fixed landmark positions before grid is built ────────────
        if use_landmarks:
            bad = [
                (r, c) for (r, c) in self._get_landmark_tiles()
                if not self._is_passable(r, c)
            ]
            if bad:
                raise ValueError(
                    f"Landmark tile(s) fall on impassable cells in "
                    f"{shape} {size}×{size}: {bad}"
                )

        mission_space = MissionSpace(mission_func=lambda: "explore")
        super().__init__(
            mission_space=mission_space,
            width=size + 2,    # +2 for outer boundary walls
            height=size + 2,
            max_steps=size * size * 100,
            agent_view_size=F,
            # render_mode omitted — triggers Pyglet init and hangs headlessly
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

    def _get_landmark_tiles(self) -> dict:
        """
        Return the fixed landmark tile positions matching Figure 1A of
        Levenstein et al. (2024).

        Returns
        -------
        dict mapping (row, col) → [R, G, B]  (normalised 0–1 floats)
        Row/col are 1-indexed interior coordinates matching _is_passable().

        Three landmark regions (arena = 18×18, cut = 9, bottom-right missing):

                    Blue staircase  — top-left quadrant  (rows 1–9,  cols 1–9)
                        6 steps, step size 1, with a 1-tile boundary (no tiles on row/col 1 or 9):
                            cols 2–7 → heights 2–7 rows  (rows 2–3 up to rows 2–8)

                    Red cross       — top-right quadrant (rows 1–9,  cols 10–18)
                        Plus shape (width 2), symmetric with a 1-tile boundary:
                            horizontal bar: rows 5–6, cols 12–17
                            vertical bar:   rows 3–8, cols 14–15

                    Yellow composite— bottom-left quadrant (rows 10–18, cols 1–9)
                        Square (rows 12–16, cols 3–7) plus L-shaped protrusions
                        at each corner creating the irregular / corner-block boundary.
        """
        BLUE   = [0.0,  0.0,  0.45]
        RED    = [0.45, 0.0,  0.0 ]
        YELLOW = [0.45, 0.45, 0.0 ]

        tiles: dict[tuple[int, int], list[float]] = {}

        # ── Blue staircase (top-left quadrant: rows 1–9, cols 1–9) ───────────
        # 6 steps, step size 1, 1-tile boundary around the quadrant
        stair_row0 = 2
        for i in range(6):
            c = 2 + i
            height = 2 + i  # 2..7 (rows 2..8)
            for r in range(stair_row0, stair_row0 + height):
                tiles[(r, c)] = BLUE

        # ── Red cross (top-right quadrant: rows 1–9, cols 10–18) ─────────────
        # Width 2, symmetric, 1-tile boundary inside the quadrant
        for r in range(5, 7):
            for c in range(12, 18):
                tiles[(r, c)] = RED
        for r in range(3, 9):
            for c in range(14, 16):
                tiles[(r, c)] = RED  # overlapping cells keep same color

        # ── Yellow composite (bottom-left quadrant: rows 10–18, cols 1–9) ────
        # Main body: 5×5 square (2 less per side than the 7×7 baseline)
        main_top, main_left = 12, 3
        main_bottom, main_right = 16, 7
        for r in range(main_top, main_bottom + 1):
            for c in range(main_left, main_right + 1):
                tiles[(r, c)] = YELLOW
        # Corner protrusions (L-shaped blocks at each corner of the main square)
        for pos in [
            (main_top - 1, main_left - 1), (main_top - 1, main_left), (main_top, main_left - 1),
            (main_top - 1, main_right), (main_top - 1, main_right + 1), (main_top, main_right + 1),
            (main_bottom, main_left - 1), (main_bottom + 1, main_left - 1), (main_bottom + 1, main_left),
            (main_bottom, main_right + 1), (main_bottom + 1, main_right + 1), (main_bottom + 1, main_right),
        ]:
            tiles[pos] = YELLOW

        # Force row 2 to be fully grey (no landmarks) as requested.
        tiles = {(r, c): v for (r, c), v in tiles.items() if r != 2}

        # Clip to the actual arena size — landmark positions are specified for
        # the canonical 18×18 design; smaller arenas receive whatever subset
        # of the regions fit within [1, arena_size] × [1, arena_size].
        s = self.arena_size
        tiles = {(r, c): v for (r, c), v in tiles.items()
                 if 1 <= r <= s and 1 <= c <= s}

        return tiles

    def _is_passable(self, row: int, col: int) -> bool:
        """Row, col are 1-indexed interior positions."""
        s = self.arena_size
        if not (1 <= row <= s and 1 <= col <= s):
            return False
        if self.arena_shape == 'l_shape':
            cut = s // 2
            # Remove BOTTOM-RIGHT quadrant: rows cut+1..s, cols cut+1..s
            # (matches Figure 1A — the bottom-right is used for agent view
            #  illustration, not a navigable region)
            if row > cut and col > cut:
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

        # Build the landmark lookup once for this grid generation.
        # Keys are (row, col) 1-indexed; values are [R, G, B] float lists.
        # When use_landmarks=False the dict is empty → all tiles stay grey.
        landmark_tiles: dict[tuple, list] = (
            self._get_landmark_tiles() if self.use_landmarks else {}
        )

        for r in range(1, s + 1):
            for c in range(1, s + 1):
                if self._is_passable(r, c):
                    if landmark_tiles and (r, c) in landmark_tiles:
                        # Override base floor with paper-faithful landmark colour.
                        # _LANDMARK_RGB_TO_COLOR maps the normalised RGB tuple
                        # to the nearest MiniGrid Floor color name so that
                        # Floor.render() returns the correct pixel values.
                        rgb_key = tuple(landmark_tiles[(r, c)])
                        color   = _LANDMARK_RGB_TO_COLOR.get(rgb_key, 'grey')
                    else:
                        color = 'grey'   # base floor
                    # MiniGrid uses (col, row) = (x, y) ordering
                    self.grid.set(c, r, Floor(color))
                else:
                    # Interior walls (L-shape cutout or out-of-bounds)
                    self.grid.set(c, r, Wall())

        # Place agent in the yellow (bottom-left) region, matching Figure 1A.
        # Falls back to the first passable tile if the preferred position is
        # somehow impassable (e.g. non-18×18 or square arena).
        # We do NOT call self.place_agent() — np_random is uninitialised here.
        preferred = [(14, 4), (14, 5), (13, 4), (13, 5), (12, 4)]
        start_r, start_c = next(
            ((r, c) for r, c in preferred if self._is_passable(r, c)),
            next(
                (r, c) for r in range(1, s + 1) for c in range(1, s + 1)
                if self._is_passable(r, c)
            ),
        )
        self.agent_pos = np.array([start_c, start_r])  # MiniGrid (x,y) = (col, row)
        self.agent_dir = 3   # heading North (↑), matching figure

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
# Pixel observation wrapper
# ------------------------------------------------------------------

class PixelObsWrapper(ObservationWrapper):
    """
    Replace the 'image' key in the observation dict with an F×F×3 uint8 RGB
    pixel render of the agent's egocentric view (tile_size=1).

    Why not RGBImgPartialObsWrapper?
      That wrapper calls self.get_frame() inside observation(), but
      gymnasium.Wrapper has no __getattr__ forwarding — so get_frame
      (defined on MiniGridEnv) is unreachable from the wrapper.
      We fix this by explicitly calling self.env.get_frame().
    """

    def __init__(self, env: MiniGridEnv, tile_size: int = 1):
        super().__init__(env)
        self.tile_size = tile_size
        F = env.agent_view_size
        px = F * tile_size
        new_image_space = spaces.Box(
            low=0, high=255, shape=(px, px, 3), dtype=np.uint8
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        # self.env is SymmetryArena — get_frame lives there, not on the wrapper
        rgb = self.env.get_frame(tile_size=self.tile_size, agent_pov=True)
        return {**obs, "image": rgb}


# ------------------------------------------------------------------
# Factory — returns a wrapped env ready for trajectory collection
# ------------------------------------------------------------------

def make_symmetry_env(shape: str, size: int, U: int, F: int = 7, seed: int = 0,
                      use_landmarks: bool = True):
    """
    Create and wrap a SymmetryArena with PixelObsWrapper(tile_size=1).

    Observation: dict with 'image' key, shape (F, F, 3), dtype uint8.
    Flattened obs_size = F * F * 3.

    Parameters
    ----------
    use_landmarks : bool
        True  (default) — fixed paper-faithful landmark tiles (Figure 1A).
        False           — uniform grey floor, equivalent to ablation U=0.
    """
    arena = SymmetryArena(shape=shape, size=size, U=U, F=F, seed=seed,
                          use_landmarks=use_landmarks)
    wrapped = PixelObsWrapper(arena, tile_size=1)
    return wrapped


def get_obs_at(wrapped_env, pos_xy: tuple, heading: int) -> np.ndarray:
    """
    Return the flattened float32 observation (obs_size,) at (col, row), heading.
    heading: 0=right(East), 1=down(South), 2=left(West), 3=up(North) — MiniGrid convention.

    Used for H2 and SCI computation without needing a full episode rollout.
    Uses unwrapped to bypass gymnasium Wrapper attribute forwarding limitation.
    """
    inner = wrapped_env.unwrapped  # SymmetryArena — always the raw MiniGridEnv
    inner.agent_pos = np.array(pos_xy)
    inner.agent_dir = heading
    raw_obs = inner.gen_obs()
    obs_dict = wrapped_env.observation(raw_obs)
    img = obs_dict['image']                        # (F, F, 3) uint8
    return (img.reshape(-1).astype(np.float32) / 255.0)


def compute_H2(wrapped_env) -> dict:
    """
    For each (pos, heading) state, count aliased states with identical observation.

    Returns dict: 'mean', 'distribution' (list), 'n_passable_tiles', 'n_states'.
    """
    inner = wrapped_env.unwrapped
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
    inner = wrapped_env.unwrapped
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
