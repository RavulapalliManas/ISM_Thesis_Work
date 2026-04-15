import hashlib
import json
import numpy as np

# Action indices
LEFT, RIGHT, FORWARD, STAY = 0, 1, 2, 3

# Heading: 0=North, 1=East, 2=South, 3=West
# Rotation deltas for (row, col): North=(-1,0), East=(0,1), South=(1,0), West=(0,-1)
_HEADING_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

# Rotation transform for C4: (r, c) -> (c, size-1-r) [90° CW]
def _rotate_pos_cw(r, c, size):
    return c, size - 1 - r


class Arena:
    """
    Discrete grid arena for project5_symmetry experiments.

    Parameters
    ----------
    shape : 'square' | 'l_shape'
    size  : int — grid edge length (e.g. 18 for 18x18)
    U     : int — number of distinct landmark colour classes (0 = uniform grey)
    F     : int — visual field edge (3, 5, or 7); must be odd
    seed  : int
    """

    def __init__(self, shape: str, size: int, U: int, F: int, seed: int = 0):
        assert F % 2 == 1, "F must be odd"
        self.shape = shape
        self.size = size
        self.U = U
        self.F = F
        self.seed = seed
        self.obs_size = F * F * 3
        self.rng = np.random.default_rng(seed)

        self.passable = self._make_passable()
        self.tile_rgb = self._make_tile_rgb()
        self.passable_positions = list(zip(*np.where(self.passable)))  # list of (r,c)

    # ------------------------------------------------------------------
    # Passability mask
    # ------------------------------------------------------------------

    def _make_passable(self) -> np.ndarray:
        mask = np.ones((self.size, self.size), dtype=bool)
        if self.shape == 'l_shape':
            cut = self.size // 2
            mask[:cut, cut:] = False
        elif self.shape != 'square':
            raise ValueError(f"Unknown shape: {self.shape}")
        return mask

    # ------------------------------------------------------------------
    # Landmark / tile colour map
    # ------------------------------------------------------------------

    def _make_tile_rgb(self) -> np.ndarray:
        rgb = np.zeros((self.size, self.size, 3), dtype=np.float32)
        if self.U == 0:
            rgb[self.passable] = 0.5
            return rgb

        # Sample U distinct colours
        palette = self.rng.random((self.U, 3)).astype(np.float32)

        # Assign each passable tile a random colour class
        rows, cols = np.where(self.passable)
        assignments = self.rng.integers(0, self.U, size=len(rows))
        rgb[rows, cols] = palette[assignments]
        return rgb

    # ------------------------------------------------------------------
    # Observation rendering
    # ------------------------------------------------------------------

    def get_obs(self, pos: tuple, heading: int) -> np.ndarray:
        """
        Return a flat float32 array of shape (F*F*3,) representing the
        F×F visual field centred on pos, oriented by heading.

        heading: 0=North, 1=East, 2=South, 3=West
        The patch is extracted in world coords then rotated so the agent
        always faces 'up' in the patch.
        """
        r, c = pos
        half = self.F // 2
        pad = half

        # Pad the RGB map with zeros (impassable / wall colour = black)
        padded = np.pad(self.tile_rgb, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        # Agent world position in padded coords
        pr, pc = r + pad, c + pad

        patch = padded[pr - half: pr + half + 1, pc - half: pc + half + 1].copy()

        # Rotate patch so heading faces 'up' (North in patch coords)
        # heading 0 (North)  → no rotation
        # heading 1 (East)   → rotate 90° CCW (k=1 in np.rot90 = CCW)
        # heading 2 (South)  → rotate 180°
        # heading 3 (West)   → rotate 90° CW (k=3)
        k = [0, 1, 2, 3][heading]
        patch = np.rot90(patch, k=k)

        return patch.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # H2 aliasing map
    # ------------------------------------------------------------------

    def compute_H2(self) -> dict:
        """
        For each (pos, heading), count how many other (pos', heading') pairs
        produce an identical observation.

        Returns dict with keys: 'mean', 'distribution' (array of counts),
        'by_state' (list of counts, one per state).
        """
        states = []
        for pos in self.passable_positions:
            for h in range(4):
                states.append((pos, h))

        obs_hashes = []
        for pos, h in states:
            obs = self.get_obs(pos, h)
            obs_hashes.append(hashlib.md5(obs.tobytes()).hexdigest())

        n = len(states)
        counts = np.zeros(n, dtype=np.int32)
        hash_to_indices = {}
        for i, hsh in enumerate(obs_hashes):
            hash_to_indices.setdefault(hsh, []).append(i)

        for indices in hash_to_indices.values():
            c = len(indices) - 1  # exclude self
            for i in indices:
                counts[i] = c

        return {
            'mean': float(counts.mean()),
            'distribution': counts.tolist(),
            'n_passable_tiles': len(self.passable_positions),
            'n_states': n,
        }

    # ------------------------------------------------------------------
    # Symmetry pair precomputation (C4 for squares)
    # ------------------------------------------------------------------

    def precompute_symmetry_pairs(self) -> list:
        """
        For square arenas (C4 symmetry group): return all pairs (pos, R*pos)
        where R is a 90°-CW rotation, for all 3 non-identity rotations.

        Returns list of ((r1,c1), (r2,c2)) tuples (only pairs where BOTH
        positions are passable).
        """
        if self.shape != 'square':
            return []

        passable_set = set(self.passable_positions)
        pairs = []
        for r, c in self.passable_positions:
            pos = (r, c)
            rotated = pos
            for _ in range(3):  # 3 non-identity rotations
                rotated = _rotate_pos_cw(rotated[0], rotated[1], self.size)
                if rotated in passable_set and rotated != pos:
                    pairs.append((pos, rotated))
        return pairs

    def save_metadata(self, path: str):
        """Save arena geometry and H2 to JSON for reproducibility."""
        h2 = self.compute_H2()
        sym_pairs = self.precompute_symmetry_pairs()
        meta = {
            'shape': self.shape,
            'size': self.size,
            'U': self.U,
            'F': self.F,
            'seed': self.seed,
            'n_passable': len(self.passable_positions),
            'H2': h2,
            'symmetry_pairs': [list(a) + list(b) for a, b in sym_pairs],
        }
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def step(self, pos: tuple, heading: int, action: int) -> tuple:
        """
        Apply action and return new (pos, heading).
        Wall collisions keep pos; turning always succeeds.
        """
        if action == LEFT:
            heading = (heading - 1) % 4
        elif action == RIGHT:
            heading = (heading + 1) % 4
        elif action == FORWARD:
            dr, dc = _HEADING_DELTA[heading]
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.passable[nr, nc]:
                pos = (nr, nc)
        # STAY: no change
        return pos, heading
