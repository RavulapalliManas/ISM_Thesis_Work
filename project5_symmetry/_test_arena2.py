"""Test SymmetryArena + PixelObsWrapper end-to-end."""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

LOG = open('/tmp/arena_test_log.txt', 'w')
def log(msg):
    LOG.write(msg + '\n'); LOG.flush()
    print(msg, flush=True)

log("importing arena module...")
from project5_symmetry.environments.arena import make_symmetry_env, get_obs_at, compute_H2, PixelObsWrapper
import numpy as np
log("imports done")

# ── 1. Basic env creation and reset ──────────────────────────────────────────
log("make_symmetry_env l_shape 18 U=3 F=7...")
env = make_symmetry_env('l_shape', 18, U=3, F=7, seed=0)
log(f"  wrapper type: {type(env).__name__}")
log(f"  inner type:   {type(env.unwrapped).__name__}")

log("env.reset()...")
obs, _ = env.reset()
log(f"  obs['image'].shape: {obs['image'].shape}")
assert obs['image'].shape == (7, 7, 3), f"Bad shape: {obs['image'].shape}"
log("  PASS")

# ── 2. Passable positions ─────────────────────────────────────────────────────
n = len(env.unwrapped.passable_positions)
log(f"passable tiles (l_shape 18): {n}")
assert n > 0

# ── 3. get_obs_at ─────────────────────────────────────────────────────────────
log("get_obs_at...")
pos = env.unwrapped.passable_positions[0]
flat = get_obs_at(env, pos, 0)
log(f"  shape: {flat.shape}, range: {flat.min():.3f}-{flat.max():.3f}")
assert flat.shape == (7 * 7 * 3,), f"Bad flat shape: {flat.shape}"
assert 0.0 <= flat.min() and flat.max() <= 1.0
log("  PASS")

# ── 4. get_frame (full arena render) ─────────────────────────────────────────
log("get_frame (full arena)...")
frame = env.unwrapped.get_frame(highlight=False, tile_size=10)
log(f"  frame shape: {frame.shape}")
assert frame.ndim == 3 and frame.shape[2] == 3
log("  PASS")

# ── 5. step through env ───────────────────────────────────────────────────────
log("env.step() x10...")
for a in [0, 1, 2, 2, 2, 0, 1, 2, 2, 2]:
    obs2, _, terminated, truncated, _ = env.step(a)
    assert obs2['image'].shape == (7, 7, 3)
log("  PASS")

# ── 6. compute_H2 ─────────────────────────────────────────────────────────────
log("compute_H2 (l_shape 18)...")
h2 = compute_H2(env)
log(f"  H2 mean: {h2['mean']:.3f}, n_states: {h2['n_states']}")
assert h2['n_states'] == n * 4
log("  PASS")

# ── 7. square + symmetry pairs ────────────────────────────────────────────────
log("square 12 + symmetry pairs...")
sq = make_symmetry_env('square', 12, U=3, F=7, seed=0)
sq.reset()
pairs = sq.unwrapped.precompute_symmetry_pairs()
log(f"  sym pairs: {len(pairs)}")
assert len(pairs) > 0
log("  PASS")

# ── 8. variable F ─────────────────────────────────────────────────────────────
log("variable F (3, 5, 7)...")
for F in [3, 5, 7]:
    e = make_symmetry_env('square', 18, U=2, F=F, seed=0)
    o, _ = e.reset()
    assert o['image'].shape == (F, F, 3), f"F={F}: {o['image'].shape}"
    flat = get_obs_at(e, e.unwrapped.passable_positions[0], 0)
    assert flat.shape == (F * F * 3,)
    log(f"  F={F}: obs {o['image'].shape}, flat {flat.shape}  PASS")

# ── 9. collect_trajectory ─────────────────────────────────────────────────────
log("collect_trajectory T=50...")
from project5_symmetry.environments.generate_trajectories import collect_trajectory
env2 = make_symmetry_env('l_shape', 18, U=3, F=7, seed=1)
traj = collect_trajectory(env2, T=50)
log(f"  obs:     {traj['obs'].shape}")
log(f"  act_enc: {traj['act_enc'].shape}")
log(f"  pos:     {traj['pos'].shape}")
log(f"  heading: {traj['heading'].shape}")
assert traj['obs'].shape     == (51, 7*7*3)
assert traj['act_enc'].shape == (50, 5)
assert traj['pos'].shape     == (51, 2)
assert traj['heading'].shape == (51,)
log("  PASS")

log("\n=== ALL TESTS PASSED ===")
LOG.close()
