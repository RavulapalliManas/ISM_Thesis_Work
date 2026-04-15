"""Test actual SymmetryArena (uses explicit agent_pos, no place_agent)."""
import warnings; warnings.filterwarnings('ignore')
import sys

LOG = open('/tmp/arena_test_log.txt', 'w')
def log(msg):
    LOG.write(msg + '\n'); LOG.flush()
    print(msg, flush=True)

log("importing arena module...")
from project5_symmetry.environments.arena import make_symmetry_env, get_obs_at, compute_H2
import numpy as np
log("imports done")

log("make_symmetry_env l_shape 18 U=3 F=7...")
env = make_symmetry_env('l_shape', 18, U=3, F=7, seed=0)
log("env created, calling reset...")
obs, _ = env.reset()
log(f"reset done, obs image shape: {obs['image'].shape}")

log("passable positions...")
n = len(env.env.passable_positions)
log(f"  {n} passable tiles")

log("get_obs_at...")
pos = env.env.passable_positions[0]
flat = get_obs_at(env, pos, 0)
log(f"  shape: {flat.shape}, range: {flat.min():.2f}-{flat.max():.2f}")

log("get_frame...")
frame = env.env.get_frame(highlight=False, tile_size=10)
log(f"  frame shape: {frame.shape}")

log("compute_H2...")
h2 = compute_H2(env)
log(f"  H2 mean: {h2['mean']:.3f}, n_states: {h2['n_states']}")

log("square + sym pairs...")
sq = make_symmetry_env('square', 12, U=3, F=7, seed=0)
sq.reset()
pairs = sq.env.precompute_symmetry_pairs()
log(f"  sym pairs: {len(pairs)}")

log("variable F...")
for F in [3, 5, 7]:
    e = make_symmetry_env('square', 18, U=2, F=F, seed=0)
    o, _ = e.reset()
    log(f"  F={F}: {o['image'].shape}, obs_size={F*F*3}")

log("ALL PASSED")
LOG.close()
