"""
Diagnostic script — two questions:
  1. Is torch.compile() actually triggering?        (dynamo verbose, 10 steps)
  2. Where is the time going per micro-batch?       (fine-grained timing, 5 steps)

Run:
    PYTHONPATH=. python3 project5_symmetry/_diag_compile_timing.py
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np

# ── 1. Turn on Dynamo verbose BEFORE any import of compiled modules ───────────
import torch._dynamo
torch._dynamo.config.verbose = True

from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCell
from project5_symmetry.training.dataset import TrajectoryDataset
from project5_symmetry.training.train import _build_optimizer
from torch.utils.data import DataLoader

# ── Config (matches Phase 0) ──────────────────────────────────────────────────
OBS_SIZE  = 147          # F=7, 7×7×3
ACT_SIZE  = 5
K         = 5
HIDDEN    = 500
ACCUM     = 8
DATA_DIR  = 'project5_symmetry/results/P0/trajectories'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HAS_CUDA  = torch.cuda.is_available()

print(f"Device        : {DEVICE}")
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {HAS_CUDA}")
print()

# ── Model ─────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
model = pRNN_th(obs_size=OBS_SIZE, act_size=ACT_SIZE, k=K,
	                hidden_size=HIDDEN, cell=LayerNormRNNCell,
	                dropp=0.15, neuralTimescale=2, predOffset=0).to(DEVICE)

print("=" * 60)
print("PART 1 — torch.compile() probe (dynamo verbose=True)")
print("=" * 60)

compiled = False
try:
    model.rnn.cell = torch.compile(model.rnn.cell)
    compiled = True
    print(f"torch.compile(model.rnn.cell) returned — "
          f"type: {type(model.rnn.cell).__name__}")
except Exception as e:
    print(f"torch.compile FAILED: {e}")

# ── Data ──────────────────────────────────────────────────────────────────────
if not os.path.isdir(DATA_DIR):
    print(f"\nNOTE: {DATA_DIR} not found — generating 20 trajectories for timing.")
    from project5_symmetry.environments.arena import make_symmetry_env
    from project5_symmetry.environments.generate_trajectories import generate_dataset
    env = make_symmetry_env('l_shape', 18, U=3, F=7, seed=0)
    generate_dataset(env, n_traj=20, T=200, out_dir=DATA_DIR, desc='diag data')
    print()

dataset = TrajectoryDataset(DATA_DIR)
loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                     num_workers=2, prefetch_factor=2,
                     persistent_workers=True)

optimizer = _build_optimizer(model, batch_size=1)

def _inf():
    while True:
        yield from loader

# ── Run 10 micro-batches (1 optimizer step + 2 extra) watching dynamo output ─
print("\n--- Dynamo output begins (watch for 'GRAPH CAPTURED' or 'BREAK') ---\n")
sys.stdout.flush()

micro = 0
for batch in _inf():
    obs_b, act_b, _, _ = batch
    obs_b = obs_b.to(DEVICE)
    act_b = act_b.to(DEVICE)
    pred, _, target = model(obs_b, act_b)
    loss = F.mse_loss(pred, target) / ACCUM
    loss.backward()
    micro += 1
    if micro >= 10:
        break

optimizer.zero_grad(set_to_none=True)
print("\n--- End of 10 micro-batches ---\n")

# ── PART 2 — Fine-grained timing ──────────────────────────────────────────────
print("=" * 60)
print("PART 2 — Fine-grained timing (5 optimizer steps × 8 micro-batches)")
print("=" * 60)
if not HAS_CUDA:
    print("(No CUDA — synchronize() calls are no-ops on CPU; times still valid)")
print()

def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()

t_data = t_forward = t_backward = t_optim = 0.0
step = 0
micro = 0

gen = _inf()
while step < 5:
    # ── data ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    batch = next(gen)
    obs_b, act_b, _, _ = batch
    obs_b = obs_b.to(DEVICE, non_blocking=HAS_CUDA)
    act_b = act_b.to(DEVICE, non_blocking=HAS_CUDA)
    sync()
    t_data += time.perf_counter() - t0

    # ── forward ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pred, _, tgt = model(obs_b, act_b)
    sync()
    t_forward += time.perf_counter() - t0

    # ── backward ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    loss = F.mse_loss(pred, tgt) / ACCUM
    loss.backward()
    sync()
    t_backward += time.perf_counter() - t0

    micro += 1
    if micro == ACCUM:
        # ── optimizer step ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        optimizer.step()
        sync()
        optimizer.zero_grad(set_to_none=True)
        t_optim += time.perf_counter() - t0
        step += 1
        micro = 0

n_micro = 5 * ACCUM   # 40 total micro-batches measured
print(f"Totals over {n_micro} micro-batches ({5} optimizer steps):")
print(f"  Data loading  : {t_data:.3f}s  ({t_data/n_micro*1000:.1f} ms/micro)")
print(f"  Forward pass  : {t_forward:.3f}s  ({t_forward/n_micro*1000:.1f} ms/micro)")
print(f"  Backward pass : {t_backward:.3f}s  ({t_backward/n_micro*1000:.1f} ms/micro)")
print(f"  Optimizer step: {t_optim:.3f}s  ({t_optim/5*1000:.1f} ms/step)")
total = t_data + t_forward + t_backward + t_optim
print(f"  ─────────────────────────────────────────")
print(f"  TOTAL         : {total:.3f}s → {total/5:.3f}s / optimizer step")
print(f"  Projected for 40 000 steps: {total/5*40000/3600:.1f} hours")
print()

# ── dominant cost ─────────────────────────────────────────────────────────────
costs = {
    'data':     t_data,
    'forward':  t_forward,
    'backward': t_backward,
    'optimizer': t_optim,
}
dominant = max(costs, key=costs.get)
print(f"  Dominant cost: {dominant}  ({costs[dominant]/total*100:.0f}% of wall time)")
