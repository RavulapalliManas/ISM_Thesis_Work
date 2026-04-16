#!/usr/bin/env python3
"""
project5_symmetry — single entry-point launcher.

Usage
-----
# Phase 0 only (gate check — L-shape baseline, 1 seed):
    python run_p5.py --phase 0

# Full sweep phases (runs gate first, then specified phase):
    python run_p5.py --phase 1
    python run_p5.py --phase 2a
    python run_p5.py --phase 2b
    python run_p5.py --phase 4a
    python run_p5.py --phase 4b
    python run_p5.py --phase all       # everything

# Skip gate (e.g. already passed):
    python run_p5.py --phase 1 --no-gate

# Custom output directory:
    python run_p5.py --phase 0 --out /path/to/results

Monitor training in real time
------------------------------
TensorBoard logs are written to  <out>/<condition_id>/seed_XX/tb/

    tensorboard --logdir project5_symmetry/results

Then open  http://localhost:6006  in a browser.

Each run writes three scalar streams:
  loss/train           — MSE every optimizer step
  metrics/sRSA_euclidean — Spearman r (cosine hidden vs Euclidean space)
  metrics/sRSA_cityblock — Spearman r (cosine hidden vs CityBlock space)
  metrics/delta_TG       — ΔTG = sRSA_e − sRSA_c (topology-geometry gap)

tqdm bars
---------
  Inner bar  →  training steps for the current (condition, seed)
               postfix shows live loss / sRSA_e / sRSA_c / ΔTG
  Outer bar  →  total (condition × seed) runs in the sweep

File layout after a run
-----------------------
project5_symmetry/results/
  <condition_id>/
    arena_meta.json
    trajectories/
      traj_00000.npz … traj_09999.npz
    seed_00/
      ckpt_5000.pt  ckpt_10000.pt  ckpt_20000.pt  ckpt_40000.pt  ckpt_final.pt
      training_log.json
      metrics.json
      tb/                    ← TensorBoard event files
  all_results_<phase>.json   ← aggregated metrics across all runs
"""

import os
import sys

# Make sure project root is on PYTHONPATH when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project5_symmetry.experiments.sweep import main

if __name__ == '__main__':
    main()
