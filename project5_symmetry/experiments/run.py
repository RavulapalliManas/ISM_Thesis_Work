#!/usr/bin/env python3
"""
project5_symmetry — single entry-point launcher.

Usage
-----
    python -m project5_symmetry.experiments.run --phase 1
    python project5_symmetry/experiments/run.py --phase 1

Or from repo root (with PYTHONPATH=.):
    python project5_symmetry/experiments/run.py --phase 1

Monitor training in real time
-----------------------------
TensorBoard logs are written to  <out>/<condition_id>/seed_XX/tb/
    tensorboard --logdir project5_symmetry/results
Then open  http://localhost:6006  in a browser.
"""

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from project5_symmetry.experiments.sweep import main

if __name__ == '__main__':
    main()
