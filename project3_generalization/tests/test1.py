"""
File: project3_generalization/tests/test1.py

Description:
Minimal manual visualization script for quickly rendering one Project 3 2-D
environment as a tile map.

Role in system:
This is an ad hoc developer-facing sanity check rather than part of the main
training pipeline. It provides a fast way to inspect environment geometry and
renderer output during development.
"""

import matplotlib.pyplot as plt
from project3_generalization.environments.suite_2d import build_suite_2d
from project3_generalization.visual_rnn.renderer import build_tile_map

suite = build_suite_2d()
spec = suite["C3_barrier_gap"]   # try A1_square, B2_t_maze, C3_barrier_gap, D1_annulus
tile_map = build_tile_map(spec)

plt.imshow(tile_map.as_image())
plt.title(spec.env_id)
plt.axis("off")
plt.show()
