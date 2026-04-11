"""
File: project3_generalization/visual_rnn/__init__.py

Description:
Package marker for the visual-observation branch of Project 3. This includes
the tile-map renderer, model presets, single-run trainer, and post-run plots.

Role in system:
Supports experiments that replace handcrafted sensory observations with
egocentric RGB patches generated from the 2-D arena geometry.
"""

__all__ = [
    "ExperimentConfig",
    "RunResult",
    "TileMap",
    "TileMapConfig",
    "build_tile_map",
    "build_visual_model_config",
    "generate_post_run_analysis",
    "get_patch",
    "run_single_experiment",
]
