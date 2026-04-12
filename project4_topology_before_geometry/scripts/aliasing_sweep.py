#!/usr/bin/env python3
"""Preset sweep definitions for aliasing and geometry benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_sweeps() -> dict[str, list[dict[str, str]]]:
    """Return compute-aware sweep definitions for the aliasing-controlled suite."""
    return {
        "base_alias": [{"env_type": alias_name} for alias_name in ("zero_alias", "low_alias", "medium_alias", "high_alias", "maximum_alias")],
        "geometry_sweep": [
            {"env_type": geometry_name}
            for geometry_name in ("square", "two_room", "l_shape", "maze_simple")
        ],
        "geometry_alias_cross": [
            {"env_type": f"{geometry_name}_{alias_name}"}
            for geometry_name in ("two_room", "l_shape")
            for alias_name in ("low_alias", "high_alias")
        ],
        "failure_modes": [
            {"env_type": env_name}
            for env_name in ("symmetry_trap", "long_corridor_alias", "ambiguous_junctions", "perceptual_alias_maze")
        ],
    }


def main() -> None:
    print(json.dumps(build_sweeps(), indent=2))


if __name__ == "__main__":
    main()

