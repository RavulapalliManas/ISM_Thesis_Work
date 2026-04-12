"""Shared environment abstractions for the topology-before-geometry suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class RolloutBatch:
    """Container returned by every environment rollout sampler."""

    observations: np.ndarray
    actions: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    metadata: dict[str, Any]


class BaseTopologyEnv:
    """Common interface implemented by both MiniGrid and RatInABox backends."""

    backend: str = "unknown"

    def __init__(
        self,
        env_name: str,
        *,
        obs_dim: int,
        act_dim: int,
        topology_label: dict[str, int],
        aliasing_level: str,
        aliasing_type: str,
        arena_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.env_name = str(env_name)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.topology_label = dict(topology_label)
        self.aliasing_level = str(aliasing_level)
        self.aliasing_type = str(aliasing_type)
        self.arena_scale = float(arena_scale)
        self.seed = int(seed)
        self.complexity_index = 1.0
        self.geo_euclidean_discrepancy = 1.0
        self._geodesic_graph: nx.Graph | None = None

    def sample_rollout(self, n_steps: int, seed: int | None = None) -> RolloutBatch:
        """Return one sampled rollout in the backend's native arena."""
        raise NotImplementedError

    @property
    def traversable_mask(self) -> np.ndarray:
        """Boolean occupancy grid used for geodesic computations."""
        raise NotImplementedError

    @property
    def coordinate_extent(self) -> tuple[float, float, float, float]:
        """Return min_x, min_y, max_x, max_y for position discretization."""
        raise NotImplementedError

    def is_traversable(self, position: np.ndarray | tuple[float, float]) -> bool:
        """Check whether a position falls on a traversable location."""
        row, col = self.discretize_positions(np.asarray(position, dtype=float)[None, :])[0]
        mask = self.traversable_mask
        return bool(0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col])

    def build_geodesic_graph(self) -> nx.Graph:
        """Construct a 4-neighbor grid graph over traversable cells."""
        if self._geodesic_graph is not None:
            return self._geodesic_graph

        mask = self.traversable_mask.astype(bool)
        min_x, min_y, max_x, max_y = self.coordinate_extent
        step_x = max((max_x - min_x) / max(mask.shape[1] - 1, 1), 1e-6)
        step_y = max((max_y - min_y) / max(mask.shape[0] - 1, 1), 1e-6)
        graph = nx.Graph()
        rows, cols = np.where(mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            graph.add_node((row, col))
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_row = row + d_row
                n_col = col + d_col
                if 0 <= n_row < mask.shape[0] and 0 <= n_col < mask.shape[1] and mask[n_row, n_col]:
                    weight = step_y if d_row != 0 else step_x
                    graph.add_edge((row, col), (n_row, n_col), weight=float(weight))
        self._geodesic_graph = graph
        return graph

    def discretize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Map continuous positions into grid coordinates aligned with traversable_mask."""
        positions = np.asarray(positions, dtype=float)
        min_x, min_y, max_x, max_y = self.coordinate_extent
        mask = self.traversable_mask
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        cols = np.clip(
            np.round((positions[:, 0] - min_x) / width * (mask.shape[1] - 1)).astype(int),
            0,
            mask.shape[1] - 1,
        )
        rows = np.clip(
            np.round((positions[:, 1] - min_y) / height * (mask.shape[0] - 1)).astype(int),
            0,
            mask.shape[0] - 1,
        )
        return np.column_stack([rows, cols])

    def geodesic_distance(self, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """Compute the shortest-path distance between two positions on the traversable graph."""
        graph = self.build_geodesic_graph()
        node_a = tuple(self.discretize_positions(np.asarray(pos_a, dtype=float)[None, :])[0])
        node_b = tuple(self.discretize_positions(np.asarray(pos_b, dtype=float)[None, :])[0])
        try:
            return float(nx.shortest_path_length(graph, node_a, node_b, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")


def compute_complexity_index_from_mask(mask: np.ndarray) -> float:
    """Approximate the isoperimetric-ratio complexity index from a traversable mask."""
    mask = np.asarray(mask, dtype=bool)
    area = float(mask.sum())
    if area <= 0:
        return 0.0

    perimeter = 0.0
    for row, col in zip(*np.where(mask)):
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row = row + d_row
            n_col = col + d_col
            if not (0 <= n_row < mask.shape[0] and 0 <= n_col < mask.shape[1]) or not mask[n_row, n_col]:
                perimeter += 1.0
    return float((perimeter ** 2 / area) / (4.0 * np.pi))
