"""
File: project3_generalization/visual_rnn/renderer.py

Description:
Tile-based renderer that turns a 2-D environment specification into a discrete
RGB map and egocentric visual patches.

Role in system:
This module underlies the visual-input branch of Project 3. It converts
geometry into image-like observations that can be consumed by the CNN-equipped
hippocampal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from project3_generalization.environments.suite_2d import EnvironmentSpec2D


_DEFAULT_FLOOR_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.72, 0.70, 0.63),
    (0.64, 0.69, 0.73),
    (0.76, 0.63, 0.57),
    (0.62, 0.73, 0.63),
)
_DEFAULT_ACCENT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.84, 0.78, 0.36),
    (0.33, 0.54, 0.74),
    (0.82, 0.46, 0.42),
    (0.48, 0.70, 0.59),
)
_DEFAULT_WALL_COLOR = (0.12, 0.12, 0.14)


@dataclass(frozen=True)
class TileMapConfig:
    """Configuration for discretizing a 2-D arena into an RGB tile map."""

    tile_size: float = 0.05
    patch_size: int = 7
    channels: int = 3
    wall_color: tuple[float, float, float] = _DEFAULT_WALL_COLOR
    floor_palette: tuple[tuple[float, float, float], ...] = _DEFAULT_FLOOR_PALETTE
    accent_palette: tuple[tuple[float, float, float], ...] = _DEFAULT_ACCENT_PALETTE
    landmark_radius_tiles: int = 2
    internal_wall_buffer_scale: float = 0.35
    seed: int = 0


@dataclass
class TileMap:
    """Discrete RGB overlay from which egocentric patches are sampled."""

    env_id: str
    extent: tuple[float, float, float, float]
    tile_size: float
    rgb_grid: np.ndarray
    valid_mask: np.ndarray
    landmark_mask: np.ndarray
    wall_mask: np.ndarray
    config: TileMapConfig

    @property
    def height(self) -> int:
        """Return the grid height in tiles."""
        return int(self.rgb_grid.shape[0])

    @property
    def width(self) -> int:
        """Return the grid width in tiles."""
        return int(self.rgb_grid.shape[1])

    @property
    def wall_color(self) -> np.ndarray:
        """Return the wall color as a float32 RGB vector."""
        return np.asarray(self.config.wall_color, dtype=np.float32)

    @property
    def visual_vector_size(self) -> int:
        """Return the flattened size of one egocentric patch."""
        return int(self.config.patch_size * self.config.patch_size * self.config.channels)

    def world_to_index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map world coordinates to tile indices and flag whether they land inside the grid."""
        pts = np.asarray(points, dtype=np.float32)
        left, right, bottom, top = self.extent
        width = max(right - left, 1e-9)
        height = max(top - bottom, 1e-9)
        x_idx = np.floor((pts[:, 0] - left) / width * self.width).astype(int)
        y_idx = np.floor((pts[:, 1] - bottom) / height * self.height).astype(int)
        inside = (
            (x_idx >= 0)
            & (x_idx < self.width)
            & (y_idx >= 0)
            & (y_idx < self.height)
        )
        x_idx = np.clip(x_idx, 0, self.width - 1)
        y_idx = np.clip(y_idx, 0, self.height - 1)
        return x_idx, y_idx, inside

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Sample RGB values at world coordinates, using wall color outside the map."""
        x_idx, y_idx, inside = self.world_to_index(points)
        colors = np.repeat(self.wall_color[None, :], len(points), axis=0)
        if np.any(inside):
            colors[inside] = self.rgb_grid[y_idx[inside], x_idx[inside]]
        return colors.astype(np.float32, copy=False)

    def as_image(self) -> np.ndarray:
        """Return the full tile map in display-ready image coordinates."""
        return np.flipud(self.rgb_grid).astype(np.float32, copy=False)


def _spec_to_polygon(spec: EnvironmentSpec2D) -> Polygon:
    """Convert an environment specification into a cleaned Shapely polygon."""
    polygon = Polygon(spec.boundary, holes=spec.holes).buffer(0)
    if polygon.geom_type != "Polygon":
        raise ValueError(f"Expected polygonal environment for `{spec.env_id}`, got {polygon.geom_type}.")
    return polygon


def _valid_candidate_points(spec: EnvironmentSpec2D, polygon: Polygon) -> list[tuple[float, float]]:
    """Choose landmark candidate locations that are well-spaced and inside the arena."""
    candidates: list[tuple[float, float]] = []
    candidates.extend(tuple(map(float, obj)) for obj in spec.objects)
    candidates.extend(tuple(map(float, zone.center)) for zone in spec.reward_zones)

    left, right, bottom, top = polygon.bounds
    fractional_candidates = (
        (0.20, 0.20),
        (0.80, 0.20),
        (0.20, 0.80),
        (0.80, 0.80),
        (0.50, 0.25),
        (0.25, 0.50),
        (0.75, 0.50),
        (0.50, 0.75),
    )
    for fx, fy in fractional_candidates:
        point = (left + fx * (right - left), bottom + fy * (top - bottom))
        if polygon.covers(Point(point)):
            candidates.append(point)

    representative = polygon.representative_point()
    candidates.append((float(representative.x), float(representative.y)))

    deduped: list[tuple[float, float]] = []
    for point in candidates:
        if not polygon.covers(Point(point)):
            continue
        if all(np.linalg.norm(np.subtract(point, other)) >= 0.1 for other in deduped):
            deduped.append(point)
    return deduped[:4]


def _base_floor_color(ix: int, iy: int, width: int, height: int, config: TileMapConfig) -> np.ndarray:
    """Generate a deterministic textured floor color for one tile."""
    palette = np.asarray(config.floor_palette, dtype=np.float32)
    accents = np.asarray(config.accent_palette, dtype=np.float32)
    macro_zone = ((ix // max(width // 3, 1)) + 2 * (iy // max(height // 3, 1))) % len(palette)
    color = palette[macro_zone].copy()

    if ((ix // 2) + (iy // 2)) % 2 == 0:
        color = 0.82 * color + 0.18 * palette[(macro_zone + 1) % len(palette)]
    if ((ix + 2 * iy) % 7) == 0:
        color = 0.70 * color + 0.30 * accents[(ix // 3 + iy // 3) % len(accents)]
    if ((iy // 4) % 2) == 1 and (ix % 3 == 0):
        color = 0.78 * color + 0.22 * accents[(macro_zone + 2) % len(accents)]

    return np.clip(color, 0.0, 1.0)


def _apply_landmark_pattern(
    rgb_grid: np.ndarray,
    valid_mask: np.ndarray,
    landmark_mask: np.ndarray,
    center_x: int,
    center_y: int,
    pattern_id: int,
    config: TileMapConfig,
) -> None:
    """Stamp a colored landmark motif around one landmark center."""
    accents = np.asarray(config.accent_palette, dtype=np.float32)
    radius = int(max(config.landmark_radius_tiles, 1))
    primary = accents[pattern_id % len(accents)]
    secondary = accents[(pattern_id + 1) % len(accents)]

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x_idx = center_x + dx
            y_idx = center_y + dy
            if x_idx < 0 or x_idx >= rgb_grid.shape[1] or y_idx < 0 or y_idx >= rgb_grid.shape[0]:
                continue
            if not valid_mask[y_idx, x_idx]:
                continue

            manhattan = abs(dx) + abs(dy)
            if pattern_id % 4 == 0:
                active = dx == 0 or dy == 0
                color = primary if active else secondary
            elif pattern_id % 4 == 1:
                active = manhattan <= radius and manhattan != 0
                color = secondary if (dx + dy) % 2 == 0 else primary
            elif pattern_id % 4 == 2:
                active = max(abs(dx), abs(dy)) == radius or manhattan == 0
                color = primary if active else secondary
            else:
                active = abs(dx) == abs(dy) or dx == 0 or dy == 0
                color = secondary if active else primary

            if active:
                rgb_grid[y_idx, x_idx] = color
                landmark_mask[y_idx, x_idx] = True


def build_tile_map(
    spec: EnvironmentSpec2D,
    config: TileMapConfig | Mapping[str, Any] | None = None,
) -> TileMap:
    """Create a tile map that visually encodes the arena boundary, walls, and landmarks."""

    config = TileMapConfig(**dict(config)) if isinstance(config, dict) else (config or TileMapConfig())
    polygon = _spec_to_polygon(spec)
    left, bottom, right, top = polygon.bounds
    width = max(1, int(np.ceil((right - left) / config.tile_size)))
    height = max(1, int(np.ceil((top - bottom) / config.tile_size)))

    rgb_grid = np.repeat(np.asarray(config.wall_color, dtype=np.float32)[None, None, :], height * width, axis=0)
    rgb_grid = rgb_grid.reshape(height, width, config.channels)
    valid_mask = np.zeros((height, width), dtype=bool)
    landmark_mask = np.zeros((height, width), dtype=bool)
    wall_mask = np.ones((height, width), dtype=bool)

    wall_union = None
    if spec.walls:
        wall_geoms = [
            LineString((tuple(start), tuple(end))).buffer(
                config.tile_size * config.internal_wall_buffer_scale,
                cap_style=2,
                join_style=2,
            )
            for start, end in spec.walls
        ]
        wall_union = unary_union(wall_geoms)

    for y_idx in range(height):
        for x_idx in range(width):
            center = (
                left + (x_idx + 0.5) * config.tile_size,
                bottom + (y_idx + 0.5) * config.tile_size,
            )
            point = Point(center)
            inside = polygon.covers(point)
            blocked = bool(wall_union is not None and wall_union.covers(point))
            if not inside or blocked:
                continue
            valid_mask[y_idx, x_idx] = True
            wall_mask[y_idx, x_idx] = False
            rgb_grid[y_idx, x_idx] = _base_floor_color(x_idx, y_idx, width, height, config)

    for landmark_id, point in enumerate(_valid_candidate_points(spec, polygon)):
        px = int(np.floor((point[0] - left) / max(config.tile_size, 1e-9)))
        py = int(np.floor((point[1] - bottom) / max(config.tile_size, 1e-9)))
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))
        _apply_landmark_pattern(rgb_grid, valid_mask, landmark_mask, px, py, landmark_id, config)

    return TileMap(
        env_id=spec.env_id,
        extent=(float(left), float(right), float(bottom), float(top)),
        tile_size=float(config.tile_size),
        rgb_grid=rgb_grid.astype(np.float32, copy=False),
        valid_mask=valid_mask,
        landmark_mask=landmark_mask,
        wall_mask=wall_mask,
        config=config,
    )


def _normalize_heading(head_direction: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize a 2-D heading vector and fall back to +x when ill-defined."""
    heading = np.asarray(head_direction, dtype=np.float32)
    if heading.shape != (2,):
        raise ValueError(f"Expected 2-D head direction, got shape {heading.shape}.")
    norm = float(np.linalg.norm(heading))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return heading / norm


def get_patch_from_state(
    position: Sequence[float] | np.ndarray,
    head_direction: Sequence[float] | np.ndarray,
    tile_map: TileMap,
) -> np.ndarray:
    """Render an egocentric RGB patch from position and heading."""

    position_xy = np.asarray(position, dtype=np.float32)
    if position_xy.shape != (2,):
        raise ValueError(f"Expected 2-D position, got shape {position_xy.shape}.")

    heading = _normalize_heading(head_direction)
    right = np.asarray([heading[1], -heading[0]], dtype=np.float32)

    half_width = tile_map.config.patch_size // 2
    # The bottom-middle pixel corresponds to the agent's current location and heading.
    row_offsets = np.arange(tile_map.config.patch_size - 1, -1, -1, dtype=np.float32)
    col_offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
    lateral, forward = np.meshgrid(col_offsets, row_offsets, indexing="xy")
    sample_points = (
        position_xy[None, None, :]
        + tile_map.tile_size * forward[..., None] * heading[None, None, :]
        + tile_map.tile_size * lateral[..., None] * right[None, None, :]
    )
    patch = tile_map.sample(sample_points.reshape(-1, 2))
    return patch.reshape(tile_map.config.patch_size, tile_map.config.patch_size, tile_map.config.channels)


def get_patch(agent: Any, tile_map: TileMap) -> np.ndarray:
    """Render an egocentric patch from a RatInABox agent-like object."""

    if not hasattr(agent, "pos") or not hasattr(agent, "head_direction"):
        raise AttributeError("Agent must expose `pos` and `head_direction` fields.")
    return get_patch_from_state(agent.pos, agent.head_direction, tile_map)


def flatten_patch(patch: np.ndarray) -> np.ndarray:
    """Flatten a patch into the vector format consumed by the model."""

    return np.asarray(patch, dtype=np.float32).reshape(-1)


__all__ = [
    "TileMap",
    "TileMapConfig",
    "build_tile_map",
    "flatten_patch",
    "get_patch",
    "get_patch_from_state",
]
