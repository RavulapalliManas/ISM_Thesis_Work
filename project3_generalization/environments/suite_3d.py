from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class Obstacle3D:
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class EnvironmentSpec3D:
    env_id: str
    name: str
    bounds: tuple[float, float, float]
    surface_type: str = "volume"
    obstacles: tuple[Obstacle3D, ...] = ()
    lattice_spacing: float | None = None
    rotation_degrees: float = 0.0
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def clip(self, position: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(position, dtype=float), a_min=0.0, a_max=np.asarray(self.bounds, dtype=float))
        return self.project_to_surface(clipped)

    def contains(self, position: np.ndarray) -> bool:
        position = np.asarray(position, dtype=float)
        if np.any(position < 0.0) or np.any(position > np.asarray(self.bounds, dtype=float)):
            return False
        for obstacle in self.obstacles:
            if _inside_obstacle(position, obstacle):
                return False
        return True

    def project_to_surface(self, position: np.ndarray) -> np.ndarray:
        position = np.asarray(position, dtype=float)
        if self.surface_type == "volume":
            return position
        if self.surface_type in {"lattice", "tilted_lattice"}:
            spacing = self.lattice_spacing or 0.125
            if self.surface_type == "tilted_lattice":
                rotation = _rotation_matrix_z(np.deg2rad(self.rotation_degrees))
                centered = position - 0.5 * np.asarray(self.bounds)
                rotated = centered @ rotation.T
                snapped = np.round(rotated / spacing) * spacing
                return np.clip(snapped @ rotation + 0.5 * np.asarray(self.bounds), 0.0, np.asarray(self.bounds))
            return np.clip(np.round(position / spacing) * spacing, 0.0, np.asarray(self.bounds))
        if self.surface_type == "platform":
            z_floor_low = 0.1 * self.bounds[2]
            z_floor_high = 0.7 * self.bounds[2]
            ramp_x = 0.5 * self.bounds[0]
            position = position.copy()
            if position[0] < ramp_x * 0.8:
                position[2] = z_floor_low
            elif position[0] > ramp_x * 1.2:
                position[2] = z_floor_high
            else:
                ratio = (position[0] - ramp_x * 0.8) / max(ramp_x * 0.4, 1e-9)
                position[2] = z_floor_low + ratio * (z_floor_high - z_floor_low)
            return np.clip(position, 0.0, np.asarray(self.bounds))
        return position


def _inside_obstacle(position: np.ndarray, obstacle: Obstacle3D) -> bool:
    center = np.asarray(obstacle.center, dtype=float)
    size = np.asarray(obstacle.size, dtype=float)
    delta = position - center
    if obstacle.kind == "sphere":
        return np.linalg.norm(delta) <= size[0]
    if obstacle.kind == "cuboid":
        return bool(np.all(np.abs(delta) <= size / 2))
    if obstacle.kind == "cylinder":
        radial = np.linalg.norm(delta[:2])
        return radial <= size[0] and abs(delta[2]) <= size[1] / 2
    return False


def _rotation_matrix_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


class BaseNavigator3D:
    def __init__(
        self,
        environment: EnvironmentSpec3D,
        *,
        dt: float = 0.01,
        tau: float = 0.2,
        sigma: float = 0.15,
        seed: int | None = None,
    ):
        self.environment = environment
        self.dt = dt
        self.tau = tau
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.position = 0.5 * np.asarray(environment.bounds, dtype=float)
        self.velocity = np.zeros(3, dtype=float)

    def _ou_step(self, velocity: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(size=3)
        drift = -(velocity / max(self.tau, 1e-9))
        diffusion = sigma * np.sqrt(2.0 / max(self.tau, 1e-9)) * noise
        return velocity + self.dt * drift + np.sqrt(self.dt) * diffusion

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SurfaceNavigator3D(BaseNavigator3D):
    def __init__(
        self,
        environment: EnvironmentSpec3D,
        *,
        alpha: float = 0.25,
        sigma_xy: float = 0.15,
        sigma_z: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(environment, sigma=sigma_xy, **kwargs)
        self.alpha = alpha
        self.sigma_xy = sigma_xy
        self.sigma_z = sigma_xy * alpha if sigma_z is None else sigma_z

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.array([self.sigma_xy, self.sigma_xy, self.sigma_z], dtype=float)
        self.velocity = self._ou_step(self.velocity, sigma)
        self.velocity[2] = self.alpha * self.velocity[2]
        self.position = self.environment.clip(self.position + self.velocity * self.dt)
        return self.position.copy(), self.velocity.copy()


class VolumetricNavigator3D(BaseNavigator3D):
    def step(self) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.array([self.sigma, self.sigma, self.sigma], dtype=float)
        self.velocity = self._ou_step(self.velocity, sigma)
        proposed = self.position + self.velocity * self.dt
        proposed = np.clip(proposed, 0.0, np.asarray(self.environment.bounds))
        while not self.environment.contains(proposed):
            self.velocity *= -0.5
            proposed = np.clip(self.position + self.velocity * self.dt, 0.0, np.asarray(self.environment.bounds))
        self.position = proposed
        return self.position.copy(), self.velocity.copy()


class PlaceCells3D:
    def __init__(
        self,
        environment: EnvironmentSpec3D,
        *,
        n: int = 64,
        sigma: float = 0.08,
        alpha: float = 1.0,
        seed: int = 0,
    ):
        self.environment = environment
        self.n = n
        self.sigma = sigma
        self.alpha = alpha
        rng = np.random.default_rng(seed)
        self.centers = rng.uniform(low=0.0, high=np.asarray(environment.bounds), size=(n, 3))

    def get_state(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        delta = positions[:, None, :] - self.centers[None, :, :]
        delta[..., 2] = delta[..., 2] / max(self.alpha, 1e-9)
        squared_distance = np.sum(delta**2, axis=-1)
        return np.exp(-squared_distance / (2 * self.sigma**2))


class HeadDirectionCells3D:
    def __init__(
        self,
        *,
        n: int = 36,
        angular_sigma: float = np.deg2rad(30),
    ):
        self.n = n
        self.angular_sigma = angular_sigma
        self.preferred_directions = _fibonacci_sphere(n)

    def get_state(self, head_vectors: np.ndarray) -> np.ndarray:
        head_vectors = np.asarray(head_vectors, dtype=float).reshape(-1, 3)
        head_vectors = head_vectors / (np.linalg.norm(head_vectors, axis=1, keepdims=True) + 1e-12)
        cos_angle = np.clip(head_vectors @ self.preferred_directions.T, -1.0, 1.0)
        angles = np.arccos(cos_angle)
        return np.exp(-(angles**2) / (2 * self.angular_sigma**2))


class BoundaryVectorCells3D:
    def __init__(
        self,
        environment: EnvironmentSpec3D,
        *,
        n: int = 32,
        sigma_distance: float = 0.08,
        sigma_angle: float = np.deg2rad(45),
        seed: int = 0,
    ):
        self.environment = environment
        self.n = n
        self.sigma_distance = sigma_distance
        self.sigma_angle = sigma_angle
        rng = np.random.default_rng(seed)
        self.preferred_distances = rng.uniform(0.03, 0.25, size=n)
        self.preferred_directions = _fibonacci_sphere(n)

    def _nearest_boundary(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray(self.environment.bounds, dtype=float)
        distances = np.stack(
            [
                positions[:, 0],
                bounds[0] - positions[:, 0],
                positions[:, 1],
                bounds[1] - positions[:, 1],
                positions[:, 2],
                bounds[2] - positions[:, 2],
            ],
            axis=1,
        )
        normals = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        nearest_idx = np.argmin(distances, axis=1)
        nearest_dist = distances[np.arange(len(positions)), nearest_idx]
        nearest_dir = normals[nearest_idx]
        return nearest_dist, nearest_dir

    def get_state(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        nearest_dist, nearest_dir = self._nearest_boundary(positions)
        nearest_dir = nearest_dir / (np.linalg.norm(nearest_dir, axis=1, keepdims=True) + 1e-12)
        cos_angle = np.clip(nearest_dir @ self.preferred_directions.T, -1.0, 1.0)
        angles = np.arccos(cos_angle)
        distance_term = np.exp(-((nearest_dist[:, None] - self.preferred_distances[None, :]) ** 2) / (2 * self.sigma_distance**2))
        angle_term = np.exp(-(angles**2) / (2 * self.sigma_angle**2))
        return distance_term * angle_term


def _fibonacci_sphere(n: int) -> np.ndarray:
    indices = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def simulate_navigator_3d(
    environment: EnvironmentSpec3D,
    navigator_cls: type[BaseNavigator3D],
    n_steps: int,
    *,
    seed: int = 0,
    **navigator_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    navigator = navigator_cls(environment, seed=seed, **navigator_kwargs)
    positions = np.zeros((n_steps + 1, 3), dtype=float)
    velocities = np.zeros((n_steps, 3), dtype=float)
    positions[0] = navigator.position
    for step in range(n_steps):
        positions[step + 1], velocities[step] = navigator.step()
    return positions, velocities


def collect_rollout_3d(
    environment: EnvironmentSpec3D,
    navigator_cls: type[BaseNavigator3D],
    n_steps: int,
    *,
    seed: int = 0,
    n_place: int = 64,
    n_hd: int = 36,
    n_bvc: int = 32,
    place_alpha: float = 1.0,
    **navigator_kwargs: Any,
) -> dict[str, np.ndarray]:
    positions, velocities = simulate_navigator_3d(environment, navigator_cls, n_steps, seed=seed, **navigator_kwargs)
    place_cells = PlaceCells3D(environment, n=n_place, alpha=place_alpha, seed=seed)
    head_cells = HeadDirectionCells3D(n=n_hd)
    boundary_cells = BoundaryVectorCells3D(environment, n=n_bvc, seed=seed)

    head_vectors = np.vstack([velocities, velocities[-1:]])
    head_vectors = head_vectors / (np.linalg.norm(head_vectors, axis=1, keepdims=True) + 1e-12)
    observations = np.concatenate(
        [
            place_cells.get_state(positions),
            head_cells.get_state(head_vectors),
            boundary_cells.get_state(positions),
        ],
        axis=1,
    )
    return {
        "positions": positions,
        "velocities": velocities,
        "head_vectors": head_vectors,
        "observations": observations,
    }


def build_suite_3d() -> dict[str, EnvironmentSpec3D]:
    return {
        "3D_1_cubic_lattice": EnvironmentSpec3D(
            env_id="3D_1_cubic_lattice",
            name="Cubic Lattice Climbing Frame",
            bounds=(0.5, 0.5, 0.5),
            surface_type="lattice",
            lattice_spacing=0.125,
            notes="Wire-frame cubic lattice for surface-constrained navigation.",
        ),
        "3D_2_volumetric_room": EnvironmentSpec3D(
            env_id="3D_2_volumetric_room",
            name="Volumetric Room",
            bounds=(0.5, 0.5, 0.5),
            surface_type="volume",
            notes="Isotropic volumetric exploration.",
        ),
        "3D_3_tilted_lattice": EnvironmentSpec3D(
            env_id="3D_3_tilted_lattice",
            name="Tilted Cubic Lattice",
            bounds=(0.5, 0.5, 0.5),
            surface_type="tilted_lattice",
            lattice_spacing=0.125,
            rotation_degrees=45.0,
            notes="Lattice rotated relative to gravity axis.",
        ),
        "3D_4_multilevel_platform": EnvironmentSpec3D(
            env_id="3D_4_multilevel_platform",
            name="Multi-level Platform",
            bounds=(0.6, 0.4, 0.4),
            surface_type="platform",
            notes="Two floors connected by a ramp-like bridge.",
        ),
        "3D_5_obstacle_room": EnvironmentSpec3D(
            env_id="3D_5_obstacle_room",
            name="Volumetric Room with Obstacles",
            bounds=(0.5, 0.5, 0.5),
            surface_type="volume",
            obstacles=(
                Obstacle3D("sphere", center=(0.18, 0.18, 0.18), size=(0.06, 0.06, 0.06)),
                Obstacle3D("cuboid", center=(0.34, 0.22, 0.28), size=(0.12, 0.08, 0.12)),
                Obstacle3D("cylinder", center=(0.28, 0.38, 0.22), size=(0.05, 0.18, 0.0)),
            ),
        ),
    }


__all__ = [
    "BaseNavigator3D",
    "BoundaryVectorCells3D",
    "EnvironmentSpec3D",
    "HeadDirectionCells3D",
    "Obstacle3D",
    "PlaceCells3D",
    "SurfaceNavigator3D",
    "VolumetricNavigator3D",
    "build_suite_3d",
    "collect_rollout_3d",
    "simulate_navigator_3d",
]
