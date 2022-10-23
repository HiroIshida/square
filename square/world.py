from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from square.trajectory import Trajectory


class Obstacle(ABC):
    def is_colliding(self, pos: np.ndarray) -> bool:
        dist = self.signed_distance(np.expand_dims(pos, axis=0))[0]
        return dist < 0.0

    @abstractmethod
    def signed_distance(self, pos: np.ndarray) -> np.ndarray:
        pass


@dataclass
class CircleObstacle(Obstacle):
    center: np.ndarray
    radius: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum((points - self.center) ** 2, axis=1)) - self.radius
        return dists


@dataclass
class BoxObstacle(Obstacle):
    center: np.ndarray
    width: np.ndarray

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points_from_center = points - self.center
        half_extent = self.width * 0.5

        sd_vals_each_axis = np.abs(points_from_center) - half_extent

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals


@dataclass
class GridSDF:
    itp: RegularGridInterpolator
    b_min: np.ndarray
    b_max: np.ndarray
    n_grid: int

    @classmethod
    def from_mesh(
        cls, sdf_mesh: np.ndarray, b_min: np.ndarray, b_max: np.ndarray, method="linear"
    ) -> "GridSDF":
        nx, ny = sdf_mesh.shape
        assert nx == ny
        xlin = np.linspace(b_min[0], b_max[0], nx)
        ylin = np.linspace(b_min[1], b_max[1], ny)
        X, Y = np.meshgrid(xlin, ylin)
        itp = RegularGridInterpolator((xlin, ylin), sdf_mesh.T, method=method)
        return cls(itp, b_min, b_max, nx)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return self.itp(points)


class SquareWorld:
    b_min: np.ndarray
    b_max: np.ndarray
    obstacle_list: Tuple[Obstacle, ...]

    def __init__(self, obstacle_list: Tuple[Obstacle, ...]):
        self.b_min = np.zeros(2)
        self.b_max = np.ones(2)
        self.obstacle_list = obstacle_list

    def sample(self) -> np.ndarray:
        return self.b_min + (self.b_max - self.b_min) * np.random.rand(2)

    def is_colliding(self, state: np.ndarray) -> bool:
        dist = self.signed_distance(np.expand_dims(state, axis=0))[0]
        return dist < 0.0

    def is_valid_trajectory(self, traj: Trajectory, n_points=40) -> bool:
        traj_resample = traj.resample(n_points)
        for point in traj_resample.numpy():
            if self.is_colliding(point):
                return False
        return True

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        sd_arr_list = [obs.signed_distance(points) for obs in self.obstacle_list]
        sd_vals_union = np.min(np.array(sd_arr_list), axis=0)
        return sd_vals_union

    def _get_mesh_points(self, n_grid: int) -> np.ndarray:
        xlin = np.linspace(self.b_min[0], self.b_max[0], n_grid)
        ylin = np.linspace(self.b_min[1], self.b_max[1], n_grid)
        meshes = np.meshgrid(xlin, ylin)
        meshes_flatten = [mesh.flatten() for mesh in meshes]
        pts = np.array([p for p in zip(*meshes_flatten)])
        return pts

    def get_sdf_mesh(self, n_grid: int) -> np.ndarray:
        pts = self._get_mesh_points(n_grid)
        sdf_mesh = self.signed_distance(pts).reshape((n_grid, n_grid))
        return sdf_mesh

    def get_grid_sdf(self, n_grid: int, method="linear") -> GridSDF:
        sdf_mesh = self.get_sdf_mesh(n_grid)
        return GridSDF.from_mesh(sdf_mesh, self.b_min, self.b_max, method=method)

    def visualize(
        self, n_grid: int = 100, with_contourf: bool = True, fax: Optional[Tuple] = None
    ) -> Tuple:
        pts = self._get_mesh_points(n_grid)
        sdf_mesh = self.signed_distance(pts).reshape((n_grid, n_grid))

        xlin = np.linspace(self.b_min[0], self.b_max[0], n_grid)
        ylin = np.linspace(self.b_min[1], self.b_max[1], n_grid)

        if fax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fax

        if with_contourf:
            ax.contourf(xlin, ylin, sdf_mesh, cmap="summer")
        else:
            ax.contourf(xlin, ylin, sdf_mesh, cmap="bone", levels=[-0.5, 0])
        ax.contour(xlin, ylin, sdf_mesh, cmap="gray", levels=[0.0])
        return fig, ax
