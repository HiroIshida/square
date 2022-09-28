from dataclasses import dataclass
from typing import List, Tuple, overload

import numpy as np


class InvalidSamplePointError(Exception):
    pass


@dataclass
class Trajectory:
    _points: List[np.ndarray]

    @property
    def length(self) -> float:
        n_point = len(self._points)
        total = 0.0
        for i in range(n_point - 1):
            p0 = self._points[i]
            p1 = self._points[i + 1]
            total += float(np.linalg.norm(p1 - p0))
        return total

    def sample_point(self, dist_from_start: float) -> np.ndarray:

        if dist_from_start > self.length + 1e-6:
            raise InvalidSamplePointError()

        dist_from_start = min(dist_from_start, self.length)
        edge_dist_sum = 0.0
        for i in range(len(self)):
            edge_dist_sum += float(np.linalg.norm(self._points[i + 1] - self._points[i]))
            if dist_from_start <= edge_dist_sum:
                diff = edge_dist_sum - dist_from_start
                vec_to_prev = self._points[i] - self._points[i + 1]
                vec_to_prev_unit = vec_to_prev / np.linalg.norm(vec_to_prev)
                point_new = self._points[i + 1] + vec_to_prev_unit * diff
                return point_new
        assert False

    def resample(self, n_waypoint: int) -> "Trajectory":
        # yeah, it's inefficient. n^2 instead of n ...
        point_new_list = []
        partial_length = self.length / (n_waypoint - 1)
        for i in range(n_waypoint):
            dist_from_start = partial_length * i
            point_new = self.sample_point(dist_from_start)
            point_new_list.append(point_new)
        return Trajectory(point_new_list)

    def numpy(self):
        return np.array(self._points)

    def visualize(self, fax: Tuple, *args, **kwargs) -> None:
        fig, ax = fax
        arr = self.numpy()
        ax.plot(arr[:, 0], arr[:, 1], *args, **kwargs)

    @classmethod
    def from_two_points(cls, start: np.ndarray, goal: np.ndarray, n_waypoint) -> "Trajectory":
        diff = goal - start
        points = [start + diff / (n_waypoint - 1) * i for i in range(n_waypoint)]
        return cls(points)

    @overload
    def __getitem__(self, indices: List[int]) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, indices: slice) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        pass

    def __getitem__(self, indices_like):
        points = self._points
        return points[indices_like]  # type: ignore

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return self._points.__iter__()
