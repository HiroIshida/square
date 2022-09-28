from typing import Protocol, Tuple

import numpy as np


class SDFLike(Protocol):
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        ...


class FKLike(Protocol):
    def calculate_fk(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...


class MazeLike(Protocol):
    def sample(self) -> np.ndarray:
        ...

    def is_colliding(self, state: np.ndarray) -> bool:
        ...
