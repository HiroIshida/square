import copy
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np
from scipy.linalg import block_diag


class SDFLike(Protocol):
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        ...


def construct_smoothcost_fullmat(
    n_wp: int, n_dof: int, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute A of eq. (17) of IJRR-version (2013) of CHOMP"""

    def construct_smoothcost_mat(n_wp):
        # In CHOMP (2013), squared sum of velocity is computed.
        # In this implementation we compute squared sum of acceralation
        # if you set acc_block * 0.0, vel_block * 1.0, then the trajectory
        # cost is same as the CHOMP one.
        acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        vel_block = np.array([[1, -1], [-1, 1]])
        A_ = np.zeros((n_wp, n_wp))
        for i in [1 + i for i in range(n_wp - 2)]:
            A_[i - 1 : i + 2, i - 1 : i + 2] += acc_block * 1.0
            A_[i - 1 : i + 1, i - 1 : i + 1] += vel_block * 0.0  # do nothing
        return A_

    if weights is None:
        weights = np.ones(n_dof)

    w_mat = np.diag(weights)
    A_ = construct_smoothcost_mat(n_wp)
    A = np.kron(A_, w_mat**2)
    return A


@dataclass(frozen=True)
class PlannerConfig:
    n_waypoint: int = 20


@dataclass
class OptimizationBasedPlanner:
    start: np.ndarray
    goal: np.ndarray
    sdf: SDFLike
    smooth_mat: Optional[np.ndarray] = None
    config: PlannerConfig = PlannerConfig()

    def __post_init__(self):
        self.smooth_mat = construct_smoothcost_fullmat(self.config.n_waypoint, 2)

    def fun_objective(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.smooth_mat is not None
        f = (0.5 * self.smooth_mat.dot(x).dot(x)).item() / self.config.n_waypoint
        grad = self.smooth_mat.dot(x) / self.config.n_waypoint
        return f, grad

    def fun_eq(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_dof = 2
        value = np.hstack([self.start - x[:n_dof], self.goal - x[-n_dof:]])
        grad = np.zeros((n_dof * 2, self.config.n_waypoint * n_dof))
        grad[:n_dof, :n_dof] = -np.eye(n_dof)
        grad[-n_dof:, -n_dof:] = -np.eye(n_dof)
        return value, grad

    def fun_ineq(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_dof = 2
        eps = 1e-7
        P0 = x.reshape((self.config.n_waypoint, n_dof))

        F0 = self.sdf.signed_distance(P0)
        Grad0 = np.zeros((self.config.n_waypoint, n_dof))

        for i in range(n_dof):
            P1 = copy.deepcopy(P0)
            P1[:, i] += eps
            F1 = self.sdf.signed_distance(P1)
            Grad0[:, i] = (F1 - F0) / eps

        value = F0
        grad = block_diag(*list(Grad0))
        return value, grad
