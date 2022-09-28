import copy
from dataclasses import dataclass, fields
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import OptimizeResult, minimize

from square.protocols import SDFLike
from square.trajectory import Trajectory


@dataclass(frozen=True)
class PlannerConfig:
    n_waypoint: int = 20
    clearance: float = 0.02
    ftol: float = 1e-7
    disp: bool = True
    maxiter: int = 100


@dataclass(frozen=True)
class PlanningResult:
    traj_solution: Trajectory
    success: bool
    status: int
    message: str
    fun: np.ndarray
    jac: np.ndarray
    nit: int

    @classmethod
    def from_optimize_result(cls, res: OptimizeResult) -> "PlanningResult":
        kwargs = {}
        for field in fields(cls):
            key = field.name
            if key == "traj_solution":
                points = res.x.reshape(-1, 2)
                value = Trajectory(list(points))
            else:
                value = res[key]
            kwargs[key] = value
        return cls(**kwargs)  # type: ignore


@dataclass
class OptimizationBasedPlanner:
    start: np.ndarray
    goal: np.ndarray
    sdf: SDFLike
    b_min: Optional[np.ndarray] = None
    b_max: Optional[np.ndarray] = None
    smooth_mat: Optional[np.ndarray] = None
    config: PlannerConfig = PlannerConfig()

    def __post_init__(self):
        self.smooth_mat = self.construct_smoothcost_fullmat(self.config.n_waypoint, 2)

    @staticmethod
    def scipinize(fun: Callable) -> Tuple[Callable, Callable]:
        closure_member = {"jac_cache": None}

        def fun_scipinized(x):
            f, jac = fun(x)
            closure_member["jac_cache"] = jac
            return f

        def fun_scipinized_jac(x):
            return closure_member["jac_cache"]

        return fun_scipinized, fun_scipinized_jac

    @staticmethod
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

        F0 = self.sdf.signed_distance(P0) - self.config.clearance
        Grad0 = np.zeros((self.config.n_waypoint, n_dof))

        for i in range(n_dof):
            P1 = copy.deepcopy(P0)
            P1[:, i] += eps
            F1 = self.sdf.signed_distance(P1) - self.config.clearance
            Grad0[:, i] = (F1 - F0) / eps

        value = F0
        grad = block_diag(*list(Grad0))
        return value, grad

    def solve(self, init_trajectory: Trajectory) -> PlanningResult:
        eq_const_scipy, eq_const_jac_scipy = self.scipinize(self.fun_eq)
        eq_dict = {"type": "eq", "fun": eq_const_scipy, "jac": eq_const_jac_scipy}
        ineq_const_scipy, ineq_const_jac_scipy = self.scipinize(self.fun_ineq)
        ineq_dict = {"type": "ineq", "fun": ineq_const_scipy, "jac": ineq_const_jac_scipy}
        f, jac = self.scipinize(self.fun_objective)

        if self.b_min is not None and self.b_max is not None:
            bounds = (
                list(zip(self.b_min + self.config.clearance, self.b_max - self.config.clearance))
                * self.config.n_waypoint
            )
        else:
            bounds = None

        x_init = init_trajectory.numpy().reshape((-1, 2))
        assert x_init.shape[0] == self.config.n_waypoint

        slsqp_option: Dict = {
            "ftol": self.config.ftol,
            "disp": self.config.disp,
            "maxiter": self.config.maxiter,
        }

        res = minimize(
            f,
            x_init,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=[eq_dict, ineq_dict],
            options=slsqp_option,
        )

        return PlanningResult.from_optimize_result(res)
