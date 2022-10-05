import copy
from dataclasses import dataclass, fields
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import Bounds, OptimizeResult, minimize

from square.protocols import SDFLike
from square.trajectory import Trajectory


@dataclass(frozen=True)
class PlannerConfig:
    n_dof: int = 2
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
    progress_cache: Optional[List[Trajectory]] = None

    @classmethod
    def from_optimize_result(
        cls, res: OptimizeResult, progress_cache: Optional[List[Trajectory]] = None
    ) -> "PlanningResult":
        kwargs = {}
        for field in fields(cls):
            key = field.name
            if key == "traj_solution":
                points = res.x.reshape(-1, 2)
                value = Trajectory(list(points))
            elif key in res:
                value = res[key]
            kwargs[key] = value
        kwargs["progress_cache"] = progress_cache
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
        n_dof = self.config.n_dof
        value = np.hstack([self.start - x[:n_dof], self.goal - x[-n_dof:]])
        grad = np.zeros((n_dof * 2, self.config.n_waypoint * n_dof))
        grad[:n_dof, :n_dof] = -np.eye(n_dof)
        grad[-n_dof:, -n_dof:] = -np.eye(n_dof)
        return value, grad

    def fun_eq_regular(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_dof = self.config.n_dof
        # if regular point constraint
        points = x.reshape(-1, n_dof)
        n_point = len(points)
        dist_squared_vec = np.sum((points[1:] - points[:-1]) ** 2, axis=1)
        squared_diff_vec = dist_squared_vec[1:] - dist_squared_vec[:-1]

        grad_list = []
        for i in range(1, n_point - 1):
            grad_partial = np.zeros(n_point * n_dof)
            grad_partial[(i - 1) * n_dof : i * n_dof] = -2 * points[i - 1] + 2 * points[i]
            grad_partial[i * n_dof : (i + 1) * n_dof] = -2 * points[i + 1] + 2 * points[i - 1]
            grad_partial[(i + 1) * n_dof : (i + 2) * n_dof] = -2 * points[i] + 2 * points[i + 1]
            grad_list.append(grad_partial)
        grad = np.stack(grad_list)
        return squared_diff_vec, grad

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

    def solve(self, init_trajectory: Trajectory, cache_progress: bool = False) -> PlanningResult:

        eq_const_scipy, eq_const_jac_scipy = self.scipinize(self.fun_eq)
        eq_dict = {"type": "eq", "fun": eq_const_scipy, "jac": eq_const_jac_scipy}

        ineq_const_scipy, ineq_const_jac_scipy = self.scipinize(self.fun_ineq)
        ineq_dict = {"type": "ineq", "fun": ineq_const_scipy, "jac": ineq_const_jac_scipy}

        if cache_progress:
            progress_cache = []

            def wrap(x: np.ndarray):
                traj = Trajectory(list(x.reshape(-1, 2)))
                progress_cache.append(traj)
                return self.fun_objective(x)

            f, jac = self.scipinize(wrap)
        else:
            progress_cache = None
            f, jac = self.scipinize(self.fun_objective)

        if self.b_min is not None and self.b_max is not None:
            assert len(self.b_min) == len(self.b_max)
            lb = np.tile(self.b_min + self.config.clearance, self.config.n_waypoint)
            ub = np.tile(self.b_max - self.config.clearance, self.config.n_waypoint)
            bounds = Bounds(lb, ub, keep_feasible=True)  # type: ignore
        else:
            bounds = None

        assert len(init_trajectory) == self.config.n_waypoint
        x_init = init_trajectory.numpy().flatten()

        slsqp_option: Dict = {
            "ftol": self.config.ftol,
            "disp": self.config.disp,
            "maxiter": self.config.maxiter - 1,  # somehome scipy iterate +1 more time
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

        plan_result = PlanningResult.from_optimize_result(res, progress_cache)
        assert plan_result.nit <= self.config.maxiter, "{} must be <= {}".format(
            plan_result.nit, self.config.maxiter
        )
        return plan_result
