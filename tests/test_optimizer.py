import copy

import numpy as np

from square.optimizer import OptimizationBasedPlanner
from square.trajectory import Trajectory
from square.world import CircleObstacle


def gradient_test(func, x0, decimal=4):
    f0, grad = func(x0)
    n_dim = len(x0)

    eps = 1e-7
    grad_numerical = np.zeros(n_dim)
    for idx in range(n_dim):
        x1 = copy.copy(x0)
        x1[idx] += eps
        f1, _ = func(x1)
        grad_numerical[idx] = (f1 - f0) / eps

    np.testing.assert_almost_equal(grad, grad_numerical, decimal=decimal)


def jacobian_test(func, x0, decimal=4):
    f0, jac = func(x0)
    n_dim = len(x0)

    eps = 1e-7
    jac_numerical = np.zeros(jac.shape)
    for idx in range(n_dim):
        x1 = copy.copy(x0)
        x1[idx] += eps
        f1, _ = func(x1)
        jac_numerical[:, idx] = (f1 - f0) / eps
    np.testing.assert_almost_equal(jac, jac_numerical, decimal=decimal)


def test_optimization_planner_functions():
    start = np.zeros(2)
    goal = np.ones(2)
    sdf = CircleObstacle(np.array([0.5, 0.5]), 0.3)
    planner = OptimizationBasedPlanner(start, goal, sdf)

    a, b = planner.fun_eq_regular(np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0]))
    assert a.item() == 6

    for _ in range(10):
        n_dim = planner.config.n_waypoint * 2
        x = np.random.rand(n_dim)
        a, b = planner.fun_objective(x)

        gradient_test(planner.fun_objective, x)
        jacobian_test(planner.fun_eq, x)
        jacobian_test(planner.fun_ineq, x)
        jacobian_test(planner.fun_eq_regular, x)


def test_optimization_planner():
    start = np.ones(2) * 0.05
    goal = np.ones(2) * 0.95
    sdf = CircleObstacle(np.array([0.5, 0.6]), 0.4)
    b_min = np.zeros(2)
    b_max = np.ones(2)
    planner = OptimizationBasedPlanner(start, goal, sdf, b_min, b_max)
    traj_init = Trajectory.from_two_points(start, goal, 20)
    res = planner.solve(traj_init)
    assert res.success
    for p in res.traj_solution:
        assert not sdf.is_colliding(p)
        assert np.all(p > b_min)
        assert np.all(p < b_max)
