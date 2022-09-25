import numpy as np

from square.trajectory import Trajectory


def test_trajectory_from_two_points():
    start = np.zeros(2)
    goal = np.ones(2)
    traj = Trajectory.from_two_points(start, goal, 10)
    np.testing.assert_almost_equal(start, traj[0])
    np.testing.assert_almost_equal(goal, traj[-1])


def test_trajectory():
    start = np.zeros(2)
    goal = np.ones(2)
    traj = Trajectory.from_two_points(start, goal, 10)
    np.testing.assert_almost_equal(traj.length, np.sqrt(2))

    np.testing.assert_almost_equal(traj.sample_point(0.0), traj[0])
    np.testing.assert_almost_equal(traj.sample_point(0.1), 0.1 * np.ones(2) / np.sqrt(2))
    np.testing.assert_almost_equal(traj.sample_point(0.8), 0.8 * np.ones(2) / np.sqrt(2))
