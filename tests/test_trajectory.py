import numpy as np

from square.trajectory import Trajectory


def test_trajectory_from_two_points():
    start = np.zeros(2)
    goal = np.ones(2)
    traj = Trajectory.from_two_points(start, goal, 10)
    np.testing.assert_almost_equal(start, traj[0])
    np.testing.assert_almost_equal(goal, traj[-1])
