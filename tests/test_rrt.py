import numpy as np

from square.rrt import RRT
from square.world import CircleObstacle, SquareWorld


def test_rrt():
    start = np.ones(2) * 0.05
    goal = np.ones(2) * 0.95
    CircleObstacle(np.array([0.5, 0.6]), 0.3)
    sdf2 = CircleObstacle(np.array([0.2, 0.4]), 0.2)
    CircleObstacle(np.array([0.7, 0.4]), 0.2)

    world = SquareWorld((sdf2,))
    rrt = RRT(start, goal, world)
    traj_rrt = rrt.solve()
    np.testing.assert_almost_equal(traj_rrt[0], start)
    np.testing.assert_almost_equal(traj_rrt[-1], goal)

    for point in traj_rrt:
        assert not world.is_colliding(point)
