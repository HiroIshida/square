import matplotlib.pyplot as plt
import numpy as np

from square.optimizer import OptimizationBasedPlanner
from square.trajectory import Trajectory
from square.world import CircleObstacle, SquareWorld

if __name__ == "__main__":
    start = np.ones(2) * 0.05
    goal = np.ones(2) * 0.95
    sdf1 = CircleObstacle(np.array([0.5, 0.6]), 0.3)
    sdf2 = CircleObstacle(np.array([0.2, 0.4]), 0.2)
    sdf3 = CircleObstacle(np.array([0.7, 0.4]), 0.2)

    world = SquareWorld((sdf1, sdf2, sdf3))

    b_min = np.zeros(2)
    b_max = np.ones(2)
    planner = OptimizationBasedPlanner(start, goal, world, b_min, b_max)

    traj_init = Trajectory.from_two_points(start, goal, 20)

    res = planner.solve(traj_init)
    assert res.optim_result.success

    fax = world.visualize()
    res.traj_solution.visualize(fax)
    plt.show()
