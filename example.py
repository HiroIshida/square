import matplotlib.pyplot as plt
import numpy as np

from square.optimizer import OptimizationBasedPlanner, linear_interped_trajectory
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

    traj_init = linear_interped_trajectory(start, goal, 20)
    res = planner.solve(traj_init)
    assert res.optim_result.success

    fig, ax = world.visualize()

    arr = np.array(res.traj_solution)
    ax.plot(arr[:, 0], arr[:, 1], "ro-")
    plt.show()
