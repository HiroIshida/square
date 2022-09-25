import numpy as np

from square.optimizer import OptimizationBasedPlanner, linear_interped_trajectory
from square.world import CircleObstacle

if __name__ == "__main__":
    start = np.ones(2) * 0.05
    goal = np.ones(2) * 0.95
    sdf = CircleObstacle(np.array([0.5, 0.6]), 0.4)
    b_min = np.zeros(2)
    b_max = np.ones(2)
    planner = OptimizationBasedPlanner(start, goal, sdf, b_min, b_max)

    traj_init = linear_interped_trajectory(start, goal, 20)
    res = planner.solve(traj_init)
    assert res.optim_result.success
    for p in res.traj_solution:
        # assert not sdf.is_colliding(p)
        pass

    import matplotlib.pyplot as plt

    arr = np.array(res.traj_solution)
    plt.plot(arr[:, 0], arr[:, 1], "ro-")
    plt.show()
