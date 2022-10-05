import matplotlib.pyplot as plt
import numpy as np

from square.optimizer import OptimizationBasedPlanner
from square.rrt import RRT
from square.world import CircleObstacle, SquareWorld

if __name__ == "__main__":
    start = np.ones(2) * 0.05
    goal = np.array([0.05, 0.95])
    sdf1 = CircleObstacle(np.array([0.5, 0.6]), 0.3)
    sdf2 = CircleObstacle(np.array([0.2, 0.4]), 0.2)
    sdf3 = CircleObstacle(np.array([0.7, 0.4]), 0.2)

    world = SquareWorld((sdf1, sdf2, sdf3))

    # creat initial trajectory (solution for optim) by rrt
    rrt = RRT(start, goal, world)
    traj_rrt = rrt.solve()
    assert traj_rrt is not None
    traj_init = traj_rrt.resample(20)

    # solve optimization to plan a trajectory
    planner = OptimizationBasedPlanner(start, goal, world, world.b_min, world.b_max)
    res = planner.solve(traj_init, cache_progress=True)
    assert res.success

    fax = world.visualize()

    for traj in res.progress_cache:
        traj.visualize(fax, "k-", lw=0.3)

    traj_init.visualize(fax, "bo-")
    res.traj_solution.visualize(fax, "ro-")

    plt.show()
