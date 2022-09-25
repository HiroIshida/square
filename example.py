import numpy as np

from square.rrt import RRT
from square.world import CircleObstacle, SquareWorld

if __name__ == "__main__":
    obstacles = (CircleObstacle(np.array([0.5, 0.5]), 0.4),)
    world = SquareWorld(obstacles)
    rrt = RRT(np.array([0.1, 0.1]), np.array([0.9, 0.9]), world)

    while not rrt.extend():
        pass
    rrt.visualize()
    import matplotlib.pyplot as plt

    plt.show()
