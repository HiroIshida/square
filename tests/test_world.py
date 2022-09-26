import numpy as np

from square.world import CircleObstacle, SquareWorld


def test_grid_sdf():
    sdf1 = CircleObstacle(np.array([0.5, 0.6]), 0.3)
    sdf2 = CircleObstacle(np.array([0.2, 0.4]), 0.2)
    sdf3 = CircleObstacle(np.array([0.7, 0.4]), 0.2)
    world = SquareWorld((sdf1, sdf2, sdf3))

    sdf = world.get_grid_sdf(n_grid=100)

    points = np.array([world.sample() for _ in range(100)])
    sd_ground_truth = world.signed_distance(points)
    sd_approx = sdf.signed_distance(points)
    error = sd_approx - sd_ground_truth

    is_almost_work_well = sum(error > 1e-3) / len(error) < 0.1
    assert is_almost_work_well
