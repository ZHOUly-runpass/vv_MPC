import numpy as np

from r680_safety_planner.perception import ConstantVelocityTracker, EuclideanClusterer, OccupancyGrid2D


def test_cluster_and_track_prediction() -> None:
    rng = np.random.default_rng(3)
    first = np.column_stack([rng.normal(2.0, 0.03, 20), rng.normal(1.0, 0.03, 20), np.zeros(20)])
    clusterer = EuclideanClusterer(tolerance_m=0.15, minimum_points=3)
    observations = clusterer.cluster(first)
    assert len(observations) == 1
    tracker = ConstantVelocityTracker()
    tracker.update(observations, 0.0)
    second = first.copy()
    second[:, 0] += 0.1
    tracker.update(clusterer.cluster(second), 0.1)
    predictions = tracker.predict(1.0, 0.1)
    assert len(predictions) == 1
    predictions[0].validate(points=11)
    assert predictions[0].states[-1, 0] > predictions[0].states[0, 0]


def test_unknown_grid_space_is_conservatively_occupied() -> None:
    grid = OccupancyGrid2D.from_points(np.array([[0.0, 0.0, 0.0]]), 0.1, 2.0, 2.0)
    result = grid.occupied(np.array([[0.0, 0.0], [10.0, 10.0]]))
    assert result.tolist() == [True, True]

