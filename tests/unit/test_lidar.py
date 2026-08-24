import numpy as np

from r680_safety_planner.interfaces import LidarFrame
from r680_safety_planner.lidar import LidarPreprocessor, transform_points


def test_transform_preserves_extra_fields() -> None:
    points = np.array([[1.0, 2.0, 3.0, 9.0]], dtype=np.float64)
    transform = np.eye(4)
    transform[:3, 3] = [2.0, -1.0, 0.5]
    result = transform_points(points, transform)
    np.testing.assert_allclose(result[0], [3.0, 1.0, 3.5, 9.0])


def test_preprocessor_filters_invalid_zero_self_and_roi() -> None:
    points = np.array(
        [
            [np.nan, 0, 0, 0],
            [0, 0, 0, 0],
            [0.2, 0.2, 0.1, 1],
            [2.0, 0.0, 0.2, 2],
            [20.0, 0.0, 0.2, 3],
        ],
        dtype=np.float64,
    )
    polygon = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    preprocessor = LidarPreprocessor(
        {"x": [-5, 5], "y": [-5, 5], "z": [-1, 2]}, self_polygon=polygon
    )
    output, quality = preprocessor.process(LidarFrame(points, 1.0, fields=("x", "y", "z", "intensity")))
    assert output.points.shape == (1, 4)
    np.testing.assert_allclose(output.points[0, :3], [2.0, 0.0, 0.2])
    assert quality.invalid_points == 1
    assert quality.zero_points == 1
    assert quality.self_points == 1
    assert quality.outside_roi_points == 1

