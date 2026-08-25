import numpy as np
import pytest

from r680_safety_planner.interfaces import LidarFrame
from r680_safety_planner.lidar import UniLionPointFieldContract


def test_unilion_fields_are_reordered_and_time_is_not_faked() -> None:
    points = np.array([[4.0, 0.2, 7.0, 1.0, 3.0, -0.5]], dtype=np.float64)
    frame = LidarFrame(
        points=points,
        timestamp_s=1.0,
        fields=("intensity", "time", "ring", "x", "y", "z"),
    )
    output = UniLionPointFieldContract().adapt(frame)
    np.testing.assert_array_equal(output, [[1.0, 3.0, -0.5, 4.0, 7.0]])
    assert output.dtype == np.float32


@pytest.mark.parametrize(
    ("fields", "points", "match"),
    [
        (("x", "y", "z", "intensity"), [[1, 2, 3, 4]], "missing"),
        (("x", "y", "z", "intensity", "ring"), [[1, 2, 3, 4, 16]], "ring"),
        (("x", "y", "z", "intensity", "ring"), [[1, 2, 3, 4, 1.2]], "integer"),
    ],
)
def test_unilion_field_contract_rejects_invalid_input(fields, points, match) -> None:
    frame = LidarFrame(np.asarray(points, dtype=np.float64), 1.0, fields=fields)
    with pytest.raises(ValueError, match=match):
        UniLionPointFieldContract().adapt(frame)


def test_unilion_contract_explains_internal_eleven_dimensions() -> None:
    description = UniLionPointFieldContract().describe()
    assert description["model_fields"] == ["x", "y", "z", "intensity", "ring"]
    assert description["pillar_feature_dimension"] == 11
