from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..interfaces import LidarFrame


@dataclass(frozen=True)
class PointCloudQuality:
    input_points: int
    output_points: int
    invalid_points: int
    zero_points: int
    self_points: int
    outside_roi_points: int

    @property
    def retained_fraction(self) -> float:
        return self.output_points / max(self.input_points, 1)

    def as_dict(self) -> dict[str, float]:
        return {
            "input_points": float(self.input_points),
            "output_points": float(self.output_points),
            "invalid_points": float(self.invalid_points),
            "zero_points": float(self.zero_points),
            "self_points": float(self.self_points),
            "outside_roi_points": float(self.outside_roi_points),
            "retained_fraction": self.retained_fraction,
        }


def transform_points(points: NDArray[np.float64], transform: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a 4x4 rigid transform while preserving non-XYZ fields."""
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4,4]")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N,D], D>=3")
    xyz_h = np.concatenate(
        [points[:, :3], np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    result = points.copy()
    result[:, :3] = (xyz_h @ transform.T)[:, :3]
    return result


class LidarPreprocessor:
    def __init__(
        self,
        roi: Mapping[str, Sequence[float]],
        self_polygon: NDArray[np.float64] | None = None,
        zero_epsilon: float = 1e-6,
    ) -> None:
        self.roi = {
            axis: (float(bounds[0]), float(bounds[1])) for axis, bounds in roi.items()
        }
        for axis in ("x", "y", "z"):
            if axis not in self.roi or self.roi[axis][0] >= self.roi[axis][1]:
                raise ValueError(f"Invalid ROI for axis {axis}")
        self.self_polygon = None if self_polygon is None else np.asarray(self_polygon, dtype=np.float64)
        if self.self_polygon is not None and (
            self.self_polygon.ndim != 2 or self.self_polygon.shape[1] != 2
        ):
            raise ValueError("self_polygon must have shape [K,2]")
        self.zero_epsilon = float(zero_epsilon)

    @staticmethod
    def _inside_polygon(xy: NDArray[np.float64], polygon: NDArray[np.float64]) -> NDArray[np.bool_]:
        # Vectorized ray casting. Boundary points are conservatively treated as inside.
        x, y = xy[:, 0], xy[:, 1]
        inside = np.zeros(x.shape, dtype=bool)
        j = polygon.shape[0] - 1
        for i in range(polygon.shape[0]):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) & (
                x <= (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            )
            inside ^= intersects
            j = i
        return inside

    def process(
        self,
        frame: LidarFrame,
        lidar_to_base: NDArray[np.float64] | None = None,
    ) -> tuple[LidarFrame, PointCloudQuality]:
        points = np.asarray(frame.points, dtype=np.float64)
        input_points = points.shape[0]

        finite = np.all(np.isfinite(points[:, :3]), axis=1)
        invalid_points = int(np.count_nonzero(~finite))
        points = points[finite]

        nonzero = np.linalg.norm(points[:, :3], axis=1) > self.zero_epsilon
        zero_points = int(np.count_nonzero(~nonzero))
        points = points[nonzero]

        if lidar_to_base is not None:
            points = transform_points(points, np.asarray(lidar_to_base, dtype=np.float64))

        if self.self_polygon is None:
            self_mask = np.zeros(points.shape[0], dtype=bool)
        else:
            self_mask = self._inside_polygon(points[:, :2], self.self_polygon)
        self_points = int(np.count_nonzero(self_mask))
        points = points[~self_mask]

        roi_mask = np.ones(points.shape[0], dtype=bool)
        for index, axis in enumerate(("x", "y", "z")):
            lower, upper = self.roi[axis]
            roi_mask &= (points[:, index] >= lower) & (points[:, index] <= upper)
        outside = int(np.count_nonzero(~roi_mask))
        points = points[roi_mask]

        quality = PointCloudQuality(
            input_points=input_points,
            output_points=points.shape[0],
            invalid_points=invalid_points,
            zero_points=zero_points,
            self_points=self_points,
            outside_roi_points=outside,
        )
        output = LidarFrame(
            points=points,
            timestamp_s=frame.timestamp_s,
            frame_id="base_footprint" if lidar_to_base is not None else frame.frame_id,
            fields=frame.fields,
            metadata={**dict(frame.metadata), **quality.as_dict()},
        )
        output.validate()
        return output, quality

