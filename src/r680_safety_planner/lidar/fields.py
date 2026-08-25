from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..interfaces import LidarFrame


@dataclass(frozen=True)
class UniLionPointFieldContract:
    """Map a named PointCloud2-style frame to UniLION's five input features.

    The official nuScenes configuration loads five values in this order:
    ``x, y, z, intensity, ring``. Relative point time is useful for deskewing,
    but it is not passed to the pinned UniLION voxel encoder.
    """

    model_fields: tuple[str, ...] = ("x", "y", "z", "intensity", "ring")
    expected_ring_count: int = 16
    ring_integer_tolerance: float = 1e-3

    def adapt(self, frame: LidarFrame) -> NDArray[np.float32]:
        frame.validate()
        if len(frame.fields) != frame.points.shape[1]:
            raise ValueError("point field count does not match point tensor width")
        if len(set(frame.fields)) != len(frame.fields):
            raise ValueError("point field names must be unique")

        missing = tuple(name for name in self.model_fields if name not in frame.fields)
        if missing:
            raise ValueError(f"missing UniLION point fields: {missing}")
        indices = [frame.fields.index(name) for name in self.model_fields]
        adapted = np.ascontiguousarray(frame.points[:, indices], dtype=np.float32)
        if adapted.shape[0] == 0:
            raise ValueError("UniLION input point cloud is empty")
        if not np.all(np.isfinite(adapted)):
            raise ValueError("adapted UniLION points contain NaN or Inf")

        rings = adapted[:, self.model_fields.index("ring")]
        if np.any(np.abs(rings - np.rint(rings)) > self.ring_integer_tolerance):
            raise ValueError("ring values must be integer channel indices")
        if np.any((rings < 0) | (rings >= self.expected_ring_count)):
            raise ValueError(
                f"ring values must be in [0,{self.expected_ring_count - 1}] for C16"
            )
        return adapted

    def describe(self) -> dict[str, object]:
        return {
            "model_fields": list(self.model_fields),
            "ignored_but_supported_fields": ["time"],
            "expected_ring_count": self.expected_ring_count,
            "output_dtype": "float32",
            "pillar_feature_dimension": 11,
            "pillar_feature_explanation": "5 raw + 3 cluster offsets + 3 voxel-center offsets",
        }
