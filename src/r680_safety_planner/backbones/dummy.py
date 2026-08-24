from __future__ import annotations

import numpy as np

from ..interfaces import FrozenSceneFeatures, LidarFrame
from .protocol import FrozenLidarBackbone


class DeterministicBevBackbone(FrozenLidarBackbone):
    """Deterministic occupancy/density feature backend for integration tests."""

    def __init__(self, x_range=(-10.0, 30.0), y_range=(-20.0, 20.0), resolution=0.5) -> None:
        self.x_range = tuple(map(float, x_range))
        self.y_range = tuple(map(float, y_range))
        self.resolution = float(resolution)
        self.height = int(np.ceil((self.y_range[1] - self.y_range[0]) / resolution))
        self.width = int(np.ceil((self.x_range[1] - self.x_range[0]) / resolution))

    @property
    def backend_name(self) -> str:
        return "deterministic_bev_test_backend"

    def infer(self, frame: LidarFrame) -> FrozenSceneFeatures:
        frame.validate()
        points = frame.points
        x_index = np.floor((points[:, 0] - self.x_range[0]) / self.resolution).astype(int)
        y_index = np.floor((points[:, 1] - self.y_range[0]) / self.resolution).astype(int)
        valid = (
            (x_index >= 0) & (x_index < self.width) & (y_index >= 0) & (y_index < self.height)
        )
        x_index, y_index = x_index[valid], y_index[valid]
        density = np.zeros((self.height, self.width), dtype=np.float32)
        height_max = np.full((self.height, self.width), -np.inf, dtype=np.float32)
        np.add.at(density, (y_index, x_index), 1.0)
        if x_index.size:
            np.maximum.at(height_max, (y_index, x_index), points[valid, 2].astype(np.float32))
        height_max[~np.isfinite(height_max)] = 0.0
        density = np.log1p(density) / np.log(64.0)
        occupied = (density > 0.0).astype(np.float32)
        bev = np.stack([occupied, density, height_max], axis=0)[None]
        features = FrozenSceneFeatures(
            bev_feature=bev,
            source_backend=self.backend_name,
            timestamp_s=frame.timestamp_s,
            quality={key: float(value) for key, value in frame.metadata.items() if isinstance(value, (int, float))},
        )
        features.validate()
        return features

    def healthcheck(self) -> dict[str, object]:
        return {"healthy": True, "backend": self.backend_name, "shape": [1, 3, self.height, self.width]}

