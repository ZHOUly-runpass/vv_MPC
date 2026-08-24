from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClusterObservation:
    center_xy: NDArray[np.float64]
    length: float
    width: float
    yaw: float
    point_count: int


class EuclideanClusterer:
    """Dependency-light XY grid-connected clustering for local safety perception."""

    def __init__(self, tolerance_m: float = 0.2, minimum_points: int = 3) -> None:
        if tolerance_m <= 0 or minimum_points <= 0:
            raise ValueError("Clustering parameters must be positive")
        self.tolerance = float(tolerance_m)
        self.minimum_points = int(minimum_points)

    def cluster(self, points: NDArray[np.float64]) -> list[ClusterObservation]:
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("points must have shape [N,D], D>=2")
        if points.shape[0] == 0:
            return []
        xy = points[:, :2]
        cells = np.floor(xy / self.tolerance).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, cell in enumerate(cells):
            buckets.setdefault((int(cell[0]), int(cell[1])), []).append(index)

        unvisited = set(buckets)
        observations: list[ClusterObservation] = []
        neighbor_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        while unvisited:
            seed = unvisited.pop()
            stack = [seed]
            component = [seed]
            while stack:
                cell = stack.pop()
                for dx, dy in neighbor_offsets:
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            indices = [index for cell in component for index in buckets[cell]]
            if len(indices) < self.minimum_points:
                continue
            observations.append(self._fit(xy[np.asarray(indices, dtype=np.int64)]))
        return observations

    @staticmethod
    def _fit(xy: NDArray[np.float64]) -> ClusterObservation:
        center = np.mean(xy, axis=0)
        centered = xy - center
        if xy.shape[0] >= 2:
            covariance = centered.T @ centered / max(xy.shape[0] - 1, 1)
            values, vectors = np.linalg.eigh(covariance)
            primary = vectors[:, int(np.argmax(values))]
            yaw = float(np.arctan2(primary[1], primary[0]))
        else:
            yaw = 0.0
        c, s = np.cos(-yaw), np.sin(-yaw)
        rotated = centered @ np.array([[c, -s], [s, c]], dtype=np.float64).T
        extents = np.ptp(rotated, axis=0)
        return ClusterObservation(
            center_xy=center.astype(np.float64),
            length=float(max(extents[0], 0.05)),
            width=float(max(extents[1], 0.05)),
            yaw=yaw,
            point_count=xy.shape[0],
        )


@dataclass(frozen=True)
class OccupancyGrid2D:
    data: NDArray[np.bool_]
    resolution_m: float
    origin_xy: NDArray[np.float64]

    @classmethod
    def from_points(
        cls,
        points: NDArray[np.float64],
        resolution_m: float,
        size_x_m: float,
        size_y_m: float,
    ) -> "OccupancyGrid2D":
        width = int(np.ceil(size_x_m / resolution_m))
        height = int(np.ceil(size_y_m / resolution_m))
        origin = np.array([-size_x_m / 2.0, -size_y_m / 2.0], dtype=np.float64)
        data = np.zeros((height, width), dtype=bool)
        if points.size:
            indices = np.floor((points[:, :2] - origin) / resolution_m).astype(np.int64)
            valid = (
                (indices[:, 0] >= 0)
                & (indices[:, 0] < width)
                & (indices[:, 1] >= 0)
                & (indices[:, 1] < height)
            )
            indices = indices[valid]
            data[indices[:, 1], indices[:, 0]] = True
        return cls(data=data, resolution_m=float(resolution_m), origin_xy=origin)

    def occupied(self, xy: NDArray[np.float64]) -> NDArray[np.bool_]:
        indices = np.floor((xy - self.origin_xy) / self.resolution_m).astype(np.int64)
        height, width = self.data.shape
        result = np.ones(indices.shape[0], dtype=bool)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < width)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < height)
        )
        result[valid] = self.data[indices[valid, 1], indices[valid, 0]]
        return result

