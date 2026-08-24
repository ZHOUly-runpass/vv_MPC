from __future__ import annotations

from abc import ABC, abstractmethod

from ..interfaces import FrozenSceneFeatures, LidarFrame


class FrozenLidarBackbone(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @abstractmethod
    def infer(self, frame: LidarFrame) -> FrozenSceneFeatures:
        ...

    @abstractmethod
    def healthcheck(self) -> dict[str, object]:
        ...

