from __future__ import annotations

import numpy as np

from ..interfaces import ExecutionCommand
from ..vehicle import VehicleLimits


class CommandAdapter:
    def __init__(self, variant: str, limits: VehicleLimits, motion_unlocked: bool) -> None:
        self.variant = variant
        self.limits = limits
        self.motion_unlocked = bool(motion_unlocked)

    @staticmethod
    def zero(timestamp_s: float, source: str = "safety_zero") -> ExecutionCommand:
        return ExecutionCommand(timestamp_s=timestamp_s, source=source)

    def sanitize(self, command: ExecutionCommand, now_s: float) -> ExecutionCommand:
        if not self.motion_unlocked or not command.is_finite():
            return self.zero(now_s, "locked_or_nonfinite")
        linear_x = float(np.clip(command.linear_x, -self.limits.reverse_velocity, self.limits.forward_velocity))
        linear_y_limit = self.limits.lateral_velocity if self.variant in {"mecanum", "omni"} else 0.0
        linear_y = float(np.clip(command.linear_y, -linear_y_limit, linear_y_limit))
        angular_z = float(np.clip(command.angular_z, -self.limits.yaw_rate, self.limits.yaw_rate))
        return ExecutionCommand(linear_x, linear_y, angular_z, now_s, f"sanitized:{command.source}")

