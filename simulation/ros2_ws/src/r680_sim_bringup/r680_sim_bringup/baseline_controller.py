from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.msg import Costmap
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def route_array(message, count: int = 32) -> np.ndarray:
    indices = np.rint(np.linspace(0, len(message.poses) - 1, count)).astype(int)
    return np.asarray([[message.poses[index].pose.position.x, message.poses[index].pose.position.y,
                        yaw_from_quaternion(message.poses[index].pose.orientation), 0.25]
                       for index in indices], dtype=np.float32)


def costmap_array(message: Costmap) -> np.ndarray:
    grid = np.asarray(message.data, dtype=np.uint8).reshape(message.metadata.size_y, message.metadata.size_x)
    unknown = (grid == 255).astype(np.float32)
    occupied = np.minimum(grid, 254).astype(np.float32) / 254.0
    source = np.stack([occupied, np.zeros_like(occupied), unknown])
    if source.shape[1:] != (60, 60):
        rows = np.rint(np.linspace(0, source.shape[1] - 1, 60)).astype(int)
        columns = np.rint(np.linspace(0, source.shape[2] - 1, 60)).astype(int)
        source = source[:, rows][:, :, columns]
    return source.astype(np.float32)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class BaselineController(Node):
    """Vanilla D-CBF baseline and fail-closed Proposed learned-controller bridge."""

    def __init__(self) -> None:
        super().__init__("r680_baseline_controller")
        self.declare_parameter("baseline", "vanilla_dcbf")
        self.declare_parameter("robot_radius_m", 0.31)
        self.declare_parameter("runtime_dir", "")
        self.declare_parameter("feature_timeout_s", 0.30)
        self.declare_parameter("inference_timeout_s", 0.12)
        self.baseline = str(self.get_parameter("baseline").value).lower()
        if self.baseline not in {"vanilla_dcbf", "proposed"}:
            raise ValueError("baseline must be vanilla_dcbf or proposed")
        self.runtime_dir = Path(str(self.get_parameter("runtime_dir").value))
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = self.runtime_dir / "latest_request.npz"
        self.result_path = self.runtime_dir / "latest_result.json"
        self.worker_status_path = self.runtime_dir / "worker_status.json"
        self.path = None; self.odom = None; self.costmap = None; self.obstacles = []
        self.feature_summary = {}; self.feature_healthy = False
        self.odom_wall_s = 0.0; self.feature_wall_s = 0.0; self.result_wall_s = 0.0
        self.pending_sequence = None; self.pending_started_s = 0.0
        self.sequence = int(time.time() * 1_000_000); self.last_result = None
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status = self.create_publisher(String, "/simulation/controller_status", 10)
        self.create_subscription(NavPath, "/plan", self.on_path, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Costmap, "/local_costmap/costmap_raw", self.on_costmap, 10)
        self.create_subscription(String, "/simulation/ground_truth_obstacles", self.on_obstacles, 10)
        self.create_subscription(String, "/unilion/health", self.on_feature_health, 10)
        self.create_subscription(String, "/unilion/frozen_scene_features", self.on_feature_summary, 10)
        self.create_timer(0.05, self.tick)

    def on_path(self, message): self.path = message
    def on_odom(self, message): self.odom = message; self.odom_wall_s = time.monotonic()
    def on_costmap(self, message): self.costmap = costmap_array(message)
    def on_obstacles(self, message):
        try: self.obstacles = json.loads(message.data).get("obstacles", [])
        except (ValueError, TypeError): self.obstacles = []
    def on_feature_health(self, message):
        try: self.feature_healthy = bool(json.loads(message.data).get("healthy", False))
        except (ValueError, TypeError): self.feature_healthy = False
    def on_feature_summary(self, message):
        try:
            payload = json.loads(message.data); cache = Path(str(payload["cache_path"]))
            if not cache.is_file() or digest(cache) != payload["cache_sha256"]: raise ValueError("invalid feature cache")
            self.feature_summary = payload; self.feature_wall_s = time.monotonic()
        except (ValueError, TypeError, KeyError, OSError):
            self.feature_summary = {}; self.feature_wall_s = 0.0

    def worker_state(self) -> dict:
        try: return json.loads(self.worker_status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError): return {"ready": False, "error": "worker_status_unavailable"}

    def submit_request(self, now_wall: float) -> None:
        if not (self.path and self.path.poses and self.odom and self.costmap is not None and self.feature_summary): return
        pose = self.odom.pose.pose; twist = self.odom.twist.twist
        ego = np.asarray([pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation),
                          twist.linear.x, twist.angular.z], dtype=np.float32)
        self.sequence += 1
        metadata = {"sequence": self.sequence, "stamp_s": self.get_clock().now().nanoseconds * 1e-9,
                    "feature_cache_path": self.feature_summary["cache_path"],
                    "feature_cache_sha256": self.feature_summary["cache_sha256"],
                    "feature_age_s": max(0.0, now_wall - self.feature_wall_s), "obstacles": self.obstacles}
        temporary = self.runtime_dir / "latest_request.tmp.npz"
        np.savez(temporary, metadata_json=np.asarray(json.dumps(metadata)), route=route_array(self.path), ego=ego, costmap=self.costmap)
        os.replace(temporary, self.request_path)
        self.pending_sequence = self.sequence; self.pending_started_s = now_wall

    def poll_result(self, now_wall: float) -> None:
        if self.pending_sequence is None or not self.result_path.is_file(): return
        try: payload = json.loads(self.result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError): return
        if int(payload.get("sequence", -1)) != self.pending_sequence: return
        self.pending_sequence = None; self.last_result = payload; self.result_wall_s = now_wall

    def deterministic_barrier(self, command: Twist, x: float, y: float) -> tuple[Twist, str, float]:
        h_min = float("inf")
        for obstacle in self.obstacles:
            if not obstacle.get("collision_check", True): continue
            ox, oy = obstacle["position"][:2]; vx, vy = obstacle["velocity"][:2]
            clearance = math.hypot(x - (ox + vx * 0.8), y - (oy + vy * 0.8))
            clearance -= float(self.get_parameter("robot_radius_m").value) + float(obstacle["radius_m"]) + 0.18
            h_min = min(h_min, clearance)
        reason = "learned_dcbf_mpc"
        if h_min < 0.0: command = Twist(); reason = "supervisor_obstacle_stop"
        elif h_min < 0.8: command.linear.x *= max(0.1, h_min / 0.8); reason = "supervisor_obstacle_slowdown"
        return command, reason, h_min

    def tick(self) -> None:
        now_wall = time.monotonic(); command = Twist(); reason = "waiting_for_route_or_odom"; h_min = None
        worker = self.worker_state() if self.baseline == "proposed" else {"ready": False}
        checkpoint_active = False
        if self.baseline == "vanilla_dcbf":
            if self.path and self.path.poses and self.odom:
                pose = self.odom.pose.pose; x, y = pose.position.x, pose.position.y
                yaw = yaw_from_quaternion(pose.orientation); goal = self.path.poses[-1].pose.position
                distance = math.hypot(goal.x - x, goal.y - y); desired = math.atan2(goal.y - y, goal.x - x)
                error = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
                command.linear.x = min(0.5, 0.35 * distance) * max(0.0, math.cos(error))
                command.angular.z = max(-0.8, min(0.8, 1.5 * error)); reason = "nominal"
                command, reason, h_min = self.deterministic_barrier(command, x, y)
                if distance < 0.2: command = Twist(); reason = "goal_reached"
        else:
            self.poll_result(now_wall)
            feature_timeout = float(self.get_parameter("feature_timeout_s").value)
            inference_timeout = float(self.get_parameter("inference_timeout_s").value)
            if not worker.get("ready", False): reason = "checkpoint_validation_failed"
            elif not self.feature_healthy or now_wall - self.feature_wall_s > feature_timeout: reason = "feature_timeout_stop"
            elif now_wall - self.odom_wall_s > feature_timeout: reason = "odometry_timeout_stop"
            elif self.pending_sequence is not None and now_wall - self.pending_started_s > inference_timeout:
                reason = "inference_timeout_stop"; self.pending_sequence = None; self.last_result = None
            elif self.last_result and self.last_result.get("ok") and now_wall - self.result_wall_s <= feature_timeout:
                values = np.asarray(self.last_result.get("command", []), dtype=float)
                if values.shape == (3,) and np.all(np.isfinite(values)):
                    command.linear.x, command.linear.y, command.angular.z = values.tolist(); pose = self.odom.pose.pose
                    command, reason, h_min = self.deterministic_barrier(command, pose.position.x, pose.position.y)
                    checkpoint_active = True
                else: reason = "nonfinite_inference_stop"
            elif self.last_result and not self.last_result.get("ok", False): reason = "inference_failure_stop"
            else: reason = "inference_pending_stop"
            if self.pending_sequence is None and worker.get("ready", False) and self.feature_healthy: self.submit_request(now_wall)
        self.publisher.publish(command)
        payload = {"baseline": self.baseline, "reason": reason,
                   "h_min": None if h_min is None or math.isinf(h_min) else h_min,
                   "learned_checkpoint_active": checkpoint_active, "checkpoint_validation": worker,
                   "selected_index": None if not self.last_result else self.last_result.get("selected_index"),
                   "inference_ms": None if not self.last_result else self.last_result.get("inference_ms"),
                   "mpc_solve_time_ms": None if not self.last_result else self.last_result.get("mpc_solve_time_ms")}
        self.status.publish(String(data=json.dumps(payload, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args); node = BaselineController()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally:
        if rclpy.ok(): node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
