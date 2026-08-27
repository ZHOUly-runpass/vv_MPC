from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


ROLES = ("nominal_route_following", "left_or_counterclockwise_bias",
         "right_or_clockwise_bias", "reduced_speed", "controlled_stop",
         "model_specific_escape_1", "model_specific_escape_2")


def yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PlanningTracePublisher(Node):
    """Publish auditable planning inputs during every baseline collection run."""

    def __init__(self) -> None:
        super().__init__("planning_trace_publisher")
        self.declare_parameter("baseline", "uncontrolled")
        self.declare_parameter("horizon_s", 2.0)
        self.declare_parameter("dt_s", 0.1)
        self.odom = None; self.route = None; self.obstacles = []; self.controller_status = {}
        self.trace_publishers = {name: self.create_publisher(String, f"/planning/{name}", 10) for name in
                                 ("candidates", "obstacle_predictions", "mpc_request", "mpc_result")}
        self.create_subscription(Odometry, "/odom", lambda msg: setattr(self, "odom", msg), 20)
        route_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "/simulation/reference_route", lambda msg: setattr(self, "route", msg), route_qos)
        self.create_subscription(String, "/simulation/ground_truth_obstacles", self.on_obstacles, 10)
        self.create_subscription(String, "/simulation/controller_status", self.on_controller, 10)
        self.create_timer(0.2, self.tick)

    def on_obstacles(self, message) -> None:
        try: self.obstacles = json.loads(message.data).get("obstacles", [])
        except (ValueError, TypeError): self.obstacles = []

    def on_controller(self, message) -> None:
        try: self.controller_status = json.loads(message.data)
        except (ValueError, TypeError): self.controller_status = {}

    def publish(self, name: str, payload: dict) -> None:
        payload.update(schema_version="1.0", stamp_s=self.get_clock().now().nanoseconds * 1e-9,
                       baseline=self.get_parameter("baseline").value)
        self.trace_publishers[name].publish(String(data=json.dumps(payload, separators=(",", ":"), sort_keys=True)))

    def tick(self) -> None:
        if self.odom is None or self.route is None or not self.route.poses: return
        pose = self.odom.pose.pose; twist = self.odom.twist.twist
        initial = np.array([pose.position.x, pose.position.y, yaw(pose.orientation),
                            twist.linear.x, twist.angular.z], dtype=float)
        goal = self.route.poses[-1].pose.position
        desired_yaw = math.atan2(goal.y - initial[1], goal.x - initial[0])
        heading_error = math.atan2(math.sin(desired_yaw-initial[2]), math.cos(desired_yaw-initial[2]))
        dt = float(self.get_parameter("dt_s").value)
        intervals = int(round(float(self.get_parameter("horizon_s").value) / dt))
        timestamps = np.arange(intervals + 1, dtype=float) * dt
        candidates = []
        for role in ROLES:
            controls = np.zeros((intervals, 2), dtype=float)
            target = 0.25 if role != "reduced_speed" else 0.12
            if role == "controlled_stop": target = 0.0
            controls[:, 0] = np.clip((target - initial[3]) / max(intervals*dt, dt), -0.8, 0.5)
            controls[:, 1] = np.clip((heading_error / max(intervals*dt, dt) - initial[4]) / max(intervals*dt, dt), -1.0, 1.0)
            if role in {"left_or_counterclockwise_bias", "model_specific_escape_1"}: controls[:, 1] = 0.5
            if role in {"right_or_clockwise_bias", "model_specific_escape_2"}: controls[:, 1] = -0.5
            if role.startswith("model_specific_escape"): controls[:, 0] = -0.4
            states = np.zeros((intervals + 1, 5), dtype=float); states[0] = initial
            for index in range(intervals):
                x, y, angle, velocity, rate = states[index]
                states[index+1] = [x+dt*velocity*math.cos(angle), y+dt*velocity*math.sin(angle),
                                   angle+dt*rate, np.clip(velocity+dt*controls[index, 0], 0.0, 0.5),
                                   np.clip(rate+dt*controls[index, 1], -0.8, 0.8)]
            candidates.append({"role": role, "states": states.tolist(), "controls": controls.tolist()})
        predictions = []
        for obstacle in self.obstacles:
            px, py = obstacle["position"][:2]; vx, vy = obstacle["velocity"][:2]
            states = [[px+vx*t, py+vy*t, 0.0, vx, vy, 1.0] for t in timestamps]
            predictions.append({"name": obstacle["name"], "states": states,
                                "lengths": [2.0*float(obstacle["radius_m"])]*len(timestamps),
                                "widths": [2.0*float(obstacle["radius_m"])]*len(timestamps),
                                "covariance": [(np.eye(2)*(0.01+0.05*t)).tolist() for t in timestamps],
                                "valid_mask": [True]*len(timestamps)})
        self.publish("candidates", {"timestamps_s": timestamps.tolist(), "items": candidates})
        self.publish("obstacle_predictions", {"timestamps_s": timestamps.tolist(), "items": predictions})
        self.publish("mpc_request", {"initial_state": initial.tolist(), "candidate_count": len(candidates),
                                     "obstacle_count": len(predictions), "online_solver_requested": False})
        h_min = self.controller_status.get("h_min")
        self.publish("mpc_result", {"status": "offline_teacher_pending", "solver_kind": "none",
                                    "feasible": None, "h_min": h_min, "slack_max": None,
                                    "solve_time_ms": None, "selected_index": None})


def main(args=None) -> None:
    rclpy.init(args=args); node = PlanningTracePublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
