from __future__ import annotations

import json
import math

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from .scenario import load_scenario


class ScenarioDifficultyController(Node):
    def __init__(self) -> None:
        super().__init__("scenario_difficulty_controller")
        for name, default in (("scenario_file", ""), ("scenario", "empty"), ("difficulty", "nominal")):
            self.declare_parameter(name, default)
        _, scenario = load_scenario(self.get_parameter("scenario_file").value,
                                    self.get_parameter("scenario").value,
                                    self.get_parameter("difficulty").value)
        self.difficulty = self.get_parameter("difficulty").value
        self.targets = dict(scenario.get("model_poses", {}))
        for index, name in enumerate(scenario.get("inactive_obstacles", [])):
            self.targets[name] = [1000.0+index, 1000.0, -50.0, 0.0]
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(String, "/simulation/difficulty_status", qos)
        self.client = self.create_client(SetEntityState, "/set_entity_state")
        self.pending = {}; self.completed_names = set(); self.failed = {}; self.finished = not self.targets
        self.create_timer(0.1, self.tick)

    def tick(self) -> None:
        if self.finished:
            self.publish_status(); return
        if not self.client.service_is_ready(): return
        for name, pose in self.targets.items():
            if name in self.pending or name in self.completed_names or name in self.failed: continue
            state = EntityState(name=name, reference_frame="world")
            state.pose.position.x, state.pose.position.y, state.pose.position.z = map(float, pose[:3])
            angle = float(pose[3]); state.pose.orientation.z = math.sin(angle/2.0); state.pose.orientation.w = math.cos(angle/2.0)
            request = SetEntityState.Request(); request.state = state
            future = self.client.call_async(request); self.pending[name] = future
            future.add_done_callback(lambda done, model=name: self.complete(model, done))

    def complete(self, name, future) -> None:
        self.pending.pop(name, None)
        try:
            response = future.result()
            if response.success: self.completed_names.add(name)
            else: self.failed[name] = response.status_message
        except Exception as error: self.failed[name] = f"{type(error).__name__}:{error}"
        self.finished = len(self.completed_names)+len(self.failed) == len(self.targets)
        self.publish_status()

    def publish_status(self) -> None:
        payload = {"schema_version": "1.0", "difficulty": self.difficulty, "ready": self.finished and not self.failed,
                   "configured_models": sorted(self.completed_names), "failed_models": self.failed,
                   "target_count": len(self.targets)}
        self.publisher.publish(String(data=json.dumps(payload, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args); node = ScenarioDifficultyController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
