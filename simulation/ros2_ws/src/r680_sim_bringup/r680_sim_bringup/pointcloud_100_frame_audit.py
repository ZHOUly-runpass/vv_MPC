from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


FIELDS = ("x", "y", "z", "intensity", "ring", "time")
RANGE = np.array([-54.0, -54.0, -3.0, 54.0, 54.0, 5.0], dtype=np.float64)
VOXEL = np.array([0.3, 0.3, 0.25], dtype=np.float64)


class PointCloud100FrameAudit(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_100_frame_audit")
        self.declare_parameter("topic", "/points")
        self.declare_parameter("frames", 100)
        self.declare_parameter("capture_output", "simulation_c16_100_frames.npz")
        self.declare_parameter("report_output", "simulation_c16_100_frames.json")
        self.frames = []
        self.stamps = []
        self.latencies_ms = []
        self.empty_voxel_ratios = []
        self.errors = []
        self.create_subscription(PointCloud2, self.get_parameter("topic").value, self.receive, qos_profile_sensor_data)

    def receive(self, message: PointCloud2) -> None:
        if len(self.frames) >= int(self.get_parameter("frames").value):
            return
        names = {field.name for field in message.fields}
        if not set(FIELDS).issubset(names):
            self.errors.append(f"missing fields: {sorted(set(FIELDS) - names)}")
            return
        raw = point_cloud2.read_points(message, field_names=FIELDS, skip_nans=False)
        values = np.column_stack([raw[name] for name in FIELDS]).astype(np.float32, copy=False)
        if values.size == 0 or not np.all(np.isfinite(values)):
            self.errors.append("empty or non-finite frame")
            return
        if np.min(values[:, 4]) < 0 or np.max(values[:, 4]) > 15:
            self.errors.append("ring outside [0,15]")
            return
        if np.min(values[:, 5]) < -1e-6 or np.max(values[:, 5]) > 0.100001:
            self.errors.append("relative time outside one scan period")
            return
        stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        inside = np.all((values[:, :3] >= RANGE[:3]) & (values[:, :3] < RANGE[3:]), axis=1)
        indices = np.floor((values[inside, :3] - RANGE[:3]) / VOXEL).astype(np.int64)
        occupied = len(np.unique(indices, axis=0)) if len(indices) else 0
        total = int(np.prod(np.ceil((RANGE[3:] - RANGE[:3]) / VOXEL)))
        self.frames.append(values)
        self.stamps.append(stamp)
        self.latencies_ms.append(max(0.0, (now - stamp) * 1000.0))
        self.empty_voxel_ratios.append(1.0 - occupied / total)

    def done(self) -> bool:
        return len(self.frames) >= int(self.get_parameter("frames").value)

    def write(self) -> bool:
        offsets = np.concatenate([[0], np.cumsum([len(frame) for frame in self.frames])]).astype(np.int64)
        points = np.concatenate(self.frames, axis=0) if self.frames else np.empty((0, 6), dtype=np.float32)
        stamps = np.asarray(self.stamps, dtype=np.float64)
        monotonic = bool(len(stamps) > 1 and np.all(np.diff(stamps) > 0.0))
        frequency = float(1.0 / np.median(np.diff(stamps))) if monotonic else 0.0
        report = {
            "status": "passed" if self.done() and monotonic and not self.errors else "failed",
            "frames": len(self.frames), "fields": list(FIELDS), "frame_id": "base_footprint",
            "timestamp_monotonic": monotonic, "frequency_hz": frequency,
            "points_total": int(len(points)), "points_per_frame_min": min(map(len, self.frames), default=0),
            "points_per_frame_max": max(map(len, self.frames), default=0),
            "finite": bool(np.all(np.isfinite(points))),
            "empty_voxel_ratio_mean": float(np.mean(self.empty_voxel_ratios)) if self.empty_voxel_ratios else None,
            "transport_latency_ms_median": float(np.median(self.latencies_ms)) if self.latencies_ms else None,
            "errors": self.errors,
        }
        np.savez_compressed(self.get_parameter("capture_output").value, points=points, offsets=offsets, stamps=stamps, fields=np.asarray(FIELDS))
        with open(self.get_parameter("report_output").value, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        return report["status"] == "passed"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointCloud100FrameAudit()
    started = time.monotonic()
    try:
        while rclpy.ok() and not node.done() and time.monotonic() - started < 30.0:
            rclpy.spin_once(node, timeout_sec=0.2)
        success = node.write()
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    raise SystemExit(0 if success else 1)
