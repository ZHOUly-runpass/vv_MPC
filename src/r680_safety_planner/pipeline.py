from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backbones import DeterministicBevBackbone, FrozenLidarBackbone
from .config import ProjectConfig
from .dcbf import CasadiDcbfSolver, ReferenceDcbfSolver
from .execution import CommandAdapter, SafetyInputs, SafetySupervisor
from .interfaces import ExecutionCommand, LidarFrame, MpcRequest
from .lidar import LidarPreprocessor
from .perception import ConstantVelocityTracker, EuclideanClusterer, OccupancyGrid2D
from .planning import AnalyticalCandidateFilter, CandidateGenerator
from .vehicle import DifferentialModel, VehicleLimits, VehicleModel


@dataclass(frozen=True)
class PipelineCycleResult:
    command: ExecutionCommand
    motion_allowed: bool
    obstacle_count: int
    accepted_candidates: int
    selected_role: str | None
    safety_codes: tuple[str, ...]


class SafetyPlanningPipeline:
    def __init__(
        self,
        config: ProjectConfig,
        model: VehicleModel,
        ego_radius_m: float,
        backbone: FrozenLidarBackbone | None = None,
    ) -> None:
        self.config = config
        self.model = model
        roi = config.get("lidar", "preprocessing", "planning_roi_m")
        self.preprocessor = LidarPreprocessor(roi=roi)
        clustering = config.get("geometric_perception", "clustering")
        self.clusterer = EuclideanClusterer(clustering["tolerance_m"], clustering["minimum_points"])
        tracking = config.get("geometric_perception", "tracking")
        self.tracker = ConstantVelocityTracker(
            maximum_missed_frames=tracking["maximum_missed_frames"],
            process_variance=tracking["covariance_growth_per_second_m2"],
        )
        self.backbone = backbone or DeterministicBevBackbone()
        self.generator = CandidateGenerator(
            model,
            config.get("planning", "horizon_s"),
            config.get("planning", "candidate_dt_s"),
        )
        self.filter = AnalyticalCandidateFilter(model)
        dcbf = config.get("mpc", "dcbf")
        solver_config = config.get("mpc", "solver")
        use_optimizer = bool(config.motion_unlocked and config.get("mpc", "enabled_after_vehicle_validation"))
        if use_optimizer:
            if not CasadiDcbfSolver.available():
                raise RuntimeError("motion-capable MPC requested but CasADi is unavailable")
            self.solver = CasadiDcbfSolver(
                model=model,
                ego_radius_m=ego_radius_m,
                fixed_margin_m=float(dcbf["fixed_margin_m"]),
                sigma_multiplier=dcbf["sigma_multiplier"],
                continuous_alpha=dcbf["continuous_alpha"],
                maximum_slack=dcbf["maximum_slack"],
                hard_deadline_ms=solver_config["hard_deadline_ms"],
                max_iterations=solver_config["max_iterations"],
                weights=config.get("mpc", "weights"),
            )
        else:
            self.solver = ReferenceDcbfSolver(
                ego_radius_m=ego_radius_m,
                fixed_margin_m=float(dcbf.get("fixed_margin_m") or 0.1),
                sigma_multiplier=dcbf["sigma_multiplier"],
                continuous_alpha=dcbf["continuous_alpha"],
                maximum_slack=0.0,
            )
        safety = config.get("safety_supervisor")
        self.adapter = CommandAdapter(model.variant, model.limits, config.motion_unlocked)
        self.supervisor = SafetySupervisor(
            self.adapter,
            safety["planner_command_timeout_s"],
            safety["planner_heartbeat_timeout_s"],
            safety["point_cloud_timeout_s"],
            safety["odometry_timeout_s"],
            safety["imu_timeout_s"],
            safety["tf_timeout_s"],
        )

    def cycle(
        self,
        frame: LidarFrame,
        initial_state: np.ndarray,
        now_s: float,
        route_xy: np.ndarray | None = None,
        odometry_s: float | None = None,
        imu_s: float | None = None,
        tf_s: float | None = None,
    ) -> PipelineCycleResult:
        processed, _ = self.preprocessor.process(frame)
        self.backbone.infer(processed).validate()
        observations = self.clusterer.cluster(processed.points)
        self.tracker.update(observations, processed.timestamp_s)
        obstacles = self.tracker.predict(
            self.config.get("obstacle_prediction", "horizon_s"),
            self.config.get("planning", "candidate_dt_s"),
        )
        occupancy_cfg = self.config.get("geometric_perception", "occupancy")
        occupancy = OccupancyGrid2D.from_points(
            processed.points,
            occupancy_cfg["resolution_m"],
            occupancy_cfg["size_x_m"],
            occupancy_cfg["size_y_m"],
        )
        accepted = []
        route_ready = route_xy is not None and route_xy.ndim == 2 and route_xy.shape[0] >= 2
        target_speed = None if route_ready else 0.0
        for candidate in self.generator.generate(
            initial_state, route_xy=route_xy, target_speed_mps=target_speed
        ):
            check = self.filter.check(candidate, occupancy=None)  # obstacle barriers handle occupied points
            if check.accepted:
                accepted.append(candidate)

        selected = None
        solver_timed_out = False
        proposed = CommandAdapter.zero(now_s, "no_feasible_candidate")
        obstacle_emergency = False
        for candidate in accepted[: int(self.config.get("planning", "top_k_for_mpc"))]:
            request = MpcRequest(initial_state.copy(), candidate, obstacles)
            result = self.solver.solve(request)
            deadline = self.config.get("mpc", "solver", "hard_deadline_ms")
            solver_timed_out = result.solve_time_ms > deadline
            if result.h_min < 0.0:
                obstacle_emergency = True
            if result.feasible and not solver_timed_out:
                selected = candidate
                proposed = self.model.command(initial_state, result.controls[0], now_s)
                break

        decision = self.supervisor.evaluate(
            SafetyInputs(
                now_s=now_s,
                command=proposed,
                planner_heartbeat_s=now_s,
                point_cloud_s=frame.timestamp_s,
                odometry_s=float("-inf") if odometry_s is None else odometry_s,
                imu_s=float("-inf") if imu_s is None else imu_s,
                tf_s=float("-inf") if tf_s is None else tf_s,
                solver_timed_out=solver_timed_out,
                obstacle_emergency=obstacle_emergency and selected is None,
                route_ready=route_ready,
            )
        )
        return PipelineCycleResult(
            command=decision.command,
            motion_allowed=decision.allowed,
            obstacle_count=len(obstacles),
            accepted_candidates=len(accepted),
            selected_role=None if selected is None else selected.role,
            safety_codes=tuple(event.code for event in decision.events),
        )


def run_synthetic_smoke(config: ProjectConfig) -> dict[str, object]:
    limits = VehicleLimits.from_mapping(config.get("vehicle", "commissioning_limits"))
    model = DifferentialModel(limits)
    pipeline = SafetyPlanningPipeline(config, model=model, ego_radius_m=0.35)
    rng = np.random.default_rng(7)
    obstacle = np.column_stack(
        [rng.normal(3.0, 0.15, 80), rng.normal(1.0, 0.15, 80), rng.normal(0.4, 0.1, 80)]
    )
    extra = np.zeros((obstacle.shape[0], 3), dtype=np.float64)
    frame = LidarFrame(np.concatenate([obstacle, extra], axis=1), timestamp_s=10.0)
    result = pipeline.cycle(
        frame,
        np.zeros(5, dtype=np.float64),
        now_s=10.02,
        route_xy=np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64),
        odometry_s=10.02,
        imu_s=10.02,
        tf_s=10.02,
    )
    if np.any(result.command.as_array() != 0.0):
        raise RuntimeError("Synthetic smoke violated commissioning motion lock")
    return {
        "motion_allowed": result.motion_allowed,
        "command": result.command.as_array().tolist(),
        "obstacle_count": result.obstacle_count,
        "accepted_candidates": result.accepted_candidates,
        "selected_role": result.selected_role,
        "safety_codes": list(result.safety_codes),
    }
