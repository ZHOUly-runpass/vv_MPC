from __future__ import annotations

from time import perf_counter

import numpy as np

from ..interfaces import MpcRequest, MpcResult
from ..vehicle.models import VehicleModel
from .barrier import barrier_series, dcbf_residual


class ReferenceDcbfSolver:
    """Deterministic validator used before the CasADi optimizer is available.

    It does not claim to optimize a trajectory. It validates the provided
    dynamically rolled-out candidate against the same barrier and D-CBF
    diagnostics expected by the future optimizer.
    """

    def __init__(
        self,
        ego_radius_m: float,
        fixed_margin_m: float = 0.1,
        sigma_multiplier: float = 3.0,
        continuous_alpha: float = 1.0,
        latency_s: float = 0.0,
        maximum_slack: float = 0.0,
    ) -> None:
        if ego_radius_m <= 0:
            raise ValueError("ego_radius_m must be positive")
        self.ego_radius_m = float(ego_radius_m)
        self.fixed_margin_m = float(fixed_margin_m)
        self.sigma_multiplier = float(sigma_multiplier)
        self.continuous_alpha = float(continuous_alpha)
        self.latency_s = float(latency_s)
        self.maximum_slack = float(maximum_slack)

    def solve(self, request: MpcRequest) -> MpcResult:
        started = perf_counter()
        request.validate()
        reference = request.reference
        dt_s = float(np.median(np.diff(reference.timestamps_s)))
        all_h: list[np.ndarray] = []
        all_residuals: list[np.ndarray] = []
        for obstacle in request.obstacles:
            h_values = barrier_series(
                reference.states,
                obstacle,
                ego_radius_m=self.ego_radius_m,
                fixed_margin_m=self.fixed_margin_m,
                sigma_multiplier=self.sigma_multiplier,
                latency_s=self.latency_s,
            )
            all_h.append(h_values)
            all_residuals.append(dcbf_residual(h_values, self.continuous_alpha, dt_s))
        if all_h:
            h_min = float(np.min(np.concatenate(all_h)))
            residual_min = float(np.min(np.concatenate(all_residuals)))
        else:
            h_min = float("inf")
            residual_min = float("inf")
        required_slack = max(0.0, -residual_min) if np.isfinite(residual_min) else 0.0
        feasible = bool(h_min >= 0.0 and required_slack <= self.maximum_slack)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return MpcResult(
            states=reference.states.copy(),
            controls=reference.controls.copy(),
            feasible=feasible,
            h_min=h_min,
            slack_max=required_slack,
            solve_time_ms=elapsed_ms,
            status="reference_feasible" if feasible else "reference_infeasible",
        )


class CasadiDcbfSolver:
    """Multiple-shooting MPC with soft discrete D-CBF constraints.

    CasADi is imported lazily so perception-only commissioning remains usable
    on computers without ROS or a nonlinear solver installation.
    """

    def __init__(
        self,
        model: VehicleModel,
        ego_radius_m: float,
        fixed_margin_m: float = 0.1,
        sigma_multiplier: float = 3.0,
        continuous_alpha: float = 1.0,
        maximum_slack: float = 1.0,
        hard_deadline_ms: float = 80.0,
        max_iterations: int = 100,
        weights: dict[str, float] | None = None,
    ) -> None:
        if ego_radius_m <= 0.0 or maximum_slack < 0.0:
            raise ValueError("invalid MPC safety dimensions")
        self.model = model
        self.ego_radius_m = float(ego_radius_m)
        self.fixed_margin_m = float(fixed_margin_m)
        self.sigma_multiplier = float(sigma_multiplier)
        self.continuous_alpha = float(continuous_alpha)
        self.maximum_slack = float(maximum_slack)
        self.hard_deadline_ms = float(hard_deadline_ms)
        self.max_iterations = int(max_iterations)
        self.weights = {
            "position": 10.0, "yaw": 2.0, "velocity": 2.0,
            "control": 0.2, "smoothness": 2.0, "obstacle_slack": 10000.0,
            **(weights or {}),
        }
        self._problem_cache: dict[tuple[int, int, float], dict[str, object]] = {}

    @staticmethod
    def available() -> bool:
        try:
            import casadi  # noqa: F401
        except ImportError:
            return False
        return True

    def _dynamics(self, ca, state, control, dt_s):
        variant = self.model.variant
        x, y, yaw = state[0], state[1], state[2]
        if variant in {"differential", "skid_steer"}:
            velocity, yaw_rate = state[3], state[4]
            yaw_gain = float(getattr(self.model, "slip_yaw_gain", 1.0))
            return ca.vertcat(x + dt_s * velocity * ca.cos(yaw),
                              y + dt_s * velocity * ca.sin(yaw),
                              yaw + dt_s * yaw_rate * yaw_gain,
                              velocity + dt_s * control[0],
                              yaw_rate + dt_s * control[1])
        if variant in {"mecanum", "omni"}:
            vx, vy, yaw_rate = state[3], state[4], state[5]
            return ca.vertcat(x + dt_s * (ca.cos(yaw) * vx - ca.sin(yaw) * vy),
                              y + dt_s * (ca.sin(yaw) * vx + ca.cos(yaw) * vy),
                              yaw + dt_s * yaw_rate,
                              vx + dt_s * control[0], vy + dt_s * control[1],
                              yaw_rate + dt_s * control[2])
        if variant == "ackermann":
            velocity = state[3]
            return ca.vertcat(x + dt_s * velocity * ca.cos(yaw),
                              y + dt_s * velocity * ca.sin(yaw),
                              yaw + dt_s * velocity / self.model.wheelbase_m * ca.tan(control[1]),
                              velocity + dt_s * control[0])
        raise ValueError(f"unsupported MPC vehicle model: {variant}")

    def _apply_bounds(self, opti, state, control) -> None:
        limits = self.model.limits
        opti.subject_to(opti.bounded(-limits.reverse_velocity, state[3, :], limits.forward_velocity))
        opti.subject_to(opti.bounded(-limits.braking_deceleration, control[0, :], limits.acceleration))
        if self.model.variant in {"differential", "skid_steer"}:
            opti.subject_to(opti.bounded(-limits.yaw_rate, state[4, :], limits.yaw_rate))
            opti.subject_to(opti.bounded(-limits.yaw_acceleration, control[1, :], limits.yaw_acceleration))
        elif self.model.variant in {"mecanum", "omni"}:
            opti.subject_to(opti.bounded(-limits.lateral_velocity, state[4, :], limits.lateral_velocity))
            opti.subject_to(opti.bounded(-limits.yaw_rate, state[5, :], limits.yaw_rate))
            opti.subject_to(opti.bounded(-limits.lateral_acceleration, control[1, :], limits.lateral_acceleration))
            opti.subject_to(opti.bounded(-limits.yaw_acceleration, control[2, :], limits.yaw_acceleration))
        elif self.model.variant == "ackermann":
            opti.subject_to(opti.bounded(-self.model.limits.steering_angle, control[1, :], self.model.limits.steering_angle))

    def select_reachable_obstacles(self, request: MpcRequest) -> tuple:
        """Conservatively remove obstacles outside the horizon reachable disk.

        The disk assumes the vehicle can instantly move at its maximum speed in
        any direction, so it over-approximates every supported vehicle model.
        Obstacles with invalid masks remain active (fail-closed).
        """
        timestamps = request.reference.timestamps_s - request.reference.timestamps_s[0]
        limits = self.model.limits
        longitudinal = max(limits.forward_velocity, limits.reverse_velocity)
        speed_bound = (float(np.hypot(longitudinal, limits.lateral_velocity))
                       if self.model.variant in {"mecanum", "omni"} else longitudinal)
        reachable_radius = speed_bound * timestamps
        initial_xy = request.initial_state[:2]
        active = []
        for obstacle in request.obstacles:
            if not np.all(obstacle.valid_mask):
                active.append(obstacle)
                continue
            covariance_radius = self.sigma_multiplier * np.sqrt(np.maximum(
                0.0, np.linalg.eigvalsh(obstacle.covariance)[:, -1]))
            obstacle_radius = 0.5 * np.hypot(obstacle.lengths, obstacle.widths)
            safe_radius = self.ego_radius_m + obstacle_radius + self.fixed_margin_m + covariance_radius
            center_distance = np.linalg.norm(obstacle.states[:, :2] - initial_xy, axis=1)
            if np.any(center_distance <= reachable_radius + safe_radius):
                active.append(obstacle)
        return tuple(active)

    def _build_problem(self, ca, intervals: int, obstacle_count: int, dt_s: float) -> dict[str, object]:
        opti = ca.Opti()
        states = opti.variable(self.model.state_size, intervals + 1)
        controls = opti.variable(self.model.control_size, intervals)
        initial = opti.parameter(self.model.state_size)
        reference_states = opti.parameter(self.model.state_size, intervals + 1)
        reference_controls = opti.parameter(self.model.control_size, intervals)
        slack = opti.variable(obstacle_count, intervals) if obstacle_count else None
        obstacle_x = opti.parameter(obstacle_count, intervals + 1) if obstacle_count else None
        obstacle_y = opti.parameter(obstacle_count, intervals + 1) if obstacle_count else None
        safe_radius = opti.parameter(obstacle_count, intervals + 1) if obstacle_count else None
        opti.subject_to(states[:, 0] == initial)
        for index in range(intervals):
            opti.subject_to(states[:, index + 1] == self._dynamics(
                ca, states[:, index], controls[:, index], dt_s))
        self._apply_bounds(opti, states, controls)

        objective = self.weights["position"] * ca.sumsqr(states[0:2, :] - reference_states[0:2, :])
        objective += self.weights["yaw"] * ca.sumsqr(states[2, :] - reference_states[2, :])
        objective += self.weights["velocity"] * ca.sumsqr(states[3:, :] - reference_states[3:, :])
        objective += self.weights["control"] * ca.sumsqr(controls - reference_controls)
        if intervals > 1:
            objective += self.weights["smoothness"] * ca.sumsqr(controls[:, 1:] - controls[:, :-1])
        if slack is not None:
            opti.subject_to(opti.bounded(0.0, slack, self.maximum_slack))
            objective += self.weights["obstacle_slack"] * ca.sumsqr(slack)
            gamma = 1.0 - np.exp(-self.continuous_alpha * dt_s)
            for obstacle_index in range(obstacle_count):
                h_values = []
                for index in range(intervals + 1):
                    dx = states[0, index] - obstacle_x[obstacle_index, index]
                    dy = states[1, index] - obstacle_y[obstacle_index, index]
                    h_values.append(dx * dx + dy * dy - safe_radius[obstacle_index, index] ** 2)
                for index in range(intervals):
                    opti.subject_to(h_values[index] + slack[obstacle_index, index] >= 0.0)
                    opti.subject_to(h_values[index + 1] - (1.0 - gamma) * h_values[index]
                                    + slack[obstacle_index, index] >= 0.0)
        opti.minimize(objective)
        opti.solver("ipopt", {"expand": True, "print_time": False},
                    {"max_iter": self.max_iterations, "print_level": 0, "sb": "yes"})
        return {
            "opti": opti, "states": states, "controls": controls, "slack": slack,
            "initial": initial, "reference_states": reference_states,
            "reference_controls": reference_controls, "obstacle_x": obstacle_x,
            "obstacle_y": obstacle_y, "safe_radius": safe_radius,
            "last_states": None, "last_controls": None, "last_slack": None,
        }

    def solve(self, request: MpcRequest) -> MpcResult:
        started = perf_counter()
        request.validate()
        if not self.available():
            raise RuntimeError("CasADi is not installed; install project extra 'solver'")
        import casadi as ca

        reference = request.reference
        intervals = reference.controls.shape[0]
        dt_s = float(np.median(np.diff(reference.timestamps_s)))
        active_obstacles = self.select_reachable_obstacles(request)
        obstacle_count = len(active_obstacles)
        cache_key = (intervals, obstacle_count, round(dt_s, 9))
        problem = self._problem_cache.get(cache_key)
        if problem is None:
            problem = self._build_problem(ca, intervals, obstacle_count, dt_s)
            self._problem_cache[cache_key] = problem
        opti, states, controls, slack = (problem["opti"], problem["states"],
                                          problem["controls"], problem["slack"])
        opti.set_value(problem["initial"], request.initial_state)
        opti.set_value(problem["reference_states"], reference.states.T)
        opti.set_value(problem["reference_controls"], reference.controls.T)
        if obstacle_count:
            centers = np.stack([obstacle.states[:, :2] for obstacle in active_obstacles])
            radii = []
            for obstacle in active_obstacles:
                covariance_radius = self.sigma_multiplier * np.sqrt(np.maximum(
                    0.0, np.linalg.eigvalsh(obstacle.covariance)[:, -1]))
                radii.append(self.ego_radius_m + 0.5 * np.hypot(
                    obstacle.lengths, obstacle.widths) + self.fixed_margin_m + covariance_radius)
            opti.set_value(problem["obstacle_x"], centers[:, :, 0])
            opti.set_value(problem["obstacle_y"], centers[:, :, 1])
            opti.set_value(problem["safe_radius"], np.stack(radii))
        opti.set_initial(states, problem["last_states"] if problem["last_states"] is not None else reference.states.T)
        opti.set_initial(controls, problem["last_controls"] if problem["last_controls"] is not None else reference.controls.T)
        if slack is not None:
            opti.set_initial(slack, problem["last_slack"] if problem["last_slack"] is not None else 0.0)
        try:
            solution = opti.solve()
            solved_states = np.asarray(solution.value(states), dtype=np.float64).T
            solved_controls = np.asarray(solution.value(controls), dtype=np.float64).T
            slack_max = 0.0 if slack is None else float(np.max(solution.value(slack)))
            problem["last_states"] = solved_states.T
            problem["last_controls"] = solved_controls.T
            problem["last_slack"] = None if slack is None else solution.value(slack)
            status = str(opti.stats().get("return_status", "solved"))
            feasible = True
        except RuntimeError:
            solved_states, solved_controls = reference.states.copy(), reference.controls.copy()
            slack_max, feasible = float("inf"), False
            status = str(opti.stats().get("return_status", "solver_failure"))

        all_h = [barrier_series(solved_states, obstacle, self.ego_radius_m,
                                self.fixed_margin_m, self.sigma_multiplier)
                 for obstacle in request.obstacles]
        h_min = float(np.min(np.concatenate(all_h))) if all_h else float("inf")
        elapsed_ms = (perf_counter() - started) * 1000.0
        if elapsed_ms > self.hard_deadline_ms:
            feasible, status = False, "deadline_exceeded"
        return MpcResult(solved_states, solved_controls, feasible, h_min,
                         slack_max, elapsed_ms, status)
