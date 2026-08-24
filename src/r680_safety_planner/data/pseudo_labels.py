from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from ..interfaces import CandidateTrajectory, MpcResult


@dataclass(frozen=True)
class CandidatePseudoLabel:
    role: str
    outcome: str
    feasible: bool
    h_min: float
    slack_max: float
    solve_time_ms: float


def classify_solver_result(result: MpcResult, deadline_ms: float) -> str:
    status = result.status.lower()
    if result.solve_time_ms > deadline_ms or "deadline" in status or "timeout" in status:
        return "timeout"
    if any(token in status for token in ("nan", "exception", "failure", "invalid")):
        return "numeric_failure"
    if not result.feasible:
        return "infeasible"
    return "feasible"


def build_pseudo_labels(
    candidates: Iterable[CandidateTrajectory],
    results: Iterable[MpcResult],
    deadline_ms: float,
) -> list[dict[str, object]]:
    pairs = list(zip(candidates, results, strict=True))
    return [asdict(CandidatePseudoLabel(
        role=candidate.role,
        outcome=classify_solver_result(result, deadline_ms),
        feasible=result.feasible,
        h_min=result.h_min,
        slack_max=result.slack_max,
        solve_time_ms=result.solve_time_ms,
    )) for candidate, result in pairs]
