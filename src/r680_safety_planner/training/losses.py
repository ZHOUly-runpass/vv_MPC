from __future__ import annotations

import torch
from torch import nn


class PlanningLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.weights = {"control": 1.0, "ranking": 0.5, "h_min": 0.2, "feasible": 0.2,
                        "slack": 0.1, "correction": 0.1, "risk": 0.2}
        if weights:
            self.weights.update(weights)

    def forward(self, prediction: dict[str, torch.Tensor], target: dict[str, object]):
        values = {
            "control": torch.nn.functional.smooth_l1_loss(prediction["controls"], target["target_controls"]),
            "ranking": torch.nn.functional.cross_entropy(prediction["candidate_logits"], target["selected_index"]),
            "h_min": torch.nn.functional.smooth_l1_loss(prediction["predicted_h_min"], target["h_min"]),
            "feasible": torch.nn.functional.binary_cross_entropy_with_logits(prediction["feasibility_logits"], target["feasible"]),
            "slack": torch.nn.functional.smooth_l1_loss(torch.nn.functional.softplus(prediction["predicted_slack"]), target["slack"]),
            "correction": torch.nn.functional.smooth_l1_loss(torch.nn.functional.softplus(prediction["predicted_correction"]), target["correction"]),
            "risk": torch.nn.functional.binary_cross_entropy_with_logits(prediction["predicted_risk"], target["risk"]),
        }
        total = sum(self.weights[name] * value for name, value in values.items())
        return total, {**values, "total": total}


def planning_metrics(prediction: dict[str, torch.Tensor], target: dict[str, object]) -> dict[str, float]:
    with torch.no_grad():
        return {
            "control_mae": float(torch.mean(torch.abs(prediction["controls"] - target["target_controls"])).item()),
            "ranking_accuracy": float((prediction["candidate_logits"].argmax(-1) == target["selected_index"]).float().mean().item()),
            "feasible_accuracy": float(((prediction["feasibility_logits"] >= 0) == (target["feasible"] >= 0.5)).float().mean().item()),
        }
