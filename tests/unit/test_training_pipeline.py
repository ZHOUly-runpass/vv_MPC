import numpy as np
import pytest


torch = pytest.importorskip("torch")

from r680_safety_planner.models import PlanningSafetyModel
from r680_safety_planner.training import PlanningLoss


def test_model_and_loss_are_finite():
    model = PlanningSafetyModel(16, candidates=7, intervals=10, ego_dim=5, hidden_dim=32)
    prediction = model(torch.zeros(2, 16, 8, 8), torch.zeros(2, 16, 4), torch.zeros(2, 5), torch.zeros(2, 3, 32, 32))
    target = {
        "target_controls": torch.zeros(2, 7, 10, 2), "selected_index": torch.zeros(2, dtype=torch.long),
        "h_min": torch.zeros(2, 7), "feasible": torch.ones(2, 7), "slack": torch.zeros(2, 7),
        "correction": torch.zeros(2, 7), "risk": torch.zeros(2, 7),
    }
    loss, parts = PlanningLoss()(prediction, target)
    assert np.isfinite(float(loss)); assert all(torch.isfinite(value) for value in parts.values())
