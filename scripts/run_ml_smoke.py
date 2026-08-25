#!/usr/bin/env python3
from __future__ import annotations

import json

import torch

from r680_safety_planner.models import (
    C16FeatureAdapter, MultiCandidateHead, RouteEgoEncoder, SafetyPredictionHeads,
)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the development-machine ML smoke test")
    device = torch.device("cuda")
    adapter = C16FeatureAdapter(64, hidden_dim=128).to(device)
    context = RouteEgoEncoder(route_dim=4, ego_dim=8, hidden_dim=128).to(device)
    planner = MultiCandidateHead(128, candidates=7, intervals=10, control_dim=2).to(device)
    safety = SafetyPredictionHeads(128).to(device)
    bev = torch.randn(2, 64, 20, 20, device=device)
    route = torch.randn(2, 32, 4, device=device)
    ego = torch.randn(2, 8, device=device)
    bev_tokens = adapter(bev).flatten(2).transpose(1, 2)
    route_tokens, ego_token = context(route, ego)
    scene_tokens = torch.cat([bev_tokens, route_tokens, ego_token[:, None, :]], dim=1)
    controls, logits = planner(scene_tokens)
    candidate_tokens = planner.queries.unsqueeze(0).expand(2, -1, -1)
    predictions = safety(candidate_tokens)
    loss = controls.square().mean() + logits.square().mean()
    loss += sum(value.square().mean() for value in predictions.values())
    loss.backward()
    finite = bool(torch.isfinite(loss).item())
    report = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "controls_shape": list(controls.shape), "logits_shape": list(logits.shape),
        "finite_loss": finite, "loss": float(loss.detach().cpu()),
    }
    print(json.dumps(report, indent=2))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
