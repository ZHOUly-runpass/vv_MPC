from __future__ import annotations

"""Optional PyTorch heads. Importing the core package does not require torch."""

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised on lightweight deployments
    torch = None
    Tensor = object  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class C16FeatureAdapter(nn.Module):
        def __init__(self, input_channels: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.projection = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            )

        def forward(self, bev: Tensor) -> Tensor:
            return self.projection(bev)


    class RouteEgoEncoder(nn.Module):
        def __init__(self, route_dim: int = 4, ego_dim: int = 8, hidden_dim: int = 128) -> None:
            super().__init__()
            self.route = nn.Sequential(nn.Linear(route_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.ego = nn.Sequential(nn.Linear(ego_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        def forward(self, route: Tensor, ego: Tensor) -> tuple[Tensor, Tensor]:
            return self.route(route), self.ego(ego)


    class MultiCandidateHead(nn.Module):
        def __init__(
            self,
            hidden_dim: int = 128,
            candidates: int = 7,
            intervals: int = 10,
            control_dim: int = 2,
        ) -> None:
            super().__init__()
            self.candidates = candidates
            self.intervals = intervals
            self.control_dim = control_dim
            self.queries = nn.Parameter(torch.randn(candidates, hidden_dim) * 0.02)
            layer = nn.TransformerDecoderLayer(hidden_dim, nhead=4, batch_first=True)
            self.decoder = nn.TransformerDecoder(layer, num_layers=2)
            self.control_head = nn.Linear(hidden_dim, intervals * control_dim)
            self.logit_head = nn.Linear(hidden_dim, 1)

        def forward(self, scene_tokens: Tensor, return_tokens: bool = False):
            batch = scene_tokens.shape[0]
            queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
            decoded = self.decoder(queries, scene_tokens)
            controls = self.control_head(decoded).reshape(
                batch, self.candidates, self.intervals, self.control_dim
            )
            logits = self.logit_head(decoded).squeeze(-1)
            if return_tokens:
                return controls, logits, decoded
            return controls, logits


    class SafetyPredictionHeads(nn.Module):
        def __init__(self, hidden_dim: int = 128) -> None:
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            self.output = nn.Linear(hidden_dim, 5)

        def forward(self, candidate_tokens: Tensor) -> dict[str, Tensor]:
            values = self.output(self.shared(candidate_tokens))
            return {
                "predicted_h_min": values[..., 0],
                "feasibility_logits": values[..., 1],
                "predicted_correction": values[..., 2],
                "predicted_risk": values[..., 3],
                "predicted_slack": values[..., 4],
            }


    class PlanningSafetyModel(nn.Module):
        """Compact trainable planner over frozen BEV features and route/ego context."""

        def __init__(self, feature_channels: int, candidates: int = 7, intervals: int = 10,
                     ego_dim: int = 5, hidden_dim: int = 128) -> None:
            super().__init__()
            self.config = {
                "feature_channels": feature_channels, "candidates": candidates,
                "intervals": intervals, "ego_dim": ego_dim, "hidden_dim": hidden_dim,
            }
            self.feature = C16FeatureAdapter(feature_channels, hidden_dim)
            self.costmap = nn.Sequential(
                nn.Conv2d(3, hidden_dim // 2, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1), nn.SiLU(),
            )
            self.route_ego = RouteEgoEncoder(4, ego_dim, hidden_dim)
            self.head = MultiCandidateHead(hidden_dim, candidates, intervals, 2)
            self.safety = SafetyPredictionHeads(hidden_dim)

        @staticmethod
        def _tokens(value: Tensor, grid: int = 4) -> Tensor:
            pooled = torch.nn.functional.adaptive_avg_pool2d(value, (grid, grid))
            return pooled.flatten(2).transpose(1, 2)

        def forward(self, features: Tensor, route: Tensor, ego: Tensor, costmap: Tensor) -> dict[str, Tensor]:
            route_tokens, ego_token = self.route_ego(route, ego)
            scene = torch.cat([
                self._tokens(self.feature(features)), self._tokens(self.costmap(costmap)),
                route_tokens, ego_token.unsqueeze(1),
            ], dim=1)
            controls, logits, candidate_tokens = self.head(scene, return_tokens=True)
            return {"controls": controls, "candidate_logits": logits, **self.safety(candidate_tokens)}

else:

    class _TorchRequired:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Install the 'ml' extra to use trainable planning heads")

    C16FeatureAdapter = RouteEgoEncoder = MultiCandidateHead = SafetyPredictionHeads = PlanningSafetyModel = _TorchRequired
