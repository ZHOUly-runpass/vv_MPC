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

        def forward(self, scene_tokens: Tensor) -> tuple[Tensor, Tensor]:
            batch = scene_tokens.shape[0]
            queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
            decoded = self.decoder(queries, scene_tokens)
            controls = self.control_head(decoded).reshape(
                batch, self.candidates, self.intervals, self.control_dim
            )
            logits = self.logit_head(decoded).squeeze(-1)
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

else:

    class _TorchRequired:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Install the 'ml' extra to use trainable planning heads")

    C16FeatureAdapter = RouteEgoEncoder = MultiCandidateHead = SafetyPredictionHeads = _TorchRequired

