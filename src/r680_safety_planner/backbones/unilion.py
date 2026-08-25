from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterator, Mapping

import numpy as np

from ..interfaces import FrozenSceneFeatures, LidarFrame
from ..lidar import UniLionPointFieldContract
from .protocol import FrozenLidarBackbone


@dataclass(frozen=True)
class ModuleWeightReport:
    module: str
    tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    layout_converted_keys: tuple[str, ...] = ()
    load_method: str = "strict_named_parameter_buffer_copy"


@dataclass(frozen=True)
class UniLionInferenceReport:
    input_points: int
    voxel_count: int
    output_shape: tuple[int, ...]
    model_time_ms: float
    total_time_ms: float
    peak_memory_bytes: int
    maximum_repeat_difference: float | None
    hook_statistics: Mapping[str, Mapping[str, object]]


def spconv_layout_permutation(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Return the legacy-spconv to spconv2 kernel permutation when required."""

    if len(source_shape) != 5 or len(target_shape) != 5:
        return None
    permutation = (0, 2, 3, 4, 1)
    converted = tuple(source_shape[index] for index in permutation)
    return permutation if converted == target_shape else None


@contextmanager
def _repository_context(repository: Path) -> Iterator[None]:
    """Temporarily expose the official ``projects`` namespace and relative CUDA sources."""

    previous_directory = Path.cwd()
    repository_text = str(repository)
    inserted = repository_text not in sys.path
    if inserted:
        sys.path.insert(0, repository_text)
    os.chdir(repository)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        if inserted:
            sys.path.remove(repository_text)


class _BevFeatureHook:
    def __init__(self) -> None:
        self.statistics: dict[str, dict[str, object]] = {}
        self._handles: list[Any] = []

    @staticmethod
    def _tensors(output: Any) -> list[Any]:
        if hasattr(output, "features"):
            return [output.features]
        if isinstance(output, (tuple, list)):
            return [value for value in output if hasattr(value, "shape")]
        return [output] if hasattr(output, "shape") else []

    def register(self, name: str, module: Any) -> None:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensors = self._tensors(output)
            if not tensors:
                raise RuntimeError(f"feature hook {name} produced no tensor")
            finite = all(bool(tensor.isfinite().all().item()) for tensor in tensors)
            self.statistics[name] = {
                "shapes": [list(tensor.shape) for tensor in tensors],
                "finite": finite,
                "means": [float(tensor.float().mean().item()) for tensor in tensors],
                "variances": [float(tensor.float().var(unbiased=False).item()) for tensor in tensors],
            }

        self._handles.append(module.register_forward_hook(capture))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class UniLionFrozenBackbone(FrozenLidarBackbone):
    """Frozen, single-frame LiDAR path extracted from the official UniLION LCT model.

    This adapter intentionally excludes camera, temporal and task heads. It is an
    initialization/validation backend until real C16 data passes the project gates.
    Heavy ML dependencies are imported lazily so the ROS/control environment can
    still import the rest of the project without MMCV or CUDA installed.
    """

    MODULE_PREFIXES = {
        "pts_voxel_encoder": "pts_voxel_encoder.",
        "pts_backbone": "pts_backbone.",
        "bev_backbone": "bev_backbone.",
        "pts_neck": "pts_neck.",
    }

    def __init__(
        self,
        repository: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        point_contract: UniLionPointFieldContract | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = device
        self.point_contract = point_contract or UniLionPointFieldContract()
        self._validate_paths()
        self._last_report: UniLionInferenceReport | None = None
        self._torch: Any = None
        self._hook = _BevFeatureHook()
        self.weight_report = self._build_and_load()

    @property
    def backend_name(self) -> str:
        return "unilion_lct_single_frame_lidar_initialization_candidate"

    def _validate_paths(self) -> None:
        if not (self.repository / "projects" / "mmdet3d_plugin").is_dir():
            raise FileNotFoundError("repository does not contain projects/mmdet3d_plugin")
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if not self.device.startswith("cuda"):
            raise ValueError("the pinned DynamicPillarVFE3D implementation requires a CUDA device")

    def _build_and_load(self) -> tuple[ModuleWeightReport, ...]:
        with _repository_context(self.repository):
            self._torch = importlib.import_module("torch")
            if not self._torch.cuda.is_available():
                raise RuntimeError("CUDA is required for UniLION")
            importlib.import_module("projects.mmdet3d_plugin")
            mmcv = importlib.import_module("mmcv")
            builder = importlib.import_module("mmdet3d.models.builder")
            safetensors = importlib.import_module("safetensors.torch")

            cfg = mmcv.Config.fromfile(str(self.config_path))
            model_cfg = cfg.model
            self.pts_voxel_encoder = builder.build_voxel_encoder(model_cfg.pts_voxel_encoder)
            self.pts_backbone = builder.build_backbone(model_cfg.pts_backbone)
            self.map2bev = builder.build_middle_encoder(model_cfg.map2bev)
            self.bev_backbone = builder.build_backbone(model_cfg.bev_backbone)
            self.pts_neck = builder.build_neck(model_cfg.pts_neck)

            state = safetensors.load_file(str(self.checkpoint_path), device="cpu")
            reports: list[ModuleWeightReport] = []
            for module_name, prefix in self.MODULE_PREFIXES.items():
                module_state = {
                    key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
                }
                if not module_state:
                    raise RuntimeError(f"checkpoint contains no tensors for {module_name}")
                module = getattr(self, module_name)
                # spconv2's state_dict compatibility hook may expose a legacy
                # save shape. Compare against the actual Parameter layout used
                # by kernels so KCRS checkpoints are converted before loading.
                target_state = {
                    **dict(module.named_parameters()),
                    **dict(module.named_buffers()),
                }
                missing_keys = tuple(sorted(set(target_state) - set(module_state)))
                unexpected_keys = tuple(sorted(set(module_state) - set(target_state)))
                if missing_keys or unexpected_keys:
                    raise RuntimeError(
                        f"checkpoint key mismatch for {module_name}: "
                        f"missing={missing_keys}, unexpected={unexpected_keys}"
                    )
                converted_keys: list[str] = []
                with self._torch.no_grad():
                    for key, target in target_state.items():
                        value = module_state[key]
                        if tuple(value.shape) != tuple(target.shape):
                            permutation = spconv_layout_permutation(
                                tuple(value.shape), tuple(target.shape)
                            )
                            if permutation is None:
                                raise RuntimeError(
                                    f"checkpoint shape mismatch for {module_name}.{key}: "
                                    f"source={tuple(value.shape)}, target={tuple(target.shape)}"
                                )
                            value = value.permute(permutation).contiguous()
                            converted_keys.append(key)
                        target.copy_(value.to(device=target.device, dtype=target.dtype))
                report = ModuleWeightReport(
                    module=module_name,
                    tensor_count=len(module_state),
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                    layout_converted_keys=tuple(converted_keys),
                )
                if report.missing_keys or report.unexpected_keys:
                    raise RuntimeError(f"non-strict checkpoint load for {module_name}: {report}")
                reports.append(report)

            for module_name in (*self.MODULE_PREFIXES, "map2bev"):
                module = getattr(self, module_name).to(self.device).eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

            self._hook.register("sparse_lidar_backbone", self.pts_backbone)
            self._hook.register("height_compressed_bev", self.map2bev)
            self._hook.register("bev_backbone", self.bev_backbone)
            self._hook.register("bev_neck", self.pts_neck)
            return tuple(reports)

    def _forward_tensor(self, points: np.ndarray) -> tuple[Any, int, float, int]:
        torch = self._torch
        point_tensor = torch.from_numpy(points).to(self.device, non_blocking=False)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode():
            voxel_features, voxel_coords = self.pts_voxel_encoder([point_tensor])
            sparse_features = self.pts_backbone(voxel_features, voxel_coords, batch_size=1)
            dense_bev = self.map2bev(sparse_features)
            bev_levels = self.bev_backbone(dense_bev)
            neck_levels = self.pts_neck(bev_levels)
            if not isinstance(neck_levels, (tuple, list)) or not neck_levels:
                raise RuntimeError("UniLION neck must return a non-empty feature pyramid")
            spatial_shapes = {tuple(level.shape[-2:]) for level in neck_levels}
            if len(spatial_shapes) != 1:
                raise RuntimeError(f"cannot concatenate BEV levels with shapes {spatial_shapes}")
            output = torch.cat(tuple(neck_levels), dim=1).float()
        torch.cuda.synchronize(self.device)
        elapsed_ms = (perf_counter() - started) * 1000.0
        peak_memory = int(torch.cuda.max_memory_allocated(self.device))
        if not bool(output.isfinite().all().item()):
            raise RuntimeError("UniLION BEV output contains NaN or Inf")
        return output, int(voxel_coords.shape[0]), elapsed_ms, peak_memory

    def infer(self, frame: LidarFrame) -> FrozenSceneFeatures:
        started = perf_counter()
        points = self.point_contract.adapt(frame)
        output, voxel_count, model_ms, peak_memory = self._forward_tensor(points)
        bev = output.detach().cpu().numpy()
        total_ms = (perf_counter() - started) * 1000.0
        self._last_report = UniLionInferenceReport(
            input_points=int(points.shape[0]),
            voxel_count=voxel_count,
            output_shape=tuple(bev.shape),
            model_time_ms=model_ms,
            total_time_ms=total_ms,
            peak_memory_bytes=peak_memory,
            maximum_repeat_difference=None,
            hook_statistics=dict(self._hook.statistics),
        )
        features = FrozenSceneFeatures(
            bev_feature=bev,
            source_backend=self.backend_name,
            timestamp_s=frame.timestamp_s,
            quality={
                "input_points": float(points.shape[0]),
                "voxel_count": float(voxel_count),
                "model_time_ms": model_ms,
                "total_time_ms": total_ms,
                "peak_memory_bytes": float(peak_memory),
            },
        )
        features.validate()
        return features

    def verify_repeatability(self, frame: LidarFrame) -> float:
        points = self.point_contract.adapt(frame)
        first, _, _, _ = self._forward_tensor(points)
        second, _, _, _ = self._forward_tensor(points)
        difference = float((first - second).abs().max().item())
        if not np.isfinite(difference):
            raise RuntimeError("repeatability comparison is not finite")
        if self._last_report is not None:
            self._last_report = UniLionInferenceReport(
                **{**asdict(self._last_report), "maximum_repeat_difference": difference}
            )
        return difference

    def healthcheck(self) -> dict[str, object]:
        frozen = all(
            not parameter.requires_grad
            for module_name in (*self.MODULE_PREFIXES, "map2bev")
            for parameter in getattr(self, module_name).parameters()
        )
        return {
            "healthy": bool(self._torch.cuda.is_available() and frozen),
            "backend": self.backend_name,
            "device": self.device,
            "weights_strict": all(
                not report.missing_keys and not report.unexpected_keys
                for report in self.weight_report
            ),
            "modules_frozen": frozen,
            "weight_report": [asdict(report) for report in self.weight_report],
            "point_contract": self.point_contract.describe(),
            "last_inference": None if self._last_report is None else asdict(self._last_report),
            "validation_status": "initialization_candidate_not_c16_verified",
        }

    def close(self) -> None:
        self._hook.close()
