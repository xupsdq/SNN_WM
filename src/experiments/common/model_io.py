from __future__ import annotations

import glob
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import torch

from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder

from .cache import CHECKPOINT_STATE_CACHE
from .profiling import GLOBAL_PROFILE_STATS


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float, include_target_norm: bool = False) -> None:
    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor
            if include_target_norm and hasattr(net.layer3, "target_norm"):
                net.layer3.target_norm *= scaling_factor


def load_model_state_dict_cached(model_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    resolved = Path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    GLOBAL_PROFILE_STATS.increment("model_load_calls")
    return CHECKPOINT_STATE_CACHE.get(resolved, map_location=str(device))


def resolve_model_paths(
    model_path: str | Path | None = None,
    model_paths: Iterable[str | Path] | None = None,
    model_path_glob: str = "",
) -> List[Path]:
    resolved: List[Path] = []
    seen: set[str] = set()
    candidates: List[str | Path] = []
    if model_path is not None and str(model_path).strip():
        candidates.append(model_path)
    if model_paths is not None:
        candidates.extend(model_paths)
    if str(model_path_glob).strip():
        candidates.extend(sorted(glob.glob(str(model_path_glob).strip())))
    for item in candidates:
        path_obj = Path(item).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Model not found: {path_obj}")
        key = str(path_obj)
        if key in seen:
            continue
        resolved.append(path_obj)
        seen.add(key)
    if not resolved:
        raise ValueError("At least one valid model path is required.")
    return resolved


def randomized_state_dict_from_checkpoint(
    model_path: str | Path,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    base_state = load_model_state_dict_cached(model_path, device=device)
    randomized: dict[str, torch.Tensor] = {}
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    for key, value in base_state.items():
        tensor = value.detach().cpu().clone()
        if not torch.is_floating_point(tensor) or tensor.numel() <= 1:
            randomized[key] = tensor
            continue
        flat = tensor.reshape(-1)
        perm = torch.randperm(flat.numel(), generator=rng)
        randomized[key] = flat[perm].reshape_as(tensor)
    return randomized


def load_model_and_encoder_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dt: float,
    max_duration_ms: float,
    *,
    include_target_norm: bool = False,
    network_factory: Callable[..., SDNN_Network] = SDNN_Network,
    encoder_factory: Callable[..., DoGSpikeEncoder] = DoGSpikeEncoder,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    net = network_factory(device=str(device)).to(device)
    net.load_state_dict(state_dict)
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U, include_target_norm=include_target_norm)
    net.eval()
    encoder = encoder_factory(dt=dt, max_duration=max_duration_ms * ms, device=str(device))
    return net, encoder


def load_model_and_encoder(
    model_path: str | Path,
    device: torch.device,
    dt: float,
    max_duration_ms: float,
    *,
    include_target_norm: bool = False,
    network_factory: Callable[..., SDNN_Network] = SDNN_Network,
    encoder_factory: Callable[..., DoGSpikeEncoder] = DoGSpikeEncoder,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    state_dict = load_model_state_dict_cached(model_path, device=device)
    return load_model_and_encoder_from_state_dict(
        state_dict=state_dict,
        device=device,
        dt=dt,
        max_duration_ms=max_duration_ms,
        include_target_norm=include_target_norm,
        network_factory=network_factory,
        encoder_factory=encoder_factory,
    )
