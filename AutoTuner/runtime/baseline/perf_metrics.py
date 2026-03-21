import math
import os

import torch

_KNOWN_DEVICE_PROMISED_TFLOPS = {
    "NVIDIA GeForce RTX 5090": 209.5,
    "RTX 5090": 209.5,
}


def _normalize_env_peak_flops(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        return 0.0
    if parsed > 1e6:
        return parsed / 1e12
    return parsed


def get_promised_tflops_fallback(device_name: str | None = None) -> float:
    env_peak_flops = os.getenv("GPU_PEAK_FLOPS")
    if env_peak_flops:
        try:
            return _normalize_env_peak_flops(env_peak_flops)
        except ValueError:
            return 0.0

    resolved_device_name = device_name
    if resolved_device_name is None and torch.cuda.is_available():
        resolved_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if not resolved_device_name:
        return 0.0

    for known_name, promised_tflops in sorted(
        _KNOWN_DEVICE_PROMISED_TFLOPS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if known_name in resolved_device_name:
            return promised_tflops
    return 0.0


def normalize_promised_tflops(
    promised_tflops: float | int | None, device_name: str | None = None
) -> float:
    if promised_tflops is None:
        return get_promised_tflops_fallback(device_name)

    normalized = float(promised_tflops)
    if math.isfinite(normalized) and normalized > 0:
        return normalized
    return get_promised_tflops_fallback(device_name)


def calculate_mfu(achieved_tflops: float, promised_tflops: float) -> float:
    achieved = float(achieved_tflops)
    promised = float(promised_tflops)
    if not math.isfinite(achieved) or achieved <= 0:
        return 0.0
    if not math.isfinite(promised) or promised <= 0:
        return 0.0
    return achieved / promised


def calculate_per_gpu_mfu(
    total_achieved_tflops: float,
    promised_tflops: float,
    model_parallel_world_size: int,
) -> float:
    parallel_world = int(model_parallel_world_size)
    if parallel_world <= 0:
        return 0.0
    return calculate_mfu(float(total_achieved_tflops) / float(parallel_world), promised_tflops)


def resolve_batch_seqlens_for_flops(
    attention_mask: torch.Tensor,
    *,
    shape: str,
    nominal_seqlen: int,
) -> list[int]:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be 2D [batch, seqlen], got shape={tuple(attention_mask.shape)}"
        )

    batch_size = int(attention_mask.shape[0])
    if shape == "bshd":
        return [int(nominal_seqlen)] * batch_size

    if attention_mask.is_floating_point():
        attention_mask = attention_mask > 0
    return attention_mask.to(torch.int64).sum(dim=1).tolist()
