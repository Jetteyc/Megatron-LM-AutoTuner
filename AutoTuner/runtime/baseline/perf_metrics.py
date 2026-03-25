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
    gpu_count: int,
) -> float:
    total_gpus = int(gpu_count)
    if total_gpus <= 0:
        return 0.0
    return calculate_mfu(float(total_achieved_tflops) / float(total_gpus), promised_tflops)


def calculate_estimated_model_flops(
    total_achieved_tflops: float,
    step_time_s: float,
) -> float:
    achieved = float(total_achieved_tflops)
    step_time = float(step_time_s)
    if not math.isfinite(achieved) or achieved <= 0:
        return 0.0
    if not math.isfinite(step_time) or step_time <= 0:
        return 0.0
    return achieved * step_time * 1e12


def build_mfu_breakdown(
    *,
    estimated_model_flops: float,
    step_time_s: float,
    gpu_count: int,
    gpu_top_flops_tflops: float,
    total_achieved_tflops: float,
    flops_source: str,
    raw_estimated_tflops: float,
    estimated_tflops_scale: float,
) -> dict[str, float | int | str]:
    total_gpus = max(1, int(gpu_count))
    step_time = float(step_time_s)
    gpu_top_flops = float(gpu_top_flops_tflops)
    total_achieved = float(total_achieved_tflops)
    per_gpu_achieved_tflops = (
        total_achieved / float(total_gpus) if total_gpus > 0 else 0.0
    )
    mfu = calculate_mfu(per_gpu_achieved_tflops, gpu_top_flops)
    return {
        "formula": "estimated_model_flops / (gpu_world_size * step_time_s * 1e12) / gpu_top_flops_tflops",
        "flops_source": flops_source,
        "raw_estimated_tflops": float(raw_estimated_tflops),
        "estimated_tflops_scale": float(estimated_tflops_scale),
        "estimated_model_flops": float(estimated_model_flops),
        "step_time_s": step_time,
        "gpu_world_size": total_gpus,
        "gpu_top_flops_tflops": gpu_top_flops,
        "estimated_achieved_tflops_total": total_achieved,
        "estimated_achieved_tflops_per_gpu": per_gpu_achieved_tflops,
        "mfu": mfu,
    }


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
