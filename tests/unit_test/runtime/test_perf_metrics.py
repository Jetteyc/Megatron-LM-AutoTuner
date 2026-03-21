import torch

from AutoTuner.runtime.baseline.perf_metrics import (
    calculate_mfu,
    calculate_per_gpu_mfu,
    get_promised_tflops_fallback,
    normalize_promised_tflops,
    resolve_batch_seqlens_for_flops,
)


def test_normalize_promised_tflops_uses_verl_value_when_finite() -> None:
    assert normalize_promised_tflops(312.0, device_name="NVIDIA A100-SXM4-80GB") == 312.0


def test_normalize_promised_tflops_falls_back_for_rtx_5090() -> None:
    assert normalize_promised_tflops(float("inf"), device_name="NVIDIA GeForce RTX 5090") == 209.5


def test_get_promised_tflops_fallback_accepts_short_device_name() -> None:
    assert get_promised_tflops_fallback("RTX 5090") == 209.5


def test_calculate_mfu_uses_direct_ratio() -> None:
    assert calculate_mfu(52.4, 104.8) == 0.5


def test_calculate_mfu_returns_zero_for_invalid_promised_tflops() -> None:
    assert calculate_mfu(52.4, 0.0) == 0.0


def test_calculate_per_gpu_mfu_divides_by_model_parallel_world_size() -> None:
    assert calculate_per_gpu_mfu(419.0, 209.5, 2) == 1.0


def test_resolve_batch_seqlens_for_flops_uses_nominal_seqlen_for_bshd() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)

    assert resolve_batch_seqlens_for_flops(
        attention_mask,
        shape="bshd",
        nominal_seqlen=4,
    ) == [4, 4]


def test_resolve_batch_seqlens_for_flops_uses_valid_tokens_for_thd() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)

    assert resolve_batch_seqlens_for_flops(
        attention_mask,
        shape="thd",
        nominal_seqlen=4,
    ) == [3, 2]
