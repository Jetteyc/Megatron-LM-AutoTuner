from dataclasses import asdict, dataclass

from AutoTuner.utils.runtime_config import (
    DEFAULT_DP_ALLREDUCE_BANDWIDTH_GBPS,
    DEFAULT_DP_ALLREDUCE_LATENCY_US,
)


@dataclass(frozen=True)
class StageParamStats:
    pp_rank: int
    runtime_layer_count: int
    runtime_total_param_bytes: int
    runtime_decoder_param_bytes: int
    total_device_bytes: int

    @property
    def runtime_non_decoder_param_bytes(self) -> int:
        return max(0, self.runtime_total_param_bytes - self.runtime_decoder_param_bytes)


@dataclass(frozen=True)
class StageTimingStats:
    pp_rank: int
    runtime_layer_count: int
    runtime_forward_time_s: float
    runtime_backward_time_s: float
    runtime_forward_total_time_s: float
    runtime_backward_total_time_s: float
    num_microbatches: int


class RuntimeSimulationOOMError(RuntimeError):
    """Raised when the simulated full stage parameters cannot fit on device."""


def build_chunk_layer_counts(
    total_layers: int, pp_size: int, vpp_size: int
) -> list[int]:
    total_layers = max(0, int(total_layers))
    pp_size = max(1, int(pp_size))
    vpp_size = max(1, int(vpp_size))
    total_chunks = pp_size * vpp_size
    counts: list[int] = []
    for chunk_id in range(total_chunks):
        start = (chunk_id * total_layers) // total_chunks
        end = ((chunk_id + 1) * total_layers) // total_chunks
        counts.append(max(0, end - start))
    return counts


def build_pp_stage_layer_counts(
    total_layers: int, pp_size: int, vpp_size: int
) -> list[int]:
    chunk_counts = build_chunk_layer_counts(total_layers, pp_size, vpp_size)
    stage_counts: list[int] = []
    for pp_rank in range(max(1, int(pp_size))):
        stage_counts.append(
            sum(
                chunk_counts[vp_stage * pp_size + pp_rank]
                for vp_stage in range(max(1, int(vpp_size)))
            )
        )
    return stage_counts


def simulate_pp_step_durations(
    stage_times_s: list[float], num_microbatches: int
) -> list[float]:
    if not stage_times_s or num_microbatches <= 0:
        return []

    num_stages = len(stage_times_s)
    total_steps = num_microbatches + num_stages - 1
    step_durations: list[float] = []
    for step_idx in range(total_steps):
        start_stage = max(0, step_idx - (num_microbatches - 1))
        end_stage = min(num_stages - 1, step_idx)
        step_durations.append(max(stage_times_s[start_stage : end_stage + 1]))
    return step_durations


def simulate_pp_time_s(stage_times_s: list[float], num_microbatches: int) -> float:
    return sum(simulate_pp_step_durations(stage_times_s, num_microbatches))


def simulate_train_pipeline_1f1b(
    stage_forward_times_s: list[float],
    stage_backward_times_s: list[float],
    num_microbatches: int,
) -> dict:
    if (
        not stage_forward_times_s
        or not stage_backward_times_s
        or len(stage_forward_times_s) != len(stage_backward_times_s)
        or num_microbatches <= 0
    ):
        return {
            "total_time_s": 0.0,
            "num_scheduled_tasks": 0,
            "max_stage_time_s": 0.0,
            "avg_stage_time_s": 0.0,
        }

    num_stages = len(stage_forward_times_s)
    stage_available_at_s = [0.0] * num_stages
    warmup_remaining = [
        min(num_microbatches, num_stages - stage_idx - 1)
        for stage_idx in range(num_stages)
    ]
    next_forward_idx = [0] * num_stages
    next_backward_idx = [0] * num_stages
    num_forward_done = [0] * num_stages
    forward_done_at: list[list[float | None]] = [
        [None] * num_microbatches for _ in range(num_stages)
    ]
    backward_done_at: list[list[float | None]] = [
        [None] * num_microbatches for _ in range(num_stages)
    ]
    scheduled_task_durations_s: list[float] = []

    def _next_task(stage_idx: int):
        stage_free_at = stage_available_at_s[stage_idx]
        forward_ready_at = None
        backward_ready_at = None
        fwd_idx = next_forward_idx[stage_idx]
        bwd_idx = next_backward_idx[stage_idx]

        if fwd_idx < num_microbatches:
            if stage_idx == 0:
                forward_ready_at = 0.0
            else:
                forward_ready_at = forward_done_at[stage_idx - 1][fwd_idx]

        if bwd_idx < num_microbatches and bwd_idx < num_forward_done[stage_idx]:
            backward_dep_at = forward_done_at[stage_idx][bwd_idx]
            next_stage_dep_at = (
                0.0
                if stage_idx == (num_stages - 1)
                else backward_done_at[stage_idx + 1][bwd_idx]
            )
            if backward_dep_at is not None and next_stage_dep_at is not None:
                backward_ready_at = max(backward_dep_at, next_stage_dep_at)

        if warmup_remaining[stage_idx] > 0:
            if forward_ready_at is None:
                return None
            start_at = max(stage_free_at, forward_ready_at)
            end_at = start_at + stage_forward_times_s[stage_idx]
            return ("forward", fwd_idx, start_at, end_at)

        forward_start_at = (
            None if forward_ready_at is None else max(stage_free_at, forward_ready_at)
        )
        backward_start_at = (
            None if backward_ready_at is None else max(stage_free_at, backward_ready_at)
        )

        if forward_start_at is None and backward_start_at is None:
            return None
        if backward_start_at is None:
            return (
                "forward",
                fwd_idx,
                forward_start_at,
                forward_start_at + stage_forward_times_s[stage_idx],
            )
        if forward_start_at is None:
            return (
                "backward",
                bwd_idx,
                backward_start_at,
                backward_start_at + stage_backward_times_s[stage_idx],
            )
        if backward_start_at <= forward_start_at:
            return (
                "backward",
                bwd_idx,
                backward_start_at,
                backward_start_at + stage_backward_times_s[stage_idx],
            )
        return (
            "forward",
            fwd_idx,
            forward_start_at,
            forward_start_at + stage_forward_times_s[stage_idx],
        )

    total_tasks = 2 * num_stages * num_microbatches
    scheduled_tasks = 0
    while scheduled_tasks < total_tasks:
        candidates = []
        for stage_idx in range(num_stages):
            task = _next_task(stage_idx)
            if task is not None:
                candidates.append((task[2], stage_idx, task))
        if not candidates:
            raise RuntimeError(
                "No schedulable PP tasks found while building training PP schedule"
            )

        _, stage_idx, task = min(candidates, key=lambda item: (item[0], item[1]))
        task_type, microbatch_idx, start_at, end_at = task
        stage_available_at_s[stage_idx] = end_at
        scheduled_task_durations_s.append(end_at - start_at)
        if task_type == "forward":
            forward_done_at[stage_idx][microbatch_idx] = end_at
            next_forward_idx[stage_idx] += 1
            num_forward_done[stage_idx] += 1
            if warmup_remaining[stage_idx] > 0:
                warmup_remaining[stage_idx] -= 1
        else:
            backward_done_at[stage_idx][microbatch_idx] = end_at
            next_backward_idx[stage_idx] += 1
        scheduled_tasks += 1

    total_time_s = max(stage_available_at_s, default=0.0)
    return {
        "total_time_s": total_time_s,
        "num_scheduled_tasks": scheduled_tasks,
        "max_stage_time_s": max(scheduled_task_durations_s, default=0.0),
        "avg_stage_time_s": (
            sum(scheduled_task_durations_s) / len(scheduled_task_durations_s)
            if scheduled_task_durations_s
            else 0.0
        ),
    }


def estimate_dp_allreduce_time_s(
    payload_bytes: int,
    dp_world_size: int,
    bandwidth_gbps: float,
    latency_s: float,
) -> float:
    if payload_bytes <= 0 or dp_world_size <= 1 or bandwidth_gbps <= 0:
        return 0.0

    bandwidth_bytes_per_s = bandwidth_gbps * 1e9
    ring_bytes = 2.0 * (dp_world_size - 1) * float(payload_bytes) / float(dp_world_size)
    latency_cost = 2.0 * max(0, dp_world_size - 1) * latency_s
    return latency_cost + (ring_bytes / bandwidth_bytes_per_s)


def estimate_full_stage_param_bytes(
    stage_stats: list[StageParamStats], full_stage_layer_counts: list[int]
) -> list[int]:
    full_stage_param_bytes: list[int] = []
    for stats, full_layer_count in zip(stage_stats, full_stage_layer_counts):
        per_layer_bytes = 0.0
        if stats.runtime_layer_count > 0:
            per_layer_bytes = float(stats.runtime_decoder_param_bytes) / float(
                stats.runtime_layer_count
            )
        total_bytes = stats.runtime_non_decoder_param_bytes + (
            per_layer_bytes * float(full_layer_count)
        )
        full_stage_param_bytes.append(int(round(total_bytes)))
    return full_stage_param_bytes


def validate_full_stage_param_fit(
    stage_stats: list[StageParamStats],
    full_stage_param_bytes: list[int],
) -> None:
    if len(stage_stats) != len(full_stage_param_bytes):
        raise ValueError("stage_stats and full_stage_param_bytes must have same length")

    oom_messages: list[str] = []
    for stats, param_bytes in zip(stage_stats, full_stage_param_bytes):
        if param_bytes <= stats.total_device_bytes:
            continue
        oom_messages.append(
            "pp_rank={pp_rank} required={required} available={available}".format(
                pp_rank=stats.pp_rank,
                required=param_bytes,
                available=stats.total_device_bytes,
            )
        )

    if oom_messages:
        raise RuntimeSimulationOOMError(
            "Simulated config OOM: full-model stage parameters exceed device memory for "
            + ", ".join(oom_messages)
        )


def _infer_backward_times_from_observed_pp_time(
    stage_forward_times_s: list[float],
    observed_pp_compute_time_s: float,
    num_microbatches: int,
) -> list[float]:
    if not stage_forward_times_s or num_microbatches <= 0:
        return [0.0 for _ in stage_forward_times_s]

    reference_backward_times_s = [
        max(0.0, float(stage_time_s)) for stage_time_s in stage_forward_times_s
    ]
    baseline_pipeline = simulate_train_pipeline_1f1b(
        stage_forward_times_s=stage_forward_times_s,
        stage_backward_times_s=[0.0 for _ in stage_forward_times_s],
        num_microbatches=num_microbatches,
    )
    baseline_total_time_s = baseline_pipeline["total_time_s"]
    target_total_time_s = max(baseline_total_time_s, float(observed_pp_compute_time_s))
    if target_total_time_s <= baseline_total_time_s:
        return [0.0 for _ in stage_forward_times_s]

    low = 0.0
    high = 1.0
    while True:
        candidate_backward_times_s = [
            stage_time_s * high for stage_time_s in reference_backward_times_s
        ]
        candidate_total_time_s = simulate_train_pipeline_1f1b(
            stage_forward_times_s=stage_forward_times_s,
            stage_backward_times_s=candidate_backward_times_s,
            num_microbatches=num_microbatches,
        )["total_time_s"]
        if candidate_total_time_s >= target_total_time_s or high >= 1024.0:
            break
        high *= 2.0

    for _ in range(48):
        mid = (low + high) / 2.0
        candidate_backward_times_s = [
            stage_time_s * mid for stage_time_s in reference_backward_times_s
        ]
        candidate_total_time_s = simulate_train_pipeline_1f1b(
            stage_forward_times_s=stage_forward_times_s,
            stage_backward_times_s=candidate_backward_times_s,
            num_microbatches=num_microbatches,
        )["total_time_s"]
        if candidate_total_time_s < target_total_time_s:
            low = mid
        else:
            high = mid

    return [stage_time_s * high for stage_time_s in reference_backward_times_s]


def _resolve_stage_timing_stats(
    stage_timing_stats: list[StageTimingStats],
    observed_pp_compute_time_s: float,
    num_microbatches: int,
) -> tuple[list[StageTimingStats], str]:
    runtime_stage_forward_times_s = [
        max(0.0, float(stats.runtime_forward_time_s)) for stats in stage_timing_stats
    ]
    runtime_stage_backward_times_s = [
        max(0.0, float(stats.runtime_backward_time_s)) for stats in stage_timing_stats
    ]
    raw_pipeline = simulate_train_pipeline_1f1b(
        stage_forward_times_s=runtime_stage_forward_times_s,
        stage_backward_times_s=runtime_stage_backward_times_s,
        num_microbatches=num_microbatches,
    )
    raw_forward_total_time_s = sum(runtime_stage_forward_times_s)
    raw_backward_total_time_s = sum(runtime_stage_backward_times_s)
    if raw_backward_total_time_s > max(1e-6, raw_forward_total_time_s * 0.05):
        return stage_timing_stats, "measured"
    if observed_pp_compute_time_s <= raw_pipeline["total_time_s"] * 1.05:
        return stage_timing_stats, "measured"

    inferred_backward_times_s = _infer_backward_times_from_observed_pp_time(
        stage_forward_times_s=runtime_stage_forward_times_s,
        observed_pp_compute_time_s=observed_pp_compute_time_s,
        num_microbatches=num_microbatches,
    )
    resolved_stage_timing_stats: list[StageTimingStats] = []
    for stats, inferred_backward_time_s in zip(
        stage_timing_stats, inferred_backward_times_s
    ):
        stage_num_microbatches = max(0, int(stats.num_microbatches))
        inferred_backward_total_time_s = (
            inferred_backward_time_s * float(stage_num_microbatches)
        )
        resolved_stage_timing_stats.append(
            StageTimingStats(
                pp_rank=stats.pp_rank,
                runtime_layer_count=stats.runtime_layer_count,
                runtime_forward_time_s=stats.runtime_forward_time_s,
                runtime_backward_time_s=inferred_backward_time_s,
                runtime_forward_total_time_s=stats.runtime_forward_total_time_s,
                runtime_backward_total_time_s=inferred_backward_total_time_s,
                num_microbatches=stats.num_microbatches,
            )
        )
    return resolved_stage_timing_stats, "inferred_from_observed_pp_compute_time"


def simulate_full_iteration(
    observed_iteration_time_s: float,
    num_microbatches: int,
    full_stage_layer_counts: list[int],
    runtime_stage_layer_counts: list[int],
    stage_param_stats: list[StageParamStats],
    stage_timing_stats: list[StageTimingStats],
    dp_world_size: int,
    include_runtime_dp_in_observed_time: bool = True,
    bandwidth_gbps: float | None = None,
    latency_s: float | None = None,
) -> dict:
    if len(full_stage_layer_counts) != len(runtime_stage_layer_counts):
        raise ValueError("full/runtime PP stage counts must have identical lengths")
    if len(stage_param_stats) != len(full_stage_layer_counts):
        raise ValueError("stage_param_stats must match PP stage count")
    if len(stage_timing_stats) != len(full_stage_layer_counts):
        raise ValueError("stage_timing_stats must match PP stage count")

    bandwidth_gbps = (
        DEFAULT_DP_ALLREDUCE_BANDWIDTH_GBPS
        if bandwidth_gbps is None
        else float(bandwidth_gbps)
    )
    latency_s = (
        DEFAULT_DP_ALLREDUCE_LATENCY_US / 1e6 if latency_s is None else float(latency_s)
    )

    runtime_stage_param_bytes = [
        stats.runtime_total_param_bytes for stats in stage_param_stats
    ]
    full_stage_param_bytes = estimate_full_stage_param_bytes(
        stage_stats=stage_param_stats,
        full_stage_layer_counts=full_stage_layer_counts,
    )
    validate_full_stage_param_fit(
        stage_stats=stage_param_stats,
        full_stage_param_bytes=full_stage_param_bytes,
    )
    runtime_dp_stage_times_s = [
        estimate_dp_allreduce_time_s(
            payload_bytes=payload_bytes,
            dp_world_size=dp_world_size,
            bandwidth_gbps=bandwidth_gbps,
            latency_s=latency_s,
        )
        for payload_bytes in runtime_stage_param_bytes
    ]
    runtime_dp_allreduce_time_s = max(runtime_dp_stage_times_s, default=0.0)

    observed_pp_compute_time_s = float(observed_iteration_time_s)
    if include_runtime_dp_in_observed_time:
        observed_pp_compute_time_s = max(
            0.0, observed_pp_compute_time_s - runtime_dp_allreduce_time_s
        )

    resolved_stage_timing_stats, stage_timing_source = _resolve_stage_timing_stats(
        stage_timing_stats=stage_timing_stats,
        observed_pp_compute_time_s=observed_pp_compute_time_s,
        num_microbatches=num_microbatches,
    )

    runtime_stage_forward_times_s = [
        stats.runtime_forward_time_s for stats in resolved_stage_timing_stats
    ]
    runtime_stage_backward_times_s = [
        stats.runtime_backward_time_s for stats in resolved_stage_timing_stats
    ]
    stage_layer_scales: list[float] = []
    full_stage_forward_times_s: list[float] = []
    full_stage_backward_times_s: list[float] = []
    for stage_idx, full_layer_count in enumerate(full_stage_layer_counts):
        runtime_layer_count = max(1, runtime_stage_layer_counts[stage_idx])
        layer_scale = float(full_layer_count) / float(runtime_layer_count)
        stage_layer_scales.append(layer_scale)
        full_stage_forward_times_s.append(
            runtime_stage_forward_times_s[stage_idx] * layer_scale
        )
        full_stage_backward_times_s.append(
            runtime_stage_backward_times_s[stage_idx] * layer_scale
        )

    runtime_pipeline = simulate_train_pipeline_1f1b(
        stage_forward_times_s=runtime_stage_forward_times_s,
        stage_backward_times_s=runtime_stage_backward_times_s,
        num_microbatches=num_microbatches,
    )
    runtime_pp_schedule_time_s = runtime_pipeline["total_time_s"]
    full_pipeline = simulate_train_pipeline_1f1b(
        stage_forward_times_s=full_stage_forward_times_s,
        stage_backward_times_s=full_stage_backward_times_s,
        num_microbatches=num_microbatches,
    )
    simulated_pp_compute_time_s = full_pipeline["total_time_s"]

    dp_stage_allreduce_times_s = [
        estimate_dp_allreduce_time_s(
            payload_bytes=payload_bytes,
            dp_world_size=dp_world_size,
            bandwidth_gbps=bandwidth_gbps,
            latency_s=latency_s,
        )
        for payload_bytes in full_stage_param_bytes
    ]
    simulated_dp_allreduce_time_s = max(dp_stage_allreduce_times_s, default=0.0)
    simulated_time_s = simulated_pp_compute_time_s + simulated_dp_allreduce_time_s

    return {
        "simulated_time_s": simulated_time_s,
        "simulated_pp_compute_time_s": simulated_pp_compute_time_s,
        "simulated_dp_allreduce_time_s": simulated_dp_allreduce_time_s,
        "simulated_pp_step_count": full_pipeline["num_scheduled_tasks"],
        "simulated_pp_step_time_s_max": full_pipeline["max_stage_time_s"],
        "simulated_pp_step_time_s_avg": full_pipeline["avg_stage_time_s"],
        "observed_pp_compute_time_s": observed_pp_compute_time_s,
        "observed_runtime_dp_allreduce_time_s": runtime_dp_allreduce_time_s,
        "runtime_stage_layer_counts": list(runtime_stage_layer_counts),
        "full_stage_layer_counts": list(full_stage_layer_counts),
        "stage_layer_scales": stage_layer_scales,
        "runtime_stage_forward_times_s": runtime_stage_forward_times_s,
        "runtime_stage_backward_times_s": runtime_stage_backward_times_s,
        "runtime_stage_timing_source": stage_timing_source,
        "full_stage_forward_times_s": full_stage_forward_times_s,
        "full_stage_backward_times_s": full_stage_backward_times_s,
        "runtime_pp_schedule_time_s": runtime_pp_schedule_time_s,
        "runtime_pp_schedule_scale": 1.0,
        "runtime_stage_param_bytes": runtime_stage_param_bytes,
        "full_stage_param_bytes": full_stage_param_bytes,
        "runtime_dp_stage_allreduce_times_s": runtime_dp_stage_times_s,
        "dp_stage_allreduce_times_s": dp_stage_allreduce_times_s,
        "dp_world_size": dp_world_size,
        "dp_allreduce_bandwidth_gbps": bandwidth_gbps,
        "dp_allreduce_latency_s": latency_s,
        "stage_param_stats": [asdict(stats) for stats in stage_param_stats],
        "stage_timing_stats": [asdict(stats) for stats in resolved_stage_timing_stats],
    }
