#!/usr/bin/env python3
"""Draw overall performance figures for short and long context workloads."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to draw figures. Install it with: pip install matplotlib"
    ) from exc


SYSTEM_ORDER = ("aceso", "megatron", "ours")
SYSTEM_LABELS = {
    "aceso": "Aceso",
    "megatron": "Megatron",
    "ours": "Ours",
}
SYSTEM_COLORS = {
    "aceso": "#9AA6B2",
    "megatron": "#4C78A8",
    "ours": "#2A9D8F",
}
CATEGORY_ORDER = ("dense", "moe")
CATEGORY_LABELS = {"dense": "Dense", "moe": "MoE"}
CONTEXTS = ("short_context", "long_context")
CONTEXT_LABELS = {
    "short_context": "短文本",
    "long_context": "长文本",
}
THROUGHPUT_LOW_RANGE = (0.0, 50000.0)
THROUGHPUT_HIGH_RANGE = (80000.0, 110000.0)
OOM_FONT_SIZE = 13.0


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / "drawing" / "overall_perf" / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw short-context and long-context overall performance figures."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="Input JSON path. Default: drawing/overall_perf/data.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory used to save both figures and the augmented JSON copy.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--font-size", type=float, default=15.5, help="Base font size.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic +/- perturbation.",
    )
    parser.add_argument(
        "--throughput-jitter-ratio",
        type=float,
        default=0.0,
        help="Max throughput relative jitter. Default 0.0 keeps plotted throughput identical to the input JSON.",
    )
    parser.add_argument(
        "--mfu-jitter",
        type=float,
        default=0.0,
        help="Max MFU absolute jitter. Default 0.0 keeps plotted MFU identical to the input JSON.",
    )
    return parser.parse_args()


def load_data(path: Path) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty object in: {path}")
    return payload


def _is_oom(metrics: object) -> bool:
    return isinstance(metrics, str) and metrics.strip().upper() == "OOM"


def build_augmented_payload(
    payload: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    random_seed: int,
    throughput_jitter_ratio: float,
    mfu_jitter: float,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    rng = random.Random(random_seed)
    augmented: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}

    for category, models in payload.items():
        category_payload: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for model_name, contexts in models.items():
            context_payload: dict[str, dict[str, dict[str, float]]] = {}
            for context_name, systems in contexts.items():
                if context_name not in CONTEXTS:
                    continue
                system_payload: dict[str, dict[str, float]] = {}
                for system_name, metrics in systems.items():
                    if _is_oom(metrics):
                        system_payload[system_name] = "OOM"
                        continue
                    if not isinstance(metrics, dict):
                        raise ValueError(
                            f"Expected object or 'OOM' at {category}/{model_name}/{context_name}/{system_name}"
                        )
                    throughput = metrics.get("throughput")
                    mfu = metrics.get("mfu")
                    if not isinstance(throughput, (int, float)) or not isinstance(
                        mfu, (int, float)
                    ):
                        raise ValueError(
                            f"Each system needs numeric throughput and mfu: {category}/{model_name}/{context_name}/{system_name}"
                        )

                    throughput_scale = 1.0 + rng.uniform(
                        -throughput_jitter_ratio, throughput_jitter_ratio
                    )
                    mfu_delta = rng.uniform(-mfu_jitter, mfu_jitter)
                    adjusted_throughput = max(0.0, float(throughput) * throughput_scale)
                    if float(throughput) >= THROUGHPUT_HIGH_RANGE[0]:
                        adjusted_throughput = min(
                            THROUGHPUT_HIGH_RANGE[1],
                            max(THROUGHPUT_HIGH_RANGE[0], adjusted_throughput),
                        )
                    adjusted_mfu = min(0.99, max(0.0, float(mfu) + mfu_delta))
                    system_payload[system_name] = {
                        "throughput": round(adjusted_throughput, 3),
                        "mfu": round(adjusted_mfu, 4),
                    }
                context_payload[context_name] = system_payload
            category_payload[model_name] = context_payload
        augmented[category] = category_payload

    return augmented


def short_model_label(model_name: str) -> str:
    return model_name.split("/")[-1]


def collect_records(
    payload: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    context_name: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        models = payload.get(category, {})
        if not isinstance(models, dict):
            continue
        for model_name, contexts in models.items():
            if not isinstance(contexts, dict):
                continue
            systems = contexts.get(context_name)
            if not isinstance(systems, dict) or not systems:
                continue
            records.append(
                {
                    "category": category,
                    "model_name": model_name,
                    "label": short_model_label(model_name),
                    "systems": systems,
                }
            )
    if not records:
        raise ValueError(f"No records found for context: {context_name}")
    return records


def available_systems(records: list[dict[str, object]]) -> list[str]:
    present = set()
    for record in records:
        systems = record["systems"]
        if isinstance(systems, dict):
            present.update(systems.keys())
    return [system for system in SYSTEM_ORDER if system in present]


def _group_bounds(records: list[dict[str, object]]) -> list[tuple[str, int, int]]:
    bounds: list[tuple[str, int, int]] = []
    start = 0
    while start < len(records):
        category = str(records[start]["category"])
        end = start
        while end + 1 < len(records) and str(records[end + 1]["category"]) == category:
            end += 1
        bounds.append((category, start, end))
        start = end + 1
    return bounds


def _setup_fonts(font_size: float) -> None:
    from matplotlib import font_manager

    font_family = "DejaVu Sans"
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_family = font_manager.FontProperties(fname=font_path).get_name()
            break

    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": font_family,
            "font.sans-serif": [font_family],
            "axes.unicode_minus": False,
        }
    )


def _should_use_broken_throughput_axis(throughput_values: list[float]) -> bool:
    if not throughput_values:
        return False

    low_min, low_max = THROUGHPUT_LOW_RANGE
    high_min, high_max = THROUGHPUT_HIGH_RANGE
    has_low_band = any(low_min <= value <= low_max for value in throughput_values)
    has_high_band = any(value >= high_min for value in throughput_values)
    return has_low_band and has_high_band


def _add_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    # Draw break marks in point units so they stay visually parallel regardless
    # of axis aspect ratio or panel height.
    marker = [(-1, -1), (1, 1)]
    common = dict(
        marker=marker,
        markersize=9,
        linestyle="none",
        color="#334155",
        markeredgewidth=1.4,
        clip_on=False,
    )

    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **common)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **common)


def _throughput_label_y(value: float, use_broken_axis: bool) -> tuple[str, float]:
    if use_broken_axis and value >= THROUGHPUT_HIGH_RANGE[0]:
        return "high", min(value, THROUGHPUT_HIGH_RANGE[1])
    return "low", min(value, THROUGHPUT_LOW_RANGE[1])


def _annotate_throughput_bar(
    axis: plt.Axes,
    bar,
    *,
    value: float,
    axis_limits: tuple[float, float],
    font_size: float,
) -> None:
    axis_min, axis_max = axis_limits
    visible_value = min(max(value, axis_min), axis_max)
    near_top = visible_value >= axis_max - (axis_max - axis_min) * 0.08
    axis.annotate(
        f"{value / 1000.0:.1f}k",
        xy=(bar.get_x() + bar.get_width() / 2.0, visible_value),
        xytext=(0, -5 if near_top else 4),
        textcoords="offset points",
        ha="center",
        va="top" if near_top else "bottom",
        fontsize=max(font_size - 1.5, 9.0),
        color="#1F1F1F",
        clip_on=False,
    )


def plot_context(
    records: list[dict[str, object]],
    context_name: str,
    output_path: Path,
    font_size: float,
    dpi: int,
) -> None:
    from matplotlib.transforms import blended_transform_factory

    _setup_fonts(font_size)

    systems = available_systems(records)
    if not systems:
        raise ValueError(f"No systems found for context: {context_name}")

    x = list(range(len(records)))
    width = min(0.24, 0.74 / max(1, len(systems)))
    center_offset = (len(systems) - 1) / 2.0
    fig_width = max(11.0, len(records) * 1.45)

    throughput_values: list[float] = []
    mfu_values: list[float] = []
    system_series_map: dict[str, tuple[list[float | None], list[float | None]]] = {}

    for system in systems:
        throughput_series: list[float | None] = []
        mfu_series: list[float | None] = []
        for record in records:
            metrics = record["systems"].get(system) if isinstance(record["systems"], dict) else None
            if isinstance(metrics, dict):
                throughput_value = float(metrics["throughput"])
                mfu_value = float(metrics["mfu"])
                throughput_values.append(throughput_value)
                mfu_values.append(mfu_value)
            else:
                throughput_value = None
                mfu_value = None
            throughput_series.append(throughput_value)
            mfu_series.append(mfu_value)
        system_series_map[system] = (throughput_series, mfu_series)

    use_broken_tp_axis = _should_use_broken_throughput_axis(throughput_values)
    tp_max = max(throughput_values) if throughput_values else 1.0
    mfu_max = max(mfu_values) if mfu_values else 1.0
    if use_broken_tp_axis:
        fig, (ax_tp_high, ax_tp_low, ax_mfu) = plt.subplots(
            3,
            1,
            figsize=(fig_width, 10.0),
            sharex=True,
            gridspec_kw={"height_ratios": [0.3, 0.6, 1.0], "hspace": 0.06},
        )
        tp_axes = {"high": ax_tp_high, "low": ax_tp_low}
    else:
        fig, (ax_tp_low, ax_mfu) = plt.subplots(
            2,
            1,
            figsize=(fig_width, 9.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1.0]},
        )
        ax_tp_high = None
        tp_axes = {"low": ax_tp_low}

    throughput_axis_ranges = (
        {"high": THROUGHPUT_HIGH_RANGE, "low": THROUGHPUT_LOW_RANGE}
        if use_broken_tp_axis
        else {"low": (0.0, tp_max * 1.24)}
    )

    for system_idx, system in enumerate(systems):
        positions = [idx + (system_idx - center_offset) * width for idx in x]
        throughput_series, mfu_series = system_series_map[system]
        plot_throughput_series = [
            value if value is not None else float("nan") for value in throughput_series
        ]
        plot_mfu_series = [
            value * 100.0 if value is not None else float("nan") for value in mfu_series
        ]

        tp_bar_sets: dict[str, list[object]] = {}
        for axis_name, axis in tp_axes.items():
            tp_bar_sets[axis_name] = axis.bar(
                positions,
                plot_throughput_series,
                width=width * 0.92,
                color=SYSTEM_COLORS[system],
                alpha=0.92,
                label=SYSTEM_LABELS[system] if axis_name == "low" else None,
            )
        bars_mfu = ax_mfu.bar(
            positions,
            plot_mfu_series,
            width=width * 0.92,
            color=SYSTEM_COLORS[system],
            alpha=0.92,
        )

        if system == "ours":
            for idx, value in enumerate(throughput_series):
                if value is None or value <= 0.0:
                    continue
                axis_name, _ = _throughput_label_y(value, use_broken_tp_axis)
                _annotate_throughput_bar(
                    tp_axes[axis_name],
                    tp_bar_sets[axis_name][idx],
                    value=value,
                    axis_limits=throughput_axis_ranges[axis_name],
                    font_size=font_size,
                )
            for bar, value in zip(bars_mfu, mfu_series):
                if value is None or value <= 0.0:
                    continue
                ax_mfu.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 100.0 + max(mfu_values) * 100.0 * 0.018,
                    f"{value * 100.0:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=max(font_size - 2.0, 8.5),
                    color="#1F1F1F",
                )

    bounds = _group_bounds(records)
    throughput_axes = [ax_tp_low] if ax_tp_high is None else [ax_tp_high, ax_tp_low]
    for ax in (*throughput_axes, ax_mfu):
        for group_idx, (category, start, end) in enumerate(bounds):
            ax.axvspan(
                start - 0.55,
                end + 0.55,
                color="#E8DDD0" if group_idx % 2 == 0 else "#DCE8F5",
                alpha=0.95,
                zorder=0,
            )
            if end < len(records) - 1:
                ax.axvspan(
                    end + 0.46,
                    end + 0.54,
                    color="#FFFFFF",
                    alpha=0.98,
                    zorder=1,
                )
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)

    if use_broken_tp_axis and ax_tp_high is not None:
        ax_tp_low.set_ylim(*THROUGHPUT_LOW_RANGE)
        ax_tp_high.set_ylim(*THROUGHPUT_HIGH_RANGE)
        ax_tp_high.spines["bottom"].set_visible(False)
        ax_tp_low.spines["top"].set_visible(False)
        ax_tp_high.tick_params(labelbottom=False, bottom=False)
        _add_break_marks(ax_tp_high, ax_tp_low)
    else:
        ax_tp_low.set_ylim(0.0, tp_max * 1.24)
    ax_mfu.set_ylim(0.0, mfu_max * 100.0 * 1.30)

    tp_oom_y = tp_max * 0.16 if not use_broken_tp_axis else THROUGHPUT_LOW_RANGE[1] * 0.16
    mfu_oom_y = mfu_max * 100.0 * 0.14
    for system_idx, system in enumerate(systems):
        positions = [idx + (system_idx - center_offset) * width for idx in x]
        for record_idx, record in enumerate(records):
            metrics = record["systems"].get(system) if isinstance(record["systems"], dict) else None
            if not _is_oom(metrics):
                continue
            xpos = positions[record_idx]
            if system == "aceso":
                xpos -= width * 0.24
            elif system == "ours":
                xpos += width * 0.24
            ax_tp_low.scatter(
                [xpos],
                [tp_oom_y],
                marker="x",
                s=180,
                linewidths=2.6,
                color="#D62828",
                zorder=6,
            )
            ax_tp_low.text(
                xpos,
                tp_oom_y + (tp_max * 0.035 if not use_broken_tp_axis else 1400.0),
                "OOM",
                ha="center",
                va="bottom",
                fontsize=OOM_FONT_SIZE,
                color="#D62828",
                fontweight="bold",
            )
            ax_mfu.scatter(
                [xpos],
                [mfu_oom_y],
                marker="x",
                s=180,
                linewidths=2.6,
                color="#D62828",
                zorder=6,
            )
            ax_mfu.text(
                xpos,
                mfu_oom_y + mfu_max * 100.0 * 0.04,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=OOM_FONT_SIZE,
                color="#D62828",
                fontweight="bold",
            )

    ax_tp_low.set_ylabel("吞吐量（per GPU tokens/s）", fontsize=font_size + 1.0)
    ax_tp_low.yaxis.set_label_coords(-0.085, 0.66)
    ax_mfu.set_ylabel("MFU (%)", fontsize=font_size + 1.0)
    # ax_mfu.set_xlabel("模型", fontsize=font_size + 1.0)
    (ax_tp_high or ax_tp_low).set_title(
        f"整体性能（{CONTEXT_LABELS.get(context_name, context_name)}）",
        fontsize=font_size + 3.0,
        pad=22.0,
    )
    ax_mfu.set_xticks(x)
    ax_mfu.set_xticklabels(
        [str(record["label"]) for record in records],
        rotation=24,
        ha="right",
        fontsize=max(font_size - 0.5, 10.0),
    )

    top_axis = ax_tp_high or ax_tp_low
    top_transform = blended_transform_factory(top_axis.transData, top_axis.transAxes)
    category_label_y = 0.84 if use_broken_tp_axis else 1.0
    category_label_va = "bottom" if use_broken_tp_axis else "center"
    for category, start, end in bounds:
        center = (start + end) / 2.0
        top_axis.text(
            center,
            category_label_y,
            CATEGORY_LABELS.get(category, category.title()),
            ha="center",
            va=category_label_va,
            fontsize=font_size + 1.0,
            fontweight="bold",
            color="#334155",
            transform=top_transform,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#FFF8EE" if category == "dense" else "#EEF5FF",
                "edgecolor": "#8B7355" if category == "dense" else "#5E7FA3",
                "linewidth": 1.2,
            },
        )
        if end < len(records) - 1:
            for ax in (*throughput_axes, ax_mfu):
                ax.axvline(end + 0.5, color="#6B7C8F", linewidth=1.8, linestyle="-")

    handles, labels = ax_tp_low.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=len(systems),
        fontsize=max(font_size - 1.0, 10.0),
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.12,
        left=0.11,
        right=0.98,
        hspace=0.08 if use_broken_tp_axis else 0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.throughput_jitter_ratio < 0:
        raise ValueError("--throughput-jitter-ratio must be >= 0.")
    if args.mfu_jitter < 0:
        raise ValueError("--mfu-jitter must be >= 0.")
    if args.font_size <= 0:
        raise ValueError("--font-size must be > 0.")

    payload = load_data(args.input)
    plot_payload = payload
    if args.throughput_jitter_ratio > 0 or args.mfu_jitter > 0:
        plot_payload = build_augmented_payload(
            payload=payload,
            random_seed=args.random_seed,
            throughput_jitter_ratio=args.throughput_jitter_ratio,
            mfu_jitter=args.mfu_jitter,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted_json_path = args.output_dir / "plotted_data.json"
    with plotted_json_path.open("w", encoding="utf-8") as f:
        json.dump(plot_payload, f, indent=2)
        f.write("\n")

    saved_paths: list[Path] = []
    for context_name in CONTEXTS:
        records = collect_records(plot_payload, context_name)
        output_path = args.output_dir / f"{context_name}.png"
        plot_context(
            records=records,
            context_name=context_name,
            output_path=output_path,
            font_size=args.font_size,
            dpi=args.dpi,
        )
        saved_paths.append(output_path)

    for path in saved_paths:
        print(f"Saved figure to: {path}")
    print(f"Saved plotted JSON to: {plotted_json_path}")


if __name__ == "__main__":
    main()
