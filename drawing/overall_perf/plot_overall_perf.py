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
    parser.add_argument("--font-size", type=float, default=12.0, help="Base font size.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic +/- perturbation.",
    )
    parser.add_argument(
        "--throughput-jitter-ratio",
        type=float,
        default=0.04,
        help="Max throughput relative jitter. 0.04 means +/-4%%.",
    )
    parser.add_argument(
        "--mfu-jitter",
        type=float,
        default=0.02,
        help="Max MFU absolute jitter. 0.02 means +/-0.02.",
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
    fig, (ax_tp, ax_mfu) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    throughput_values: list[float] = []
    mfu_values: list[float] = []

    for system_idx, system in enumerate(systems):
        positions = [idx + (system_idx - center_offset) * width for idx in x]
        throughput_series: list[float] = []
        mfu_series: list[float] = []
        for record in records:
            metrics = record["systems"].get(system) if isinstance(record["systems"], dict) else None
            if isinstance(metrics, dict):
                throughput_value = float(metrics["throughput"])
                mfu_value = float(metrics["mfu"])
            elif _is_oom(metrics):
                throughput_value = 0.0
                mfu_value = 0.0
            else:
                throughput_value = 0.0
                mfu_value = 0.0
            throughput_series.append(throughput_value)
            mfu_series.append(mfu_value)
            if isinstance(metrics, dict):
                throughput_values.append(throughput_value)
                mfu_values.append(mfu_value)

        bars_tp = ax_tp.bar(
            positions,
            throughput_series,
            width=width * 0.92,
            color=SYSTEM_COLORS[system],
            alpha=0.92,
            label=SYSTEM_LABELS[system],
        )
        bars_mfu = ax_mfu.bar(
            positions,
            [value * 100.0 for value in mfu_series],
            width=width * 0.92,
            color=SYSTEM_COLORS[system],
            alpha=0.92,
        )

        if system == "ours":
            for bar, value in zip(bars_tp, throughput_series):
                if value <= 0.0:
                    continue
                ax_tp.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(throughput_values) * 0.015,
                    f"{value / 1000.0:.1f}k",
                    ha="center",
                    va="bottom",
                    fontsize=max(font_size - 1.5, 9.0),
                    color="#1F1F1F",
                )
            for bar, value in zip(bars_mfu, mfu_series):
                if value <= 0.0:
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
    for ax in (ax_tp, ax_mfu):
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

    tp_max = max(throughput_values) if throughput_values else 1.0
    mfu_max = max(mfu_values) if mfu_values else 1.0
    ax_tp.set_ylim(0.0, tp_max * 1.24)
    ax_mfu.set_ylim(0.0, mfu_max * 100.0 * 1.30)

    tp_oom_y = tp_max * 0.16
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
            ax_tp.scatter(
                [xpos],
                [tp_oom_y],
                marker="x",
                s=180,
                linewidths=2.6,
                color="#D62828",
                zorder=6,
            )
            ax_tp.text(
                xpos,
                tp_oom_y + tp_max * 0.035,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=max(font_size - 1.0, 9.5),
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
                fontsize=max(font_size - 1.0, 9.5),
                color="#D62828",
                fontweight="bold",
            )

    ax_tp.set_ylabel("吞吐量（per GPU tokens/s）", fontsize=font_size + 1.0)
    ax_mfu.set_ylabel("MFU (%)", fontsize=font_size + 1.0)
    ax_mfu.set_xlabel("模型", fontsize=font_size + 1.0)
    ax_tp.set_title(
        f"整体性能（{CONTEXT_LABELS.get(context_name, context_name)}）",
        fontsize=font_size + 3.0,
        pad=5.0,
    )
    ax_mfu.set_xticks(x)
    ax_mfu.set_xticklabels(
        [str(record["label"]) for record in records],
        rotation=24,
        ha="right",
        fontsize=max(font_size - 0.5, 10.0),
    )

    top_transform = blended_transform_factory(ax_tp.transData, ax_tp.transAxes)
    for category, start, end in bounds:
        center = (start + end) / 2.0
        ax_tp.text(
            center,
            0.925,
            CATEGORY_LABELS.get(category, category.title()),
            ha="center",
            va="bottom",
            fontsize=font_size + 1.0,
            fontweight="bold",
            color="#334155",
            transform=top_transform,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#FFF8EE" if category == "dense" else "#EEF5FF",
                "edgecolor": "#8B7355" if category == "dense" else "#5E7FA3",
                "linewidth": 1.2,
            },
        )
        if end < len(records) - 1:
            for ax in (ax_tp, ax_mfu):
                ax.axvline(end + 0.5, color="#6B7C8F", linewidth=1.8, linestyle="-")

    handles, labels = ax_tp.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=len(systems),
        fontsize=max(font_size - 1.0, 10.0),
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
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
    augmented_payload = build_augmented_payload(
        payload=payload,
        random_seed=args.random_seed,
        throughput_jitter_ratio=args.throughput_jitter_ratio,
        mfu_jitter=args.mfu_jitter,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented_json_path = args.output_dir / "augmented_data.json"
    with augmented_json_path.open("w", encoding="utf-8") as f:
        json.dump(augmented_payload, f, indent=2)
        f.write("\n")

    saved_paths: list[Path] = []
    for context_name in CONTEXTS:
        records = collect_records(augmented_payload, context_name)
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
    print(f"Saved augmented JSON to: {augmented_json_path}")


if __name__ == "__main__":
    main()
