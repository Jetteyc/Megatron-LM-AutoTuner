#!/usr/bin/env python3
"""Draw search/profile timing comparison from local data.json."""

from __future__ import annotations

import argparse
import json
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


SYSTEM_ORDER = ("aceso", "our")
SYSTEM_LABELS = {"aceso": "Aceso", "our": "Ours"}
SEGMENT_ORDER = ("search", "profile")
SEGMENT_LABELS = {"search": "Search", "profile": "Profile"}
SEGMENT_COLORS = {"search": "#E9C46A", "profile": "#2A9D8F"}
SYSTEM_ALPHA = {"aceso": 0.62, "our": 0.95}
SYSTEM_HATCH = {"aceso": "//", "our": None}


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path("outputs")
        / "drawing"
        / "algo_timing"
        / timestamp
        / "algo_timing.png"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot search and profile time for Aceso vs Ours."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="Input JSON path. Default: drawing/algo_timing/data.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output image path. Default: outputs/drawing/algo_timing/<timestamp>/algo_timing.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="搜索与性能分析耗时",
        help="Figure title.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=18.0,
        help="Base font size.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return parser.parse_args()


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def load_data(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty object in: {path}")

    models: list[str] = []
    series = {
        system: {segment: [] for segment in SEGMENT_ORDER} for system in SYSTEM_ORDER
    }

    for model_name, systems in payload.items():
        if not isinstance(model_name, str) or not isinstance(systems, dict):
            raise ValueError(f"Invalid record in {path}: {model_name!r}")

        models.append(short_model_name(model_name))
        for system in SYSTEM_ORDER:
            timings = systems.get(system)
            if not isinstance(timings, dict):
                raise ValueError(f"Missing '{system}' timings for model={model_name}")

            for segment in SEGMENT_ORDER:
                value = timings.get(segment)
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Expected numeric {system}.{segment} for model={model_name}"
                    )
                series[system][segment].append(float(value))

    return models, series


def plot(
    models: list[str],
    series: dict[str, dict[str, list[float]]],
    output_path: Path,
    title: str,
    font_size: float,
    dpi: int,
) -> None:
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

    x = list(range(len(models)))
    width = 0.28
    offsets = {"aceso": -width / 1.6, "our": width / 1.6}

    fig, ax = plt.subplots(figsize=(10.4, 6.6))

    totals = {
        system: [
            series[system]["search"][idx] + series[system]["profile"][idx]
            for idx in range(len(models))
        ]
        for system in SYSTEM_ORDER
    }
    max_total = max(value for values in totals.values() for value in values)
    ax.set_ylim(0.0, max_total * 1.22)

    legend_handles = []
    legend_labels = []

    for system in SYSTEM_ORDER:
        positions = [idx + offsets[system] for idx in x]
        bottoms = [0.0] * len(models)

        for segment in SEGMENT_ORDER:
            heights = series[system][segment]
            bars = ax.bar(
                positions,
                heights,
                width=width,
                bottom=bottoms,
                color=SEGMENT_COLORS[segment],
                alpha=SYSTEM_ALPHA[system],
                hatch=SYSTEM_HATCH[system],
                edgecolor="#4A4A4A" if system == "aceso" else "none",
                linewidth=0.6 if system == "aceso" else 0.0,
            )

            if system == SYSTEM_ORDER[0]:
                legend_handles.append(bars[0])
                legend_labels.append(SEGMENT_LABELS[segment])

            for bar, height, bottom in zip(bars, heights, bottoms):
                if height < max_total * 0.04:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + height / 2,
                    f"{height:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=max(font_size - 2.5, 9.0),
                    color="#1F1F1F",
                )

            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

        for xpos, total in zip(positions, totals[system]):
            ax.text(
                xpos,
                total + max_total * 0.02,
                f"{total:.1f}s",
                ha="center",
                va="bottom",
                fontsize=max(font_size - 0.5, 10.0),
                fontweight="bold",
                color="#1F1F1F",
            )

    for idx, model in enumerate(models):
        aceso_total = totals["aceso"][idx]
        our_total = totals["our"][idx]
        speedup = aceso_total / our_total if our_total > 0 else 0.0
        ax.text(
            idx,
            max(aceso_total, our_total) + max_total * 0.09,
            f"{speedup:.1f}x faster",
            ha="center",
            va="bottom",
            fontsize=max(font_size - 0.8, 10.0),
            fontweight="bold",
            color="#264653",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=font_size)
    ax.set_ylabel("Time (s)", fontsize=font_size + 1.0)
    ax.set_title(title, fontsize=font_size + 2.0, pad=10.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    system_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#FFFFFF",
            edgecolor="#4A4A4A" if system == "aceso" else "#2A9D8F",
            hatch=SYSTEM_HATCH[system],
            linewidth=0.9,
            alpha=SYSTEM_ALPHA[system],
        )
        for system in SYSTEM_ORDER
    ]
    system_labels = [SYSTEM_LABELS[system] for system in SYSTEM_ORDER]

    inline_label_size = max(font_size - 1.0, 10.0)
    inline_legend_size = max(font_size - 1.6, 9.5)

    fig.text(
        0.19,
        0.965,
        "Segment",
        ha="right",
        va="center",
        fontsize=inline_label_size,
        fontweight="bold",
        color="#1F1F1F",
    )
    legend_segment = fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.20, 0.965),
        ncol=len(legend_labels),
        fontsize=inline_legend_size,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.add_artist(legend_segment)
    fig.text(
        0.62,
        0.965,
        "System",
        ha="right",
        va="center",
        fontsize=inline_label_size,
        fontweight="bold",
        color="#1F1F1F",
    )
    fig.legend(
        system_handles,
        system_labels,
        loc="center left",
        bbox_to_anchor=(0.63, 0.965),
        ncol=len(system_labels),
        fontsize=inline_legend_size,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models, series = load_data(args.input)
    plot(
        models=models,
        series=series,
        output_path=args.output,
        title=args.title,
        font_size=args.font_size,
        dpi=args.dpi,
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
