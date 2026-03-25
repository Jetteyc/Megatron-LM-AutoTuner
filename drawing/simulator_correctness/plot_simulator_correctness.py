#!/usr/bin/env python3
"""Draw simulator-vs-real correctness figures from JSON data."""

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


MODEL_COLORS = {
    "Qwen/Qwen3-0.6B": "#4C78A8",
    "Qwen/Qwen3-1.7B": "#F58518",
}


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path("outputs")
        / "drawing"
        / "simulator_correctness"
        / timestamp
        / "simulator_vs_real.png"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot simulator-vs-real throughput and MFU correctness."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="Input JSON path. Default: drawing/simulator_correctness/data.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output image path. Default: outputs/drawing/simulator_correctness/<timestamp>/simulator_vs_real.png",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--font-size", type=float, default=15.0, help="Base font size.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty object in: {path}")

    records: list[dict[str, object]] = []
    for model_name, model_payload in payload.items():
        if not isinstance(model_payload, dict):
            raise ValueError(f"Expected object for model: {model_name}")
        configs = model_payload.get("config")
        if not isinstance(configs, list) or not configs:
            raise ValueError(f"Expected non-empty config list for model: {model_name}")

        for config_idx, config_payload in enumerate(configs, start=1):
            if not isinstance(config_payload, dict):
                raise ValueError(f"Expected object for {model_name} config {config_idx}")
            simulator = config_payload.get("simulator")
            real = config_payload.get("real")
            if not isinstance(simulator, dict) or not isinstance(real, dict):
                raise ValueError(
                    f"Expected simulator/real objects for {model_name} config {config_idx}"
                )

            sim_throughput = simulator.get("throughput")
            sim_mfu = simulator.get("MFU")
            real_throughput = real.get("throughput")
            real_mfu = real.get("MFU")

            numeric_values = (
                sim_throughput,
                sim_mfu,
                real_throughput,
                real_mfu,
            )
            if not all(isinstance(value, (int, float)) for value in numeric_values):
                raise ValueError(
                    f"Expected numeric throughput/MFU for {model_name} config {config_idx}"
                )

            records.append(
                {
                    "model": model_name,
                    "label": f"{model_name.split('/')[-1]} 配置{config_idx}",
                    "sim_throughput": float(sim_throughput),
                    "real_throughput": float(real_throughput),
                    "sim_mfu": float(sim_mfu),
                    "real_mfu": float(real_mfu),
                }
            )

    return records


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


def _plot_metric(
    ax: plt.Axes,
    records: list[dict[str, object]],
    sim_key: str,
    real_key: str,
    title: str,
    axis_label: str,
) -> None:
    values = [float(record[sim_key]) for record in records] + [
        float(record[real_key]) for record in records
    ]
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-6)
    pad = span * 0.12
    lower = min_v - pad
    upper = max_v + pad

    seen_models: set[str] = set()
    for record in records:
        model = str(record["model"])
        label = model.split("/")[-1]
        legend_label = label if model not in seen_models else None
        seen_models.add(model)

        ax.scatter(
            float(record[sim_key]),
            float(record[real_key]),
            s=82,
            color=MODEL_COLORS.get(model, "#4C78A8"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            label=legend_label,
            zorder=3,
        )
        ax.annotate(
            str(record["label"]),
            (float(record[sim_key]), float(record[real_key])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=max(11.0, plt.rcParams["font.size"] - 1.0),
            color="#334155",
        )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.2,
        color="#64748B",
        label="理想线",
        zorder=2,
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"模拟值{axis_label}")
    ax.set_ylabel(f"真实值{axis_label}")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def plot(records: list[dict[str, object]], output_path: Path, dpi: int, font_size: float) -> None:
    _setup_fonts(font_size)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4))
    _plot_metric(
        axes[0],
        records,
        sim_key="sim_throughput",
        real_key="real_throughput",
        title="吞吐模拟值与真实值对比",
        axis_label="吞吐",
    )
    _plot_metric(
        axes[1],
        records,
        sim_key="sim_mfu",
        real_key="real_mfu",
        title="MFU 模拟值与真实值对比",
        axis_label="MFU",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(2, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.962),
    )
    fig.suptitle("模拟器精度对比", fontsize=font_size + 6.0, y=0.992)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    plot(records, args.output, args.dpi, args.font_size)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
