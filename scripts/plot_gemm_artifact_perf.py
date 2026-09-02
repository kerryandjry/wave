#!/usr/bin/env python3
"""Plot GEMM backend performance summaries from artifact directories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKENDS = ("llvm", "waveasm")
LIBRARY_BACKENDS = ("waveasm", "rocblas", "hipblaslt")
COLORS = {
    "llvm": "#4C78A8",
    "waveasm": "#F58518",
    "rocblas": "#54A24B",
    "hipblaslt": "#B279A2",
}


def load_font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def find_summaries(artifacts_dir: Path) -> list[Path]:
    return sorted(
        artifacts_dir.glob("gemm_m*_n*_k*_*/compare_rocmasm_perf/summary.json")
    )


def shape_label(summary: dict) -> str:
    shape = summary["shape"]
    m = shape["M"]
    n = shape["N"]
    k = shape["K"]
    if m == n == k:
        return f"{m}"
    return f"{m}x{n}x{k}"


def shape_sort_key(summary_path: Path) -> int:
    match = re.search(r"gemm_m(\d+)_n\d+_k\d+", str(summary_path))
    return int(match.group(1)) if match else 0


def load_library_results(path: Path) -> dict[tuple[str, int, int, int], dict]:
    results = {}
    for line in path.read_text().splitlines():
        if line.strip():
            result = json.loads(line)
            key = (result["backend"], result["M"], result["N"], result["K"])
            results[key] = result
    return results


def collect_rows(
    artifacts_dir: Path, library_results_path: Path | None = None
) -> tuple[list[dict], tuple[str, ...]]:
    rows = []
    library_results = (
        load_library_results(library_results_path) if library_results_path else None
    )
    backends = LIBRARY_BACKENDS if library_results is not None else BACKENDS
    for summary_path in sorted(find_summaries(artifacts_dir), key=shape_sort_key):
        summary = json.loads(summary_path.read_text())
        shape = summary["shape"]
        row = {"label": shape_label(summary), "summary": summary_path}
        waveasm_profile = summary["backends"]["waveasm"]["profile"]
        row["waveasm_tflops"] = float(waveasm_profile["tflops"])
        row["waveasm_time_ms"] = float(waveasm_profile["time_ms"])
        if library_results is None:
            llvm_profile = summary["backends"]["llvm"]["profile"]
            row["llvm_tflops"] = float(llvm_profile["tflops"])
            row["llvm_time_ms"] = float(llvm_profile["time_ms"])
        else:
            for backend in ("rocblas", "hipblaslt"):
                key = (backend, shape["M"], shape["N"], shape["K"])
                result = library_results[key]
                row[f"{backend}_tflops"] = float(result["tflops"])
                row[f"{backend}_time_ms"] = float(result["time_ms"])
        rows.append(row)
    return rows, backends


def rounded(value: float, precision: int = 2) -> str:
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.{precision}f}"


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[dict],
    metric_key: str,
    title: str,
    y_label: str,
    font: dict[str, ImageFont.ImageFont],
    backends: tuple[str, ...],
) -> None:
    left, top, right, bottom = box
    axis_left = left + 82
    axis_top = top + 64
    axis_right = right - 34
    axis_bottom = bottom - 76
    chart_w = axis_right - axis_left
    chart_h = axis_bottom - axis_top
    max_value = max(
        row[f"{backend}_{metric_key}"] for row in rows for backend in backends
    )
    y_max = max_value * 1.18

    draw.text((left, top), title, fill="#1F2937", font=font["title"])
    draw.text((left, top + 34), y_label, fill="#4B5563", font=font["small"])

    for tick in range(6):
        value = y_max * tick / 5
        y = axis_bottom - chart_h * tick / 5
        draw.line((axis_left, y, axis_right, y), fill="#E5E7EB", width=1)
        label = rounded(value, 1)
        tw, th = text_size(draw, label, font["small"])
        draw.text(
            (axis_left - tw - 10, y - th / 2), label, fill="#6B7280", font=font["small"]
        )

    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill="#9CA3AF", width=2)
    draw.line(
        (axis_left, axis_bottom, axis_right, axis_bottom), fill="#9CA3AF", width=2
    )

    group_w = chart_w / len(rows)
    bar_w = min(50, group_w * 0.24)
    gap = bar_w * 0.22
    for idx, row in enumerate(rows):
        center = axis_left + group_w * (idx + 0.5)
        total_bar_w = len(backends) * bar_w + (len(backends) - 1) * gap
        group_left = center - total_bar_w / 2
        for backend_index, backend in enumerate(backends):
            value = row[f"{backend}_{metric_key}"]
            bar_h = chart_h * value / y_max
            x0 = group_left + backend_index * (bar_w + gap)
            x1 = x0 + bar_w
            y0 = axis_bottom - bar_h
            draw.rounded_rectangle(
                (x0, y0, x1, axis_bottom), radius=4, fill=COLORS[backend]
            )
            label = rounded(value)
            tw, th = text_size(draw, label, font["small"])
            draw.text(
                (x0 + (bar_w - tw) / 2, y0 - th - 5),
                label,
                fill="#111827",
                font=font["small"],
            )
        tw, th = text_size(draw, row["label"], font["label"])
        draw.text(
            (center - tw / 2, axis_bottom + 16),
            row["label"],
            fill="#374151",
            font=font["label"],
        )


def draw_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    backends: tuple[str, ...],
) -> None:
    cursor = x
    for backend in backends:
        draw.rounded_rectangle(
            (cursor, y + 4, cursor + 22, y + 18), radius=3, fill=COLORS[backend]
        )
        draw.text((cursor + 30, y), backend, fill="#374151", font=font)
        tw, _ = text_size(draw, backend, font)
        cursor += 30 + tw + 34


def render(rows: list[dict], backends: tuple[str, ...], output: Path) -> None:
    width, height = 1440, 820
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font = {
        "h1": load_font(34, bold=True),
        "title": load_font(25, bold=True),
        "label": load_font(18),
        "small": load_font(15),
    }

    draw.text((64, 38), "GEMM Backend Performance", fill="#111827", font=font["h1"])
    subtitle = (
        "WaveASM kernel trace and BLAS GPU event averages; f16 x f16 -> f32"
        if backends == LIBRARY_BACKENDS
        else "rocmasm kernel trace averages; f16 x f16 -> f32"
    )
    draw.text((64, 82), subtitle, fill="#4B5563", font=font["label"])
    legend_x = 880 if len(backends) == 3 else 1050
    draw_legend(draw, legend_x, 48, font["label"], backends)

    draw_panel(
        draw,
        (64, 150, 705, 760),
        rows,
        "tflops",
        "Throughput",
        "TFLOP/s, higher is better",
        font,
        backends,
    )
    draw_panel(
        draw,
        (770, 150, 1376, 760),
        rows,
        "time_ms",
        "Execution Time",
        "milliseconds, lower is better",
        font,
        backends,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gemm_backend_perf.png")
    )
    parser.add_argument(
        "--library-results",
        type=Path,
        help="JSON Lines output from benchmark_blas_gemm",
    )
    args = parser.parse_args()

    rows, backends = collect_rows(args.artifacts_dir, args.library_results)
    if not rows:
        raise SystemExit(
            f"No compare_rocmasm_perf/summary.json files found under {args.artifacts_dir}"
        )
    render(rows, backends, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
