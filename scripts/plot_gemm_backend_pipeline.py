#!/usr/bin/env python3
"""Draw a compact Wave GEMM backend pipeline diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE = "#4C78A8"
ORANGE = "#F58518"
INK = "#111827"
MUTED = "#4B5563"
GRID = "#CBD5E1"
BG = "#FFFFFF"
SHARED = "#0F766E"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf" if bold else "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    fonts: list[ImageFont.ImageFont],
    fills: list[str],
) -> None:
    x0, y0, x1, y1 = box
    heights = [text_size(draw, line, font)[1] for line, font in zip(lines, fonts)]
    total_h = sum(heights) + 8 * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, font, fill, h in zip(lines, fonts, fills, heights):
        w, _ = text_size(draw, line, font)
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=font, fill=fill)
        y += h + 8


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    color: str,
    font: dict[str, ImageFont.ImageFont],
    fill: str = "#FFFFFF",
) -> None:
    x0, y0, x1, y1 = box
    shadow = (x0 + 3, y0 + 4, x1 + 3, y1 + 4)
    draw.rounded_rectangle(shadow, radius=10, fill="#E5E7EB")
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=color, width=3)
    draw.rectangle((x0, y0, x0 + 12, y1), fill=color)
    draw_centered_text(
        draw,
        (x0 + 18, y0 + 8, x1 - 12, y1 - 8),
        [title, subtitle],
        [font["box"], font["small"]],
        [INK, MUTED],
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    size = 15
    back = 18
    points = [
        (ex, ey),
        (ex - ux * back + px * size / 2, ey - uy * back + py * size / 2),
        (ex - ux * back - px * size / 2, ey - uy * back - py * size / 2),
    ]
    draw.polygon(points, fill=color)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    pad_x = 10
    pad_y = 5
    w, h = text_size(draw, text, font)
    draw.rounded_rectangle(
        (x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y),
        radius=8,
        fill="#FFFFFF",
        outline=color,
        width=2,
    )
    draw.text((x, y), text, fill=color, font=font)


def render(output: Path) -> None:
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    font = {
        "h1": load_font(36, bold=True),
        "sub": load_font(20),
        "box": load_font(23, bold=True),
        "small": load_font(17),
        "label": load_font(18, bold=True),
    }

    draw.text((70, 46), "Wave GEMM Backend Pipeline", font=font["h1"], fill=INK)
    draw.text(
        (70, 95),
        "Same Wave DSL kernel, two codegen paths, two assembly outputs",
        font=font["sub"],
        fill=MUTED,
    )

    source = (520, 155, 980, 255)
    mlir = (520, 345, 980, 445)
    llvm_codegen = (155, 560, 515, 660)
    llvm_asm = (155, 730, 515, 830)
    waveasm_codegen = (985, 560, 1345, 660)
    waveasm_asm = (985, 730, 1345, 830)

    rounded_box(draw, source, "Wave DSL Kernel", "dsl_source.py / wave_kernel", SHARED, font, "#F8FAFC")
    rounded_box(draw, mlir, "Shared Wave MLIR", "wave_module.mlir", SHARED, font, "#F8FAFC")
    rounded_box(draw, llvm_codegen, "LLVM Backend", "Wave MLIR -> LLVM IR", BLUE, font, "#F8FAFC")
    rounded_box(draw, llvm_asm, "LLVM Assembly", "simple_gemm_llvm.rocmasm", BLUE, font, "#FFFFFF")
    rounded_box(draw, waveasm_codegen, "WaveASM Backend", "direct WaveASM lowering", ORANGE, font, "#F8FAFC")
    rounded_box(draw, waveasm_asm, "WaveASM Assembly", "simple_gemm_waveasm.rocmasm", ORANGE, font, "#FFFFFF")

    arrow(draw, (750, 255), (750, 345), SHARED)
    label(draw, (783, 289), "compile", SHARED, font["label"])

    arrow(draw, (600, 445), (335, 560), BLUE)
    label(draw, (364, 491), "backend = llvm", BLUE, font["label"])
    arrow(draw, (335, 660), (335, 730), BLUE)
    label(draw, (362, 690), ".ll -> rocmasm", BLUE, font["label"])

    arrow(draw, (900, 445), (1165, 560), ORANGE)
    label(draw, (1008, 491), "backend = waveasm", ORANGE, font["label"])
    arrow(draw, (1165, 660), (1165, 730), ORANGE)
    label(draw, (1192, 690), "emit rocmasm", ORANGE, font["label"])

    draw.line((750, 470, 750, 838), fill=GRID, width=2)
    draw.text((270, 515), "LLVM path", font=font["label"], fill=BLUE)
    draw.text((1090, 515), "WaveASM path", font=font["label"], fill=ORANGE)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemm_backend_pipeline.png"))
    args = parser.parse_args()
    render(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
