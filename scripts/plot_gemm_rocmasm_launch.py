#!/usr/bin/env python3
"""Draw a simple diagram for launching LLVM and WaveASM rocmasm kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#0F766E"
INK = "#111827"
MUTED = "#4B5563"
LINE = "#CBD5E1"
BG = "#FFFFFF"


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


def centered_text(
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


def box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    color: str,
    font: dict[str, ImageFont.ImageFont],
    fill: str = "#F8FAFC",
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle((x0 + 3, y0 + 4, x1 + 3, y1 + 4), radius=10, fill="#E5E7EB")
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=color, width=3)
    draw.rectangle((x0, y0, x0 + 12, y1), fill=color)
    centered_text(
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
    draw.polygon(
        [
            (ex, ey),
            (ex - ux * back + px * size / 2, ey - uy * back + py * size / 2),
            (ex - ux * back - px * size / 2, ey - uy * back - py * size / 2),
        ],
        fill=color,
    )


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    w, h = text_size(draw, text, font)
    draw.rounded_rectangle((x - 10, y - 5, x + w + 10, y + h + 5), radius=8, fill="#FFFFFF", outline=color, width=2)
    draw.text((x, y), text, font=font, fill=color)


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
        "tiny": load_font(16),
    }

    draw.text((70, 46), "Launching the Two rocmasm Kernels", font=font["h1"], fill=INK)
    draw.text(
        (70, 95),
        "Both assemblies become HSACO, then go through the same HIP launch path",
        font=font["sub"],
        fill=MUTED,
    )

    llvm_asm = (115, 190, 485, 290)
    waveasm_asm = (1015, 190, 1385, 290)
    llvm_hsaco = (115, 420, 485, 520)
    waveasm_hsaco = (1015, 420, 1385, 520)
    launcher = (515, 585, 985, 705)
    gpu = (615, 760, 885, 850)

    box(draw, llvm_asm, "LLVM rocmasm", "simple_gemm_llvm.rocmasm", BLUE, font)
    box(draw, waveasm_asm, "WaveASM rocmasm", "simple_gemm_waveasm.rocmasm", ORANGE, font)
    box(draw, llvm_hsaco, "LLVM HSACO", "compiled code object", BLUE, font, "#FFFFFF")
    box(draw, waveasm_hsaco, "WaveASM HSACO", "compiled code object", ORANGE, font, "#FFFFFF")
    box(draw, launcher, "Same HIP Launcher", "load HSACO and launch wave_kernel", GREEN, font, "#F8FAFC")
    box(draw, gpu, "MI210 GPU", "runs wave_kernel", GREEN, font, "#FFFFFF")

    arrow(draw, (300, 290), (300, 420), BLUE)
    pill(draw, (325, 350), "compile", BLUE, font["label"])
    arrow(draw, (1200, 290), (1200, 420), ORANGE)
    pill(draw, (1225, 350), "compile", ORANGE, font["label"])

    arrow(draw, (485, 520), (590, 585), BLUE)
    arrow(draw, (1015, 520), (910, 585), ORANGE)

    arrow(draw, (750, 705), (750, 760), GREEN)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemm_rocmasm_launch.png"))
    args = parser.parse_args()
    render(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
