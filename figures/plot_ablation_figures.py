#!/usr/bin/env python3
"""Generate vector ablation figures from the manuscript's updated tables.

The plots intentionally use compact small multiples, direct labels, and a
colorblind-safe palette.  PDF is the publication artifact; PNG files are only
previews generated separately with ``pdftoppm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent

BLACK = HexColor("#202124")
MID = HexColor("#666A70")
FRAME = HexColor("#8A8D91")
GRID = HexColor("#E2E4E7")
PURPLE = HexColor("#7B2CBF")
BLUE = HexColor("#0072B2")
ORANGE = HexColor("#D55E00")
PALE_PURPLE = HexColor("#C8A5DE")
WHITE = Color(1, 1, 1)


@dataclass(frozen=True)
class Variant:
    name: str
    short: tuple[str, ...]
    color: Color
    marker: str
    glue: float
    nll: float
    generation: float


TRANSFORMATIONS = (
    Variant("QKV identity", ("QKV", "identity"), ORANGE, "triangle", 70.0, 0.2996, 7.312),
    Variant("QK identity", ("QK", "identity"), BLUE, "square", 70.2, 0.2961, 7.881),
    Variant("GT-MHA", ("GT-MHA",), PURPLE, "diamond", 70.5, 0.2903, 8.035),
)


@dataclass(frozen=True)
class BaseCount:
    c: int
    attention_m: float
    mlm: float
    mlm_std: float
    glue: float


BASE_COUNTS = (
    BaseCount(1, 9.67, 2.25, 0.041, 68.3),
    BaseCount(2, 11.44, 2.15, 0.008, 70.0),
    BaseCount(4, 14.98, 2.12, 0.023, 70.5),
)

MAPPINGS = (
    ("First-order", 0.2903, PURPLE, "diamond"),
    ("Second-order", 0.2939, BLUE, "square"),
    ("Exact exp.", 0.2942, ORANGE, "circle"),
)


def draw_marker(c: canvas.Canvas, x: float, y: float, marker: str, color: Color, *, main: bool = False) -> None:
    r = 4.5 if main else 4.0
    c.setFillColor(color)
    c.setStrokeColor(WHITE)
    c.setLineWidth(0.8)
    if marker == "circle":
        c.circle(x, y, r, stroke=1, fill=1)
    elif marker == "square":
        c.rect(x - r, y - r, 2 * r, 2 * r, stroke=1, fill=1)
    elif marker == "triangle":
        p = c.beginPath()
        p.moveTo(x, y + 1.15 * r)
        p.lineTo(x - 1.08 * r, y - r)
        p.lineTo(x + 1.08 * r, y - r)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
    elif marker == "diamond":
        p = c.beginPath()
        p.moveTo(x, y + 1.25 * r)
        p.lineTo(x - r, y)
        p.lineTo(x, y - 1.25 * r)
        p.lineTo(x + r, y)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
    else:
        raise ValueError(marker)


def panel_title(c: canvas.Canvas, x: float, y: float, text: str) -> None:
    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 8.3)
    c.drawString(x, y, text)


def categorical_panel(
    c: canvas.Canvas,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    values: Sequence[float],
    domain: tuple[float, float],
    ticks: Sequence[float],
    digits: int,
    y_label: str,
    value_format: Callable[[float], str],
) -> None:
    left, right, bottom, top = 43, 8, 37, 22
    x0, y0 = x + left, y + bottom
    pw, ph = w - left - right, h - bottom - top
    xs = [x0 + pw * q for q in (0.12, 0.50, 0.88)]

    def sy(value: float) -> float:
        lo, hi = domain
        return y0 + (value - lo) / (hi - lo) * ph

    panel_title(c, x + 6, y + h - 10, title)

    for tick in ticks:
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x0, yy, x0 + pw, yy)
        c.setFillColor(MID)
        c.setFont("Helvetica", 6.6)
        c.drawRightString(x0 - 5, yy - 2.2, f"{tick:.{digits}f}")

    c.setStrokeColor(FRAME)
    c.setLineWidth(0.5)
    c.rect(x0, y0, pw, ph, stroke=1, fill=0)

    # Connecting line emphasizes that these form a nested sequence.
    c.setStrokeColor(PALE_PURPLE)
    c.setLineWidth(1.25)
    path = c.beginPath()
    path.moveTo(xs[0], sy(values[0]))
    for xx, value in zip(xs[1:], values[1:]):
        path.lineTo(xx, sy(value))
    c.drawPath(path, stroke=1, fill=0)

    for xx, value, variant in zip(xs, values, TRANSFORMATIONS):
        yy = sy(value)
        draw_marker(c, xx, yy, variant.marker, variant.color, main=variant.name == "GT-MHA")
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold" if variant.name == "GT-MHA" else "Helvetica", 7.0)
        c.drawCentredString(xx, yy + 8.0, value_format(value))
        c.setFont("Helvetica-Bold" if variant.name == "GT-MHA" else "Helvetica", 6.5)
        if len(variant.short) == 1:
            c.drawCentredString(xx, y0 - 13, variant.short[0])
        else:
            c.drawCentredString(xx, y0 - 10, variant.short[0])
            c.drawCentredString(xx, y0 - 18, variant.short[1])

    c.saveState()
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 7.0)
    c.translate(x + 9, y0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_label)
    c.restoreState()


def draw_transformation_figure() -> Path:
    out = HERE / "ablation_transformations.pdf"
    width, height = 7.0 * inch, 2.38 * inch
    c = canvas.Canvas(str(out), pagesize=(width, height))
    c.setTitle("Contribution of learned GT-MHA transformations")
    gap = 8
    panel_w = (width - 2 * gap) / 3

    categorical_panel(
        c,
        x=0,
        y=0,
        w=panel_w,
        h=height,
        title="(a) BERT + GLUE",
        values=[v.glue for v in TRANSFORMATIONS],
        domain=(69.88, 70.62),
        ticks=(70.0, 70.2, 70.4, 70.6),
        digits=1,
        y_label="Official GLUE test score",
        value_format=lambda v: f"{v:.1f}",
    )
    categorical_panel(
        c,
        x=panel_w + gap,
        y=0,
        w=panel_w,
        h=height,
        title="(b) TinyStories loss",
        values=[v.nll for v in TRANSFORMATIONS],
        domain=(0.2888, 0.3010),
        ticks=(0.290, 0.294, 0.298),
        digits=3,
        y_label="Validation NLL",
        value_format=lambda v: f"{v:.4f}",
    )
    categorical_panel(
        c,
        x=2 * (panel_w + gap),
        y=0,
        w=panel_w,
        h=height,
        title="(c) Generation quality",
        values=[v.generation for v in TRANSFORMATIONS],
        domain=(7.18, 8.16),
        ticks=(7.2, 7.6, 8.0),
        digits=1,
        y_label="Mean rating",
        value_format=lambda v: f"{v:.3f}",
    )
    c.showPage()
    c.save()
    return out


def quantitative_panel(
    c: canvas.Canvas,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    y_getter: Callable[[BaseCount], float],
    domain: tuple[float, float],
    ticks: Sequence[float],
    digits: int,
    y_label: str,
    with_error: bool,
) -> None:
    left, right, bottom, top = 42, 8, 30, 22
    x0, y0 = x + left, y + bottom
    pw, ph = w - left - right, h - bottom - top
    x_domain = (9.0, 15.6)
    x_ticks = (10, 12, 14)

    def sx(value: float) -> float:
        lo, hi = x_domain
        return x0 + (value - lo) / (hi - lo) * pw

    def sy(value: float) -> float:
        lo, hi = domain
        return y0 + (value - lo) / (hi - lo) * ph

    panel_title(c, x + 6, y + h - 10, title)
    for tick in ticks:
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x0, yy, x0 + pw, yy)
        c.setFillColor(MID)
        c.setFont("Helvetica", 6.5)
        c.drawRightString(x0 - 5, yy - 2.2, f"{tick:.{digits}f}")
    for tick in x_ticks:
        xx = sx(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.3)
        c.line(xx, y0, xx, y0 + ph)
        c.setFillColor(MID)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(xx, y0 - 10, f"{tick:g}")
    c.setStrokeColor(FRAME)
    c.setLineWidth(0.5)
    c.rect(x0, y0, pw, ph, stroke=1, fill=0)

    points = [(sx(row.attention_m), sy(y_getter(row)), row) for row in BASE_COUNTS]
    c.setStrokeColor(PALE_PURPLE)
    c.setLineWidth(1.35)
    path = c.beginPath()
    path.moveTo(points[0][0], points[0][1])
    for xx, yy, _ in points[1:]:
        path.lineTo(xx, yy)
    c.drawPath(path, stroke=1, fill=0)

    markers = ("circle", "square", "diamond")
    label_offsets = ((5, -10), (5, 5), (-27, 5))
    for (xx, yy, row), marker, (dx, dy) in zip(points, markers, label_offsets):
        if with_error:
            lo, hi = sy(row.mlm - row.mlm_std), sy(row.mlm + row.mlm_std)
            c.setStrokeColor(PURPLE)
            c.setLineWidth(1.0)
            c.line(xx, lo, xx, hi)
            c.line(xx - 3, lo, xx + 3, lo)
            c.line(xx - 3, hi, xx + 3, hi)
        draw_marker(c, xx, yy, marker, PURPLE, main=row.c == 4)
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold" if row.c == 4 else "Helvetica", 7.0)
        c.drawString(xx + dx, yy + dy, f"C={row.c}")

    c.setFillColor(BLACK)
    c.setFont("Helvetica", 7.0)
    c.drawCentredString(x0 + pw / 2, y + 3, "Attention parameters (M)")
    c.saveState()
    c.translate(x + 8, y0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_label)
    c.restoreState()


def mapping_panel(c: canvas.Canvas, *, x: float, y: float, w: float, h: float) -> None:
    left, right, bottom, top = 42, 8, 37, 22
    x0, y0 = x + left, y + bottom
    pw, ph = w - left - right, h - bottom - top
    xs = [x0 + pw * q for q in (0.12, 0.50, 0.88)]
    domain = (0.2895, 0.2950)
    ticks = (0.290, 0.292, 0.294)

    def sy(value: float) -> float:
        lo, hi = domain
        return y0 + (value - lo) / (hi - lo) * ph

    panel_title(c, x + 6, y + h - 10, "(c) Transformation map")
    for tick in ticks:
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x0, yy, x0 + pw, yy)
        c.setFillColor(MID)
        c.setFont("Helvetica", 6.5)
        c.drawRightString(x0 - 5, yy - 2.2, f"{tick:.3f}")
    c.setStrokeColor(FRAME)
    c.setLineWidth(0.5)
    c.rect(x0, y0, pw, ph, stroke=1, fill=0)

    c.setStrokeColor(PALE_PURPLE)
    c.setLineWidth(1.2)
    path = c.beginPath()
    path.moveTo(xs[0], sy(MAPPINGS[0][1]))
    for xx, (_, value, _, _) in zip(xs[1:], MAPPINGS[1:]):
        path.lineTo(xx, sy(value))
    c.drawPath(path, stroke=1, fill=0)

    for xx, (label, value, color, marker) in zip(xs, MAPPINGS):
        yy = sy(value)
        draw_marker(c, xx, yy, marker, color, main=label == "First-order")
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold" if label == "First-order" else "Helvetica", 6.9)
        c.drawCentredString(xx, yy + 8, f"{value:.4f}")
        words = label.split()
        if len(words) == 1:
            c.drawCentredString(xx, y0 - 13, label)
        else:
            c.drawCentredString(xx, y0 - 10, words[0])
            c.drawCentredString(xx, y0 - 18, " ".join(words[1:]))

    c.saveState()
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 7.0)
    c.translate(x + 8, y0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Validation NLL")
    c.restoreState()


def draw_capacity_mapping_figure() -> Path:
    out = HERE / "ablation_capacity_mapping.pdf"
    width, height = 7.0 * inch, 2.38 * inch
    c = canvas.Canvas(str(out), pagesize=(width, height))
    c.setTitle("GT-MHA base-count and transformation-mapping ablations")
    gap = 8
    panel_w = (width - 2 * gap) / 3
    quantitative_panel(
        c,
        x=0,
        y=0,
        w=panel_w,
        h=height,
        title="(a) Base count: MLM",
        y_getter=lambda row: row.mlm,
        domain=(2.06, 2.33),
        ticks=(2.10, 2.20, 2.30),
        digits=2,
        y_label="MLM validation loss",
        with_error=True,
    )
    quantitative_panel(
        c,
        x=panel_w + gap,
        y=0,
        w=panel_w,
        h=height,
        title="(b) Base count: GLUE",
        y_getter=lambda row: row.glue,
        domain=(67.8, 70.9),
        ticks=(68, 69, 70),
        digits=0,
        y_label="Official GLUE test score",
        with_error=False,
    )
    mapping_panel(c, x=2 * (panel_w + gap), y=0, w=panel_w, h=height)
    c.showPage()
    c.save()
    return out


def main() -> None:
    for path in (draw_transformation_figure(), draw_capacity_mapping_figure()):
        print(path)


if __name__ == "__main__":
    main()
