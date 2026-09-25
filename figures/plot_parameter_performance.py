#!/usr/bin/env python3
"""Generate publication-ready parameter--performance trade-off figures.

The values are copied from Tables 1 and 2 of the current manuscript.  The
script writes vector PDFs.  PNG previews can be produced with pdftoppm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Point:
    method: str
    attention_m: float
    model_m: float
    performance: float


BERT = [
    Point("MHA", 28.37, 109.5, 70.3),
    Point("GQA", 18.92, 100.0, 70.3),
    Point("Collaborative MHA", 15.38, 96.5, 68.9),
    Point("GT-MHA", 14.98, 96.1, 70.5),
]

TINYSTORIES = [
    Point("MHA", 50.33, 152.10, 0.2893),
    Point("GQA", 31.46, 133.23, 0.2905),
    Point("MQA", 26.74, 128.51, 0.2957),
    Point("Collaborative MHA", 26.75, 128.52, 0.2896),
    Point("GT-MHA", 22.81, 124.58, 0.2875),
]

TINYSTORIES_GENERATION = [
    Point("MHA", 50.33, 152.10, 8.065),
    Point("GQA", 31.46, 133.23, 7.995),
    Point("MQA", 26.74, 128.51, 7.667),
    Point("Collaborative MHA", 26.75, 128.52, 7.910),
    Point("GT-MHA", 22.81, 124.58, 8.035),
]


# Colorblind-safe palette.  GT-MHA is intentionally the strongest color.
COLORS = {
    "MHA": HexColor("#E69F00"),
    "GQA": HexColor("#009E73"),
    "MQA": HexColor("#D55E00"),
    "Collaborative MHA": HexColor("#0072B2"),
    "GT-MHA": HexColor("#7B2CBF"),
}

MARKERS = {
    "MHA": "circle",
    "GQA": "square",
    "MQA": "triangle",
    "Collaborative MHA": "down_triangle",
    "GT-MHA": "diamond",
}

DISPLAY_LABELS = {
    "Collaborative MHA": "Collab. MHA",
}

BLACK = HexColor("#1A1A1A")
GRID = HexColor("#D7D7D7")
FRAME = HexColor("#777777")
WHITE = Color(1, 1, 1)


def draw_marker(c: canvas.Canvas, x: float, y: float, method: str) -> None:
    size = 4.5 if method != "GT-MHA" else 5.5
    color = COLORS[method]
    marker = MARKERS[method]
    c.setFillColor(color)
    c.setStrokeColor(WHITE)
    c.setLineWidth(0.9)
    if marker == "circle":
        c.circle(x, y, size, stroke=1, fill=1)
    elif marker == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    elif marker == "triangle":
        p = c.beginPath()
        p.moveTo(x, y + size * 1.2)
        p.lineTo(x - size * 1.1, y - size)
        p.lineTo(x + size * 1.1, y - size)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
    elif marker == "down_triangle":
        p = c.beginPath()
        p.moveTo(x, y - size * 1.2)
        p.lineTo(x - size * 1.1, y + size)
        p.lineTo(x + size * 1.1, y + size)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
    elif marker == "diamond":
        p = c.beginPath()
        p.moveTo(x, y + size * 1.25)
        p.lineTo(x - size, y)
        p.lineTo(x, y - size * 1.25)
        p.lineTo(x + size, y)
        p.close()
        c.drawPath(p, stroke=1, fill=1)


def draw_axis_label(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    *,
    vertical: bool = False,
) -> None:
    c.saveState()
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 7.6)
    if vertical:
        c.translate(x, y)
        c.rotate(90)
        c.drawCentredString(0, 0, text)
    else:
        c.drawCentredString(x, y, text)
    c.restoreState()


def draw_panel(
    c: canvas.Canvas,
    *,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    title: str,
    points: list[Point],
    x_attr: str,
    x_domain: tuple[float, float],
    x_ticks: list[float],
    y_domain: tuple[float, float],
    y_ticks: list[float],
    x_title: str,
    y_title: str,
    invert_y: bool,
    label_offsets: dict[str, tuple[float, float, str]],
    y_digits: int,
) -> None:
    left, right, bottom, top = 47, 12, 33, 24
    x0 = panel_x + left
    y0 = panel_y + bottom
    pw = panel_w - left - right
    ph = panel_h - bottom - top

    def sx(value: float) -> float:
        lo, hi = x_domain
        return x0 + (value - lo) / (hi - lo) * pw

    def sy(value: float) -> float:
        lo, hi = y_domain
        ratio = (value - lo) / (hi - lo)
        if invert_y:
            ratio = 1 - ratio
        return y0 + ratio * ph

    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 8.8)
    c.drawString(panel_x + 2, panel_y + panel_h - 10, title)

    # Light horizontal grid lines and a thin plot frame.
    c.setLineWidth(0.35)
    for tick in y_ticks:
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.line(x0, yy, x0 + pw, yy)
        c.setFillColor(BLACK)
        c.setFont("Helvetica", 6.8)
        label = f"{tick:.{y_digits}f}"
        c.drawRightString(x0 - 5, yy - 2.4, label)

    for tick in x_ticks:
        xx = sx(tick)
        c.setStrokeColor(GRID)
        c.line(xx, y0, xx, y0 + ph)
        c.setFillColor(BLACK)
        c.setFont("Helvetica", 6.8)
        label = f"{tick:g}"
        c.drawCentredString(xx, y0 - 10, label)

    c.setStrokeColor(FRAME)
    c.setLineWidth(0.55)
    c.rect(x0, y0, pw, ph, stroke=1, fill=0)

    draw_axis_label(c, x0 + pw / 2, panel_y + 4, x_title)
    draw_axis_label(c, panel_x + 8, y0 + ph / 2, y_title, vertical=True)

    for point in points:
        x_value = getattr(point, x_attr)
        xx, yy = sx(x_value), sy(point.performance)
        draw_marker(c, xx, yy, point.method)

        dx, dy, anchor = label_offsets[point.method]
        c.setFillColor(BLACK)
        font = "Helvetica-Bold" if point.method == "GT-MHA" else "Helvetica"
        c.setFont(font, 7.0)
        label = point.method
        label_x, label_y = xx + dx, yy + dy
        if anchor == "right":
            c.drawRightString(label_x, label_y, label)
        elif anchor == "center":
            c.drawCentredString(label_x, label_y, label)
        else:
            c.drawString(label_x, label_y, label)


def draw_compact_panel(
    c: canvas.Canvas,
    *,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    title: str,
    points: list[Point],
    x_domain: tuple[float, float],
    x_ticks: list[float],
    y_domain: tuple[float, float],
    y_ticks: list[float],
    y_title: str,
    y_digits: int,
    label_offsets: dict[str, tuple[float, float, str]],
) -> None:
    """Draw a dense panel intended for a three-across main-paper figure."""
    left, right, bottom, top = 34, 5, 20, 18
    x0 = panel_x + left
    y0 = panel_y + bottom
    pw = panel_w - left - right
    ph = panel_h - bottom - top

    def sx(value: float) -> float:
        lo, hi = x_domain
        return x0 + (value - lo) / (hi - lo) * pw

    def sy(value: float) -> float:
        lo, hi = y_domain
        return y0 + (value - lo) / (hi - lo) * ph

    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 7.7)
    c.drawString(panel_x + 2, panel_y + panel_h - 9, title)

    c.setLineWidth(0.3)
    for tick in y_ticks:
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.line(x0, yy, x0 + pw, yy)
        c.setFillColor(BLACK)
        c.setFont("Helvetica", 6.1)
        c.drawRightString(x0 - 4, yy - 2.1, f"{tick:.{y_digits}f}")

    for tick in x_ticks:
        xx = sx(tick)
        c.setStrokeColor(GRID)
        c.line(xx, y0, xx, y0 + ph)
        c.setFillColor(BLACK)
        c.setFont("Helvetica", 6.1)
        c.drawCentredString(xx, y0 - 8, f"{tick:g}")

    c.setStrokeColor(FRAME)
    c.setLineWidth(0.5)
    c.rect(x0, y0, pw, ph, stroke=1, fill=0)

    c.saveState()
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 6.5)
    c.translate(panel_x + 7, y0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_title)
    c.restoreState()

    for point in points:
        xx, yy = sx(point.attention_m), sy(point.performance)
        draw_marker(c, xx, yy, point.method)
        dx, dy, anchor = label_offsets[point.method]
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold" if point.method == "GT-MHA" else "Helvetica", 6.2)
        label = DISPLAY_LABELS.get(point.method, point.method)
        label_x, label_y = xx + dx, yy + dy
        if anchor == "right":
            c.drawRightString(label_x, label_y, label)
        elif anchor == "center":
            c.drawCentredString(label_x, label_y, label)
        else:
            c.drawString(label_x, label_y, label)


def draw_shared_legend(c: canvas.Canvas, *, width: float, y: float) -> None:
    labels = ["MHA", "GQA", "MQA", "Collaborative MHA", "GT-MHA"]
    font_size = 6.5
    marker_gap = 8
    item_gap = 12
    widths = [marker_gap + stringWidth(label, "Helvetica", font_size) for label in labels]
    total = sum(widths) + item_gap * (len(labels) - 1)
    x = (width - total) / 2
    for method, item_w in zip(labels, widths):
        draw_marker(c, x + 3, y + 2, method)
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold" if method == "GT-MHA" else "Helvetica", font_size)
        c.drawString(x + marker_gap, y, method)
        x += item_w + item_gap


def make_main_figure(output: Path) -> None:
    """Create the compact three-panel figure for the main paper."""
    width, height = 7.0 * 72, 2.05 * 72
    c = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    c.setTitle("GT-MHA parameter-performance trade-offs")

    gap = 7
    panel_y = 7
    panel_h = height - 9
    panel_w = (width - 2 * gap) / 3

    draw_compact_panel(
        c,
        panel_x=0,
        panel_y=panel_y,
        panel_w=panel_w,
        panel_h=panel_h,
        title="(a) BERT + GLUE",
        points=BERT,
        x_domain=(13.8, 30.0),
        x_ticks=[15, 20, 25, 30],
        y_domain=(68.6, 70.7),
        y_ticks=[69.0, 69.5, 70.0, 70.5],
        y_title="GLUE score",
        y_digits=1,
        label_offsets={
            "MHA": (-5, 4, "right"),
            "GQA": (5, 4, "left"),
            "Collaborative MHA": (5, -8, "left"),
            "GT-MHA": (5, 3, "left"),
        },
    )
    draw_compact_panel(
        c,
        panel_x=panel_w + gap,
        panel_y=panel_y,
        panel_w=panel_w,
        panel_h=panel_h,
        title="(b) TinyStories NLL",
        points=TINYSTORIES,
        x_domain=(20.0, 53.0),
        x_ticks=[20, 30, 40, 50],
        y_domain=(0.2865, 0.2965),
        y_ticks=[0.288, 0.290, 0.292, 0.294, 0.296],
        y_title="Validation NLL",
        y_digits=3,
        label_offsets={
            "MHA": (-5, 4, "right"),
            "GQA": (5, 3, "left"),
            "MQA": (5, -8, "left"),
            "Collaborative MHA": (0, -11, "center"),
            "GT-MHA": (5, 3, "left"),
        },
    )
    draw_compact_panel(
        c,
        panel_x=2 * (panel_w + gap),
        panel_y=panel_y,
        panel_w=panel_w,
        panel_h=panel_h,
        title="(c) TinyStories generation",
        points=TINYSTORIES_GENERATION,
        x_domain=(20.0, 53.0),
        x_ticks=[20, 30, 40, 50],
        y_domain=(7.62, 8.10),
        y_ticks=[7.7, 7.8, 7.9, 8.0, 8.1],
        y_title="Mean judge score",
        y_digits=1,
        label_offsets={
            "MHA": (-5, 4, "right"),
            "GQA": (5, 3, "left"),
            "MQA": (5, -8, "left"),
            "Collaborative MHA": (5, 3, "left"),
            "GT-MHA": (5, 3, "left"),
        },
    )

    c.setFillColor(BLACK)
    c.setFont("Helvetica", 7.0)
    c.drawCentredString(width / 2, 1.5, "Attention parameters (M)")
    c.showPage()
    c.save()


def make_figure(output: Path, *, whole_model: bool) -> None:
    width, height = 7.0 * 72, 2.75 * 72
    c = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    c.setTitle("GT-MHA parameter-performance trade-offs")
    gap = 16
    panel_w = (width - gap) / 2

    if whole_model:
        x_attr = "model_m"
        bert_x_domain = (94.5, 111.0)
        bert_x_ticks = [95, 100, 105, 110]
        tiny_x_domain = (122.0, 155.0)
        tiny_x_ticks = [125, 135, 145, 155]
        bert_x_title = "Backbone parameters (millions)"
        tiny_x_title = "Total parameters (millions)"
        bert_offsets = {
            "MHA": (-5, 7, "right"),
            "GQA": (5, 7, "left"),
            "Collaborative MHA": (6, -10, "left"),
            "GT-MHA": (7, 3, "left"),
        }
        tiny_offsets = {
            "MHA": (-6, 6, "right"),
            "GQA": (6, -10, "left"),
            "MQA": (6, -10, "left"),
            "Collaborative MHA": (-7, 5, "right"),
            "GT-MHA": (7, 3, "left"),
        }
    else:
        x_attr = "attention_m"
        bert_x_domain = (13.8, 30.0)
        bert_x_ticks = [15, 20, 25, 30]
        tiny_x_domain = (20.0, 53.0)
        tiny_x_ticks = [20, 30, 40, 50]
        bert_x_title = "Attention parameters (millions)"
        tiny_x_title = "Attention parameters (millions)"
        bert_offsets = {
            "MHA": (-5, 7, "right"),
            "GQA": (6, 7, "left"),
            "Collaborative MHA": (6, -10, "left"),
            "GT-MHA": (7, 3, "left"),
        }
        tiny_offsets = {
            "MHA": (-6, 6, "right"),
            "GQA": (6, -10, "left"),
            "MQA": (6, -10, "left"),
            "Collaborative MHA": (-7, 5, "right"),
            "GT-MHA": (7, 3, "left"),
        }

    draw_panel(
        c,
        panel_x=0,
        panel_y=0,
        panel_w=panel_w,
        panel_h=height,
        title="(a) BERT + GLUE",
        points=BERT,
        x_attr=x_attr,
        x_domain=bert_x_domain,
        x_ticks=bert_x_ticks,
        y_domain=(68.6, 70.7),
        y_ticks=[69.0, 69.5, 70.0, 70.5],
        x_title=bert_x_title,
        y_title="Official GLUE score (higher is better)",
        invert_y=False,
        label_offsets=bert_offsets,
        y_digits=1,
    )
    draw_panel(
        c,
        panel_x=panel_w + gap,
        panel_y=0,
        panel_w=panel_w,
        panel_h=height,
        title="(b) TinyStoriesV2",
        points=TINYSTORIES,
        x_attr=x_attr,
        x_domain=tiny_x_domain,
        x_ticks=tiny_x_ticks,
        y_domain=(0.2865, 0.2965),
        y_ticks=[0.288, 0.290, 0.292, 0.294, 0.296],
        x_title=tiny_x_title,
        y_title="Full-validation NLL (lower is better)",
        invert_y=False,
        label_offsets=tiny_offsets,
        y_digits=3,
    )
    c.showPage()
    c.save()


if __name__ == "__main__":
    make_figure(HERE / "parameter_performance_attention.pdf", whole_model=False)
    make_figure(HERE / "parameter_performance_model.pdf", whole_model=True)
    make_main_figure(HERE / "parameter_performance_main.pdf")
