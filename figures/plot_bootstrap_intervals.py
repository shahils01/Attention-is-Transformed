#!/usr/bin/env python3
"""Plot paired-bootstrap differences from MHA for TinyStories results.

Values are copied from the current manuscript appendices.  The output is a
compact, vector two-panel forest plot suitable for the main paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import inch
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
OUT = HERE / "bootstrap_pairwise_intervals.pdf"


@dataclass(frozen=True)
class Interval:
    method: str
    estimate: float
    low: float
    high: float


# Paired differences from MHA, as reported in Appendices F and G.
NLL = [
    Interval("GT-MHA", -0.00177, -0.00191, -0.00163),
    Interval("GQA", 0.00116, 0.00101, 0.00130),
    Interval("Collab. MHA", 0.00029, 0.00015, 0.00044),
    Interval("MQA", 0.00642, 0.00628, 0.00657),
]

GENERATION = [
    Interval("GT-MHA", -0.0303, -0.1122, 0.0508),
    Interval("GQA", -0.0703, -0.1538, 0.0123),
    Interval("Collab. MHA", -0.1555, -0.2392, -0.0702),
    Interval("MQA", -0.3988, -0.4860, -0.3127),
]


COLORS = {
    "GT-MHA": HexColor("#7B2CBF"),
    "GQA": HexColor("#009E73"),
    "Collab. MHA": HexColor("#0072B2"),
    "MQA": HexColor("#D55E00"),
}

BLACK = HexColor("#1A1A1A")
GRID = HexColor("#D8D8D8")
FRAME = HexColor("#777777")


def fmt_tick(value: float, digits: int) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def draw_panel(
    c: canvas.Canvas,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    rows: list[Interval],
    domain: tuple[float, float],
    ticks: list[float],
    digits: int,
    y_label: str,
) -> None:
    left, right, bottom, top = 48, 8, 31, 22
    x0, y0 = x + left, y + bottom
    pw, ph = w - left - right, h - bottom - top

    def sy(value: float) -> float:
        lo, hi = domain
        return y0 + (value - lo) / (hi - lo) * ph

    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 2, y + h - 10, title)

    # Horizontal grid and paired-difference labels.
    for tick in ticks:
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x0, yy, x0 + pw, yy)
        c.setFillColor(BLACK)
        c.setFont("Helvetica", 6.5)
        c.drawRightString(x0 - 5, yy - 2.2, fmt_tick(tick, digits))

    # Zero is the null paired difference.
    zero = sy(0.0)
    c.setStrokeColor(FRAME)
    c.setLineWidth(0.8)
    c.setDash(2, 2)
    c.line(x0, zero, x0 + pw, zero)
    c.setDash()

    c.setStrokeColor(FRAME)
    c.setLineWidth(0.45)
    c.rect(x0, y0, pw, ph, stroke=1, fill=0)

    n = len(rows)
    col_gap = pw / n
    for i, row in enumerate(rows):
        xx = x0 + (i + 0.5) * col_gap

        # Direct method labels remove the need for a legend.
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold" if row.method == "GT-MHA" else "Helvetica", 7.2)
        c.drawCentredString(xx, y0 - 11, row.method)

        # A faint method guide keeps the two panels easy to scan.
        c.setStrokeColor(GRID)
        c.setLineWidth(0.25)
        c.line(xx, y0, xx, y0 + ph)

        lo, mid, hi = sy(row.low), sy(row.estimate), sy(row.high)
        color = COLORS[row.method]
        c.setStrokeColor(color)
        c.setLineWidth(1.8 if row.method == "GT-MHA" else 1.45)
        c.line(xx, lo, xx, hi)
        c.setLineWidth(0.9)
        c.line(xx - 3.5, lo, xx + 3.5, lo)
        c.line(xx - 3.5, hi, xx + 3.5, hi)
        c.setFillColor(color)
        # Small markers keep the very narrow NLL intervals visible.
        c.circle(xx, mid, 2.1 if row.method == "GT-MHA" else 1.8, stroke=0, fill=1)

    c.setFillColor(BLACK)
    c.setFont("Helvetica", 7.1)
    c.saveState()
    c.translate(x + 7, y0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_label)
    c.restoreState()


def main() -> None:
    width, height = 7.0 * inch, 2.25 * inch
    c = canvas.Canvas(str(OUT), pagesize=(width, height))
    c.setTitle("TinyStories paired-bootstrap intervals")

    gap = 8
    panel_w = (width - gap) / 2
    draw_panel(
        c,
        x=0,
        y=0,
        w=panel_w,
        h=height,
        title="(a) Full-validation NLL",
        rows=NLL,
        domain=(-0.0024, 0.0070),
        ticks=[-0.002, 0.000, 0.002, 0.004, 0.006],
        digits=3,
        y_label="Paired difference from MHA",
    )
    draw_panel(
        c,
        x=panel_w + gap,
        y=0,
        w=panel_w,
        h=height,
        title="(b) Generation quality",
        rows=GENERATION,
        domain=(-0.55, 0.12),
        ticks=[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1],
        digits=1,
        y_label="Paired difference from MHA",
    )

    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
