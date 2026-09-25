#!/usr/bin/env python3
"""Plot layerwise GT-MHA transformation magnitude and conditioning.

The BERT panels summarize three independently pretrained checkpoints.  For
each seed and layer, the statistic is first aggregated across heads; the bold
curve is the median across seeds and the band spans their range.  TinyStories
uses one checkpoint, so its curve and band show the median and interquartile
range across heads rather than training-run uncertainty.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "analysis" / "transformation_diagnostics" / "results"
OUTPUT_STEM = HERE / "transformation_magnitude_conditioning"

BERT_FILES = {
    seed: RESULTS
    / "bert_residual_b4g8h12"
    / f"seed{seed}"
    / "per_head_metrics.csv"
    for seed in (42, 43, 44)
}
TINYSTORIES_FILE = (
    RESULTS
    / "tinystories_residual_step250000"
    / "per_head_metrics.csv"
)

PATHWAYS = ("query_key", "value")
COLORS = {"query_key": "#7B2CBF", "value": "#00796B"}
LABELS = {"query_key": "Query–key", "value": "Value"}
MARKERS = {"query_key": "o", "value": "s"}
LAYERS = np.arange(1, 13)


def read_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "pathway": row["pathway"],
                    "distance": float(row["distance_from_identity"]),
                    "condition": float(row["condition_number"]),
                }
            )
    return rows


def layer_values(
    rows: list[dict[str, float | int | str]], pathway: str, metric: str
) -> list[np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["pathway"] == pathway:
            grouped[int(row["layer"])].append(float(row[metric]))
    if set(grouped) != set(LAYERS):
        raise RuntimeError(f"Missing layers for {pathway}/{metric}: {sorted(grouped)}")
    return [np.asarray(grouped[layer], dtype=float) for layer in LAYERS]


def bert_summary(
    rows_by_seed: dict[int, list[dict[str, float | int | str]]],
    pathway: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_seed = np.asarray(
        [
            [np.median(values) for values in layer_values(rows, pathway, metric)]
            for rows in rows_by_seed.values()
        ]
    )
    return (
        np.median(per_seed, axis=0),
        np.min(per_seed, axis=0),
        np.max(per_seed, axis=0),
    )


def tinystories_summary(
    rows: list[dict[str, float | int | str]], pathway: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = layer_values(rows, pathway, metric)
    return (
        np.asarray([np.median(layer) for layer in values]),
        np.asarray([np.quantile(layer, 0.25) for layer in values]),
        np.asarray([np.quantile(layer, 0.75) for layer in values]),
    )


def condition_formatter(value: float, _: int) -> str:
    return f"{value:g}" if value in (1, 2, 4, 8, 16) else ""


def style_axis(ax: plt.Axes, *, row: int, col: int) -> None:
    ax.set_xlim(0.6, 12.4)
    ax.set_xticks((1, 3, 6, 9, 12))
    ax.grid(axis="y", color="#D9DCE1", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.5, length=3, color="#55585C")
    for spine in ax.spines.values():
        spine.set_color("#777A7E")
        spine.set_linewidth(0.7)
    if row == 1:
        ax.set_xlabel("Layer", fontsize=8.2, labelpad=3)
    else:
        ax.tick_params(labelbottom=False)
    if col == 0:
        ax.set_ylabel("Normalized distance from identity", fontsize=8.2, labelpad=5)
    else:
        ax.set_ylabel(r"Median condition number $\kappa_2$", fontsize=8.2, labelpad=5)
        if row == 1:
            ax.set_ylabel(
                r"Median condition number $\kappa_2$ (log scale)",
                fontsize=8.2,
                labelpad=5,
            )
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_locator(LogLocator(base=2, subs=(1.0,)))
            ax.yaxis.set_major_formatter(FuncFormatter(condition_formatter))
            ax.yaxis.set_minor_formatter(NullFormatter())


def draw_series(
    ax: plt.Axes,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pathway: str,
) -> None:
    color = COLORS[pathway]
    ax.fill_between(LAYERS, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.plot(
        LAYERS,
        center,
        color=color,
        linewidth=1.85,
        marker=MARKERS[pathway],
        markersize=3.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.05,
        zorder=3,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
            "axes.titlesize": 9.2,
            "text.color": "#202124",
            "axes.labelcolor": "#202124",
            "xtick.color": "#4F5358",
            "ytick.color": "#4F5358",
        }
    )

    bert_rows = {seed: read_rows(path) for seed, path in BERT_FILES.items()}
    tiny_rows = read_rows(TINYSTORIES_FILE)

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.15), sharex="col")
    plt.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.865, wspace=0.25, hspace=0.24)

    panel_titles = (
        ("(a) BERT: transformation magnitude", "(b) BERT: conditioning"),
        ("(c) TinyStories: transformation magnitude", "(d) TinyStories: conditioning"),
    )
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            style_axis(ax, row=row, col=col)
            ax.set_title(panel_titles[row][col], loc="left", pad=5)

    axes[0, 0].set_ylim(0.04, 0.35)
    axes[0, 0].set_yticks((0.05, 0.15, 0.25, 0.35))
    axes[1, 0].set_ylim(0.10, 0.70)
    axes[1, 0].set_yticks((0.1, 0.3, 0.5, 0.7))
    axes[0, 1].set_ylim(1.12, 2.18)
    axes[0, 1].set_yticks((1.2, 1.4, 1.6, 1.8, 2.0))
    axes[1, 1].set_ylim(1.0, 16.5)

    for pathway in PATHWAYS:
        draw_series(
            axes[0, 0],
            *bert_summary(bert_rows, pathway, "distance"),
            pathway,
        )
        draw_series(
            axes[0, 1],
            *bert_summary(bert_rows, pathway, "condition"),
            pathway,
        )
        draw_series(
            axes[1, 0],
            *tinystories_summary(tiny_rows, pathway, "distance"),
            pathway,
        )
        draw_series(
            axes[1, 1],
            *tinystories_summary(tiny_rows, pathway, "condition"),
            pathway,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[pathway],
            linewidth=1.9,
            marker=MARKERS[pathway],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.05,
            label=LABELS[pathway],
        )
        for pathway in PATHWAYS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.975),
        ncol=2,
        frameon=False,
        fontsize=8.2,
        handlelength=2.3,
        columnspacing=1.8,
    )
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
