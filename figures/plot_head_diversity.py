#!/usr/bin/env python3
"""Plot layerwise within- and across-base GT-MHA head diversity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "analysis" / "transformation_diagnostics" / "results" / "head_diversity"
OUTPUT_STEM = HERE / "head_diversity_layerwise"

INPUTS = {
    "bert": RESULTS / "bert_c4_seed43_fp32" / "summary.json",
    "tinystories": RESULTS / "tinystories" / "summary.json",
}
DATASET_LABELS = {"bert": "BERT", "tinystories": "TinyStories"}
COLORS = {"bert": "#00796B", "tinystories": "#7B2CBF"}
PAIRINGS = {
    "within_base": {"label": "within base", "linestyle": "-", "marker": "o"},
    "across_base": {"label": "across bases", "linestyle": "--", "marker": "s"},
}
LAYERS = np.arange(1, 13)


def read_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def layer_stat(
    summary: dict,
    metric: str,
    pairing: str,
    statistic: str,
) -> np.ndarray:
    key = f"{metric}_{pairing}"
    by_layer = summary["summary"]["by_layer"]
    return np.asarray(
        [float(by_layer[str(layer)][key][statistic]) for layer in LAYERS],
        dtype=float,
    )


def style_axis(ax: plt.Axes, *, ylabel: str, lower_limit: float) -> None:
    ax.set_yscale("log")
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(lower_limit, 1.35)
    ax.set_xticks((1, 3, 6, 9, 12))
    ax.set_xlabel("Layer", fontsize=8.4, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=8.4, labelpad=5)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=18))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", color="#D9DCE1", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=7.5, length=3, color="#55585C")
    ax.tick_params(axis="y", which="minor", length=2, color="#888B8F")
    for spine in ax.spines.values():
        spine.set_color("#777A7E")
        spine.set_linewidth(0.7)


def draw_series(
    ax: plt.Axes,
    summary: dict,
    *,
    dataset: str,
    metric: str,
    pairing: str,
    floor: float,
) -> None:
    color = COLORS[dataset]
    style = PAIRINGS[pairing]
    center = layer_stat(summary, metric, pairing, "mean")
    lower = layer_stat(summary, metric, pairing, "q1")
    upper = layer_stat(summary, metric, pairing, "q3")

    valid_band = upper > floor
    if np.any(valid_band):
        band_lower = np.maximum(lower, floor)
        band_upper = np.maximum(upper, floor)
        ax.fill_between(
            LAYERS,
            band_lower,
            band_upper,
            where=valid_band,
            interpolate=True,
            color=color,
            alpha=0.10 if pairing == "across_base" else 0.15,
            linewidth=0,
            zorder=1,
        )
    ax.plot(
        LAYERS,
        np.maximum(center, floor),
        color=color,
        linestyle=style["linestyle"],
        linewidth=1.75,
        marker=style["marker"],
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        zorder=3,
    )


def main() -> None:
    summaries = {name: read_summary(path) for name, path in INPUTS.items()}
    if summaries["bert"]["protocol"]["precision"] != "FP32 model inference and diversity calculations":
        raise RuntimeError("BERT input must be the FP32 diagnostic")
    if "C=4" not in summaries["bert"]["protocol"]["model"]:
        raise RuntimeError("BERT input is not the C=4 checkpoint")
    if "C=4" not in summaries["tinystories"]["protocol"]["model"]:
        raise RuntimeError("TinyStories input is not the C=4 checkpoint")

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

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.19,
        top=0.78,
        wspace=0.27,
    )
    panels = (
        (
            axes[0],
            "attention_jsd",
            "(a) Attention-map diversity",
            "Pairwise JSD (log scale)",
            1e-10,
        ),
        (
            axes[1],
            "output_cosine_distance",
            "(b) Head-output diversity",
            "Cosine distance (log scale)",
            1e-12,
        ),
    )
    for ax, metric, title, ylabel, floor in panels:
        style_axis(ax, ylabel=ylabel, lower_limit=floor)
        ax.set_title(title, loc="left", pad=5)
        for dataset in ("bert", "tinystories"):
            for pairing in ("within_base", "across_base"):
                draw_series(
                    ax,
                    summaries[dataset],
                    dataset=dataset,
                    metric=metric,
                    pairing=pairing,
                    floor=floor,
                )

    legend_handles = []
    for dataset in ("bert", "tinystories"):
        for pairing in ("within_base", "across_base"):
            style = PAIRINGS[pairing]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=COLORS[dataset],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    marker=style["marker"],
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=COLORS[dataset],
                    label=f"{DATASET_LABELS[dataset]}, {style['label']}",
                )
            )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handlelength=2.2,
        columnspacing=1.25,
    )

    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 300}),
    ):
        figure.savefig(
            OUTPUT_STEM.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.025,
            **kwargs,
        )
    plt.close(figure)
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
