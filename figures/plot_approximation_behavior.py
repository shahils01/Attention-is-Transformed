#!/usr/bin/env python3
"""Plot counterfactual approximation behavior of learned GT-MHA generators."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "analysis" / "transformation_diagnostics" / "results" / "approximation"
OUTPUT_STEM = HERE / "transformation_approximation_behavior"

VARIANTS = (
    ("residual_step250000", "First-order\ntrained"),
    ("quadratic_step250000", "Second-order\ntrained"),
    ("exact_step250000", "Exact\ntrained"),
)
PATHWAYS = ("query_key", "value")
PATHWAY_LABELS = {"query_key": "Query–key", "value": "Value"}
PATHWAY_COLORS = {"query_key": "#7B2CBF", "value": "#00796B"}
ERROR_FIELDS = (
    ("first_order_relative_error", r"$I+A_h$", "#D55E00"),
    ("second_order_relative_error", r"$I+A_h+\frac{1}{2}A_h^2$", "#2563A6"),
)


def read_rows(directory: str) -> list[dict[str, float | int | str]]:
    path = RESULTS / directory / "per_head_approximation.csv"
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
                    "first_order_relative_error": float(row["first_order_relative_error"]),
                    "second_order_relative_error": float(row["second_order_relative_error"]),
                    "generator_fro_normalized": float(row["generator_fro_normalized"]),
                }
            )
    return rows


def values(
    rows: list[dict[str, float | int | str]], pathway: str, field: str
) -> np.ndarray:
    result = np.asarray(
        [float(row[field]) for row in rows if row["pathway"] == pathway],
        dtype=float,
    )
    if result.size != 192:
        raise RuntimeError(f"Expected 192 {pathway} values for {field}, found {result.size}")
    return result


def add_box(
    ax: plt.Axes,
    data: np.ndarray,
    position: float,
    color: str,
    *,
    width: float = 0.28,
    label_median: bool = True,
    log_axis: bool = False,
) -> None:
    ax.boxplot(
        [data],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=True,
        whis=(5, 95),
        manage_ticks=False,
        boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.22, "linewidth": 1.3},
        whiskerprops={"color": color, "linewidth": 1.15},
        capprops={"color": color, "linewidth": 1.15},
        medianprops={"color": color, "linewidth": 2.2},
        flierprops={
            "marker": "o",
            "markersize": 2.4,
            "markerfacecolor": "none",
            "markeredgecolor": color,
            "markeredgewidth": 0.6,
            "alpha": 0.38,
        },
    )

    median = float(np.median(data))
    if label_median:
        if log_axis:
            y = median * 1.30
        else:
            y = median + 0.055
        text = f"{median:.3f}" if median >= 0.01 else f"{median:.4f}"
        ax.text(
            position,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="semibold",
            color="#262626",
            zorder=5,
        )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8DCE2", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.tick_params(colors="#444444", labelsize=8.5)


def main() -> None:
    rows_by_variant = {directory: read_rows(directory) for directory, _ in VARIANTS}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.35, 3.55), constrained_layout=False)
    x = np.arange(len(VARIANTS), dtype=float)

    for ax, pathway, title in zip(
        axes[:2],
        PATHWAYS,
        ("(a) Query–key approximation", "(b) Value approximation"),
    ):
        for variant_index, (directory, _) in enumerate(VARIANTS):
            for order_index, (field, _, color) in enumerate(ERROR_FIELDS):
                offset = -0.18 if order_index == 0 else 0.18
                add_box(
                    ax,
                    values(rows_by_variant[directory], pathway, field),
                    x[variant_index] + offset,
                    color,
                    log_axis=True,
                )
        ax.set_yscale("log")
        ax.set_ylim(2.5e-5, 12.0)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=14))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left", fontweight="semibold", pad=7)
        ax.set_ylabel(r"Relative error to $\exp(A_h)$")
        ax.set_xticks(x, [label for _, label in VARIANTS])
        ax.set_xlabel("Training transformation")
        ax.set_xlim(-0.58, 2.58)
        style_axis(ax)

    order_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=6,
            alpha=0.75,
            label=label,
        )
        for _, label, color in ERROR_FIELDS
    ]
    axes[0].legend(
        handles=order_handles,
        title="Evaluated polynomial",
        title_fontsize=8.5,
        loc="upper left",
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.4,
    )

    ax = axes[2]
    for variant_index, (directory, _) in enumerate(VARIANTS):
        for pathway_index, pathway in enumerate(PATHWAYS):
            offset = -0.18 if pathway_index == 0 else 0.18
            add_box(
                ax,
                values(rows_by_variant[directory], pathway, "generator_fro_normalized"),
                x[variant_index] + offset,
                PATHWAY_COLORS[pathway],
                log_axis=False,
            )
    ax.set_title("(c) Learned generator magnitude", loc="left", fontweight="semibold", pad=7)
    ax.set_ylabel(r"$\|A_h\|_F/\sqrt{d_h}$")
    ax.set_xticks(x, [label for _, label in VARIANTS])
    ax.set_xlabel("Training transformation")
    ax.set_xlim(-0.58, 2.58)
    ax.set_ylim(0.0, 2.35)
    style_axis(ax)
    pathway_handles = [
        Line2D(
            [0],
            [0],
            color=PATHWAY_COLORS[pathway],
            linewidth=6,
            alpha=0.75,
            label=PATHWAY_LABELS[pathway],
        )
        for pathway in PATHWAYS
    ]
    ax.legend(
        handles=pathway_handles,
        loc="upper left",
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.4,
    )

    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.20, top=0.93, wspace=0.28)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=260, bbox_inches="tight")
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
