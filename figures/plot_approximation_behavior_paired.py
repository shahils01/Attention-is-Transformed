#!/usr/bin/env python3
"""Paired visualization of GT-MHA polynomial approximation errors."""

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
OUTPUT_STEM = HERE / "transformation_approximation_paired"

VARIANTS = (
    ("residual_step250000", "First-order", "#7B2CBF", "o"),
    ("quadratic_step250000", "Second-order", "#0072B2", "s"),
    ("exact_step250000", "Exact", "#D55E00", "^"),
)
PATHWAYS = ("query_key", "value")
PATHWAY_LABELS = {"query_key": "Query–key", "value": "Value"}
PATHWAY_COLORS = {"query_key": "#7B2CBF", "value": "#00796B"}


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
                    "first_error": float(row["first_order_relative_error"]),
                    "second_error": float(row["second_order_relative_error"]),
                    "magnitude": float(row["generator_fro_normalized"]),
                }
            )
    return rows


def pathway_values(
    rows: list[dict[str, float | int | str]], pathway: str, field: str
) -> np.ndarray:
    result = np.asarray(
        [float(row[field]) for row in rows if row["pathway"] == pathway],
        dtype=float,
    )
    if result.size != 192:
        raise RuntimeError(f"Expected 192 values for {pathway}/{field}; found {result.size}")
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#D8DCE2", linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.tick_params(colors="#444444", labelsize=8.5)


def main() -> None:
    rows_by_variant = {directory: read_rows(directory) for directory, *_ in VARIANTS}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.15, 3.65), constrained_layout=False)
    limits = (2.5e-5, 10.0)
    diagonal = np.geomspace(*limits, 300)

    for ax, pathway, title in zip(
        axes[:2],
        PATHWAYS,
        ("(a) Query–key pathway", "(b) Value pathway"),
    ):
        ax.plot(
            diagonal,
            diagonal,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )
        ax.text(
            2.8,
            3.8,
            r"equal error",
            rotation=42,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#555555",
        )
        for directory, label, color, marker in VARIANTS:
            x = pathway_values(rows_by_variant[directory], pathway, "first_error")
            y = pathway_values(rows_by_variant[directory], pathway, "second_error")
            ax.scatter(
                x,
                y,
                s=13,
                marker=marker,
                facecolor=color,
                edgecolor="none",
                alpha=0.30,
                zorder=2,
            )
            median_x = float(np.median(x))
            median_y = float(np.median(y))
            ax.scatter(
                [median_x],
                [median_y],
                s=64,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=1.1,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=14))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=14))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left", fontweight="semibold", pad=7)
        ax.set_xlabel(r"First-order relative error to $\exp(A_h)$")
        ax.set_ylabel(r"Second-order relative error to $\exp(A_h)$")
        style_axis(ax)

    variant_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor="none",
            label=f"{label} trained",
        )
        for _, label, color, marker in VARIANTS
    ]
    axes[0].legend(
        handles=variant_handles,
        loc="lower right",
        frameon=False,
        borderaxespad=0.4,
    )

    ax = axes[2]
    x_centers = np.arange(len(VARIANTS), dtype=float)
    for index, ((directory, label, _, _), x_center) in enumerate(zip(VARIANTS, x_centers)):
        for pathway_index, pathway in enumerate(PATHWAYS):
            offset = -0.14 if pathway_index == 0 else 0.14
            data = pathway_values(rows_by_variant[directory], pathway, "magnitude")
            q05, q25, median, q75, q95 = np.quantile(data, [0.05, 0.25, 0.5, 0.75, 0.95])
            color = PATHWAY_COLORS[pathway]
            x = x_center + offset
            ax.vlines(x, q05, q95, color=color, linewidth=1.3, alpha=0.8)
            ax.vlines(x, q25, q75, color=color, linewidth=5.0, alpha=0.72)
            ax.scatter(
                [x],
                [median],
                s=36,
                marker="D" if pathway == "query_key" else "s",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
            ax.text(
                x,
                q95 + 0.065,
                f"{median:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="semibold",
                color="#333333",
            )
    ax.set_title("(c) Learned generator magnitude", loc="left", fontweight="semibold", pad=7)
    ax.set_xticks(x_centers, [f"{label}\ntrained" for _, label, _, _ in VARIANTS])
    ax.set_xlabel("Training transformation")
    ax.set_ylabel(r"$\|A_h\|_F/\sqrt{d_h}$")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0.0, 2.35)
    ax.grid(axis="y", color="#D8DCE2", linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.tick_params(colors="#444444", labelsize=8.5)
    pathway_handles = [
        Line2D(
            [0],
            [0],
            color=PATHWAY_COLORS[pathway],
            linewidth=5,
            marker="D" if pathway == "query_key" else "s",
            markerfacecolor="white",
            markeredgecolor=PATHWAY_COLORS[pathway],
            label=PATHWAY_LABELS[pathway],
        )
        for pathway in PATHWAYS
    ]
    ax.legend(handles=pathway_handles, loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.18, top=0.93, wspace=0.32)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=260, bbox_inches="tight")
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
