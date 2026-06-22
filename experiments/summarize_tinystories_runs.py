from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = (0.27, 0.265, 0.26)
GIB = float(1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TinyStories metrics.jsonl runs on fair runtime axes."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories containing metrics.jsonl, or direct metrics JSONL files.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=float,
        default=None,
        help="Validation-loss threshold for time-to-target columns. Repeatable.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the summary table as CSV.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional path to write the summary table as Markdown.",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=None,
        help="Optional directory for validation-loss comparison plots.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL row") from exc
            if isinstance(row, dict) and "step" in row:
                rows.append(row)
    return rows


def resolve_run(path: Path) -> tuple[str, Path, dict[str, Any]]:
    if path.is_dir():
        metrics_path = path / "metrics.jsonl"
        report = read_json(path / "final_report.json")
        name = path.name
    else:
        metrics_path = path
        report = read_json(path.with_name("final_report.json"))
        name = path.parent.name if path.name == "metrics.jsonl" else path.stem
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics JSONL: {metrics_path}")
    return name, metrics_path, report


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def infer_tokens_per_step(rows: list[dict[str, Any]], report: dict[str, Any]) -> float | None:
    for row in reversed(rows):
        value = as_float(row.get("tokens_per_step"))
        if value is not None:
            return value
    value = as_float(report.get("tokens_per_step"))
    if value is not None:
        return value
    return None


def normalize_rows(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    tokens_per_step = infer_tokens_per_step(rows, report)
    world_size = as_float(report.get("world_size"))
    normalized = []
    for row in rows:
        item = dict(row)
        step = as_float(item.get("step"))
        if item.get("tokens_seen") is None and tokens_per_step is not None and step is not None:
            item["tokens_seen"] = step * tokens_per_step
        if item.get("tokens_per_second_per_gpu") is None:
            tokens_per_second = as_float(item.get("tokens_per_second"))
            row_world_size = as_float(item.get("world_size")) or world_size
            if tokens_per_second is not None and row_world_size is not None and row_world_size > 0:
                item["tokens_per_second_per_gpu"] = tokens_per_second / row_world_size
        if item.get("peak_memory_gib") is None:
            peak_bytes = as_float(item.get("peak_memory_bytes"))
            if peak_bytes is not None:
                item["peak_memory_gib"] = peak_bytes / GIB
        normalized.append(item)
    return normalized


def first_threshold_hit(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, float] | None:
    validation_rows = [
        row
        for row in rows
        if as_float(row.get("validation_loss")) is not None
        and as_float(row.get("validation_loss")) <= threshold
    ]
    if not validation_rows:
        return None
    row = min(validation_rows, key=lambda item: as_float(item.get("step")) or math.inf)
    return {
        "step": as_float(row.get("step")) or math.nan,
        "tokens_seen": as_float(row.get("tokens_seen")) or math.nan,
        "gpu_hours": as_float(row.get("gpu_hours")) or math.nan,
        "validation_loss": as_float(row.get("validation_loss")) or math.nan,
    }


def summarize_run(
    name: str,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    thresholds: list[float],
) -> dict[str, Any]:
    rows = normalize_rows(rows, report)
    if not rows:
        raise SystemExit(f"{name}: no step rows found")
    validation_rows = [row for row in rows if as_float(row.get("validation_loss")) is not None]
    last = max(rows, key=lambda row: as_float(row.get("step")) or -math.inf)
    last_validation = validation_rows[-1] if validation_rows else {}
    best_validation = (
        min(validation_rows, key=lambda row: as_float(row.get("validation_loss")) or math.inf)
        if validation_rows
        else {}
    )
    throughput = [
        value
        for value in (as_float(row.get("tokens_per_second_per_gpu")) for row in rows)
        if value is not None
    ]
    peak_memory = [
        value
        for value in (as_float(row.get("peak_memory_gib")) for row in rows)
        if value is not None
    ]
    summary: dict[str, Any] = {
        "run": name,
        "world_size": as_float(last.get("world_size")) or as_float(report.get("world_size")),
        "final_step": as_float(last.get("step")),
        "tokens_seen": as_float(last.get("tokens_seen")) or as_float(report.get("tokens_seen")),
        "gpu_hours": as_float(last.get("gpu_hours")) or as_float(report.get("gpu_hours")),
        "last_loss": as_float(last.get("loss")),
        "last_validation_loss": as_float(last_validation.get("validation_loss")),
        "best_validation_loss": as_float(best_validation.get("validation_loss")),
        "best_validation_step": as_float(best_validation.get("step")),
        "peak_memory_gib": max(peak_memory) if peak_memory else as_float(report.get("peak_memory_gib")),
        "tokens_per_second_per_gpu_median": (
            statistics.median(throughput) if throughput else None
        ),
    }
    for threshold in thresholds:
        hit = first_threshold_hit(rows, threshold)
        suffix = threshold_suffix(threshold)
        summary[f"step_to_val_{suffix}"] = hit["step"] if hit else None
        summary[f"tokens_to_val_{suffix}"] = hit["tokens_seen"] if hit else None
        summary[f"gpu_hours_to_val_{suffix}"] = hit["gpu_hours"] if hit else None
    return summary


def threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def format_number(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if abs(number) >= 1e9:
        return f"{number / 1e9:.{digits}g}B"
    if abs(number) >= 1e6:
        return f"{number / 1e6:.{digits}g}M"
    if abs(number) >= 1e3:
        return f"{number / 1e3:.{digits}g}k"
    return f"{number:.{digits}g}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(str(value) if isinstance(value, str) else format_number(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_plots(run_payloads: list[tuple[str, list[dict[str, Any]]]], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot_dir") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    axes = [
        ("step", "Step"),
        ("tokens_seen", "Tokens seen"),
        ("gpu_hours", "GPU-hours"),
    ]
    for axis_key, axis_label in axes:
        any_series = False
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
        for name, rows in run_payloads:
            points = [
                (as_float(row.get(axis_key)), as_float(row.get("validation_loss")))
                for row in rows
            ]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if not points:
                continue
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=name)
            any_series = True
        if not any_series:
            plt.close(fig)
            continue
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Validation loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"validation_loss_vs_{axis_key}.png")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    thresholds = args.thresholds or list(DEFAULT_THRESHOLDS)
    summaries = []
    run_payloads = []
    for run_path in args.runs:
        name, metrics_path, report = resolve_run(run_path)
        rows = normalize_rows(read_jsonl(metrics_path), report)
        summaries.append(summarize_run(name, rows, report, thresholds))
        run_payloads.append((name, rows))

    threshold_columns = []
    for threshold in thresholds:
        suffix = threshold_suffix(threshold)
        threshold_columns.extend(
            [
                f"step_to_val_{suffix}",
                f"tokens_to_val_{suffix}",
                f"gpu_hours_to_val_{suffix}",
            ]
        )
    columns = [
        "run",
        "world_size",
        "final_step",
        "tokens_seen",
        "gpu_hours",
        "last_validation_loss",
        "best_validation_loss",
        "peak_memory_gib",
        "tokens_per_second_per_gpu_median",
        *threshold_columns,
    ]
    table = markdown_table(summaries, columns)
    print(table)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n", encoding="utf-8")
    if args.csv is not None:
        write_csv(args.csv, summaries, columns)
    if args.plot_dir is not None:
        write_plots(run_payloads, args.plot_dir)


if __name__ == "__main__":
    main()
