from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from summarize_tinystories_runs import (
    DEFAULT_THRESHOLDS,
    as_float,
    markdown_table,
    read_json,
    read_jsonl,
    resolve_run,
    summarize_run,
)


RUN_COLUMNS = [
    "run",
    "group",
    "seed",
    "attention_type",
    "parameters",
    "attention_parameters",
    "qkv_parameters",
    "generator_parameters",
    "final_step",
    "tokens_seen",
    "last_validation_loss",
    "best_validation_loss",
    "full_test_loss",
    "full_test_perplexity",
    "tokens_per_second_per_gpu_median",
    "peak_memory_gib",
    "slurm_gpu_hours",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "decode_median_ms_per_token",
    "kv_cache_bytes_per_token_per_layer",
    "measured_kv_cache_bytes",
    "measured_kv_cache_bytes_per_token_per_layer",
]

GROUP_METRICS = [
    "last_validation_loss",
    "best_validation_loss",
    "full_test_loss",
    "full_test_perplexity",
    "tokens_per_second_per_gpu_median",
    "peak_memory_gib",
    "slurm_gpu_hours",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "decode_median_ms_per_token",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TinyStories, Slurm, evaluation, inference, and optional W&B results."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--skip_slurm",
        action="store_true",
        help="Do not invoke sacct; useful when aggregating away from the cluster.",
    )
    parser.add_argument(
        "--fetch_wandb",
        action="store_true",
        help="Fetch state/URL/summary step for manifest entries containing wandb_path.",
    )
    return parser.parse_args()


def resolve_optional_path(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def parse_gpu_count(alloc_tres: str) -> int | None:
    match = re.search(r"(?:gres/gpu(?::[^=,]+)?|gpu)=(\d+)", alloc_tres)
    return int(match.group(1)) if match else None


def query_slurm_jobs(
    job_ids: list[str],
    *,
    fallback_gpus: int | None = None,
) -> dict[str, Any]:
    if not job_ids:
        return {}
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-S",
        "2020-01-01",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,ExitCode",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("sacct is unavailable; run on DeltaAI or use --skip_slurm") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"sacct failed: {exc.stderr.strip()}") from exc

    requested = set(job_ids)
    jobs = []
    for line in completed.stdout.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) < 6 or fields[0] not in requested:
            continue
        elapsed_seconds = int(fields[3]) if fields[3].isdigit() else 0
        gpu_count = parse_gpu_count(fields[4]) or fallback_gpus or 0
        jobs.append(
            {
                "job_id": fields[0],
                "job_name": fields[1],
                "state": fields[2],
                "elapsed_seconds": elapsed_seconds,
                "gpu_count": gpu_count,
                "gpu_hours": elapsed_seconds * gpu_count / 3600.0,
                "alloc_tres": fields[4],
                "exit_code": fields[5],
            }
        )
    return {
        "jobs": jobs,
        "slurm_elapsed_hours": sum(job["elapsed_seconds"] for job in jobs) / 3600.0,
        "slurm_gpu_hours": sum(job["gpu_hours"] for job in jobs),
        "slurm_jobs_recorded": len(jobs),
        "slurm_states": ",".join(f"{job['job_id']}:{job['state']}" for job in jobs),
    }


def read_full_evaluation(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = read_json(path)
    if not payload:
        raise SystemExit(f"missing or empty full evaluation JSON: {path}")
    return {
        "full_test_loss": as_float(payload.get("loss")),
        "full_test_perplexity": as_float(payload.get("perplexity")),
        "full_test_bits_per_character": as_float(payload.get("bits_per_character")),
        "full_test_tokens": as_float(payload.get("evaluated_tokens")),
        "full_test_source": payload.get("source"),
    }


def read_inference_benchmark(
    path: Path | None,
    primary_context_length: int | None,
) -> dict[str, Any]:
    if path is None:
        return {}
    payload = read_json(path)
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        raise SystemExit(f"inference benchmark contains no results: {path}")
    if primary_context_length is None:
        selected = max(results, key=lambda item: int(item["context_length"]))
    else:
        matches = [
            item for item in results if int(item["context_length"]) == primary_context_length
        ]
        if not matches:
            raise SystemExit(
                f"{path}: no inference result for context {primary_context_length}"
            )
        selected = matches[0]
    return {
        "inference_context_length": selected["context_length"],
        "prefill_tokens_per_second": as_float(selected["prefill"].get("tokens_per_second")),
        "prefill_median_ms": as_float(selected["prefill"].get("median_ms")),
        "decode_tokens_per_second": as_float(selected["decode"].get("tokens_per_second")),
        "decode_median_ms_per_token": as_float(
            selected["decode"].get("median_ms_per_token")
        ),
        "decode_mode": selected["decode"].get("mode"),
        "inference_peak_memory_allocated_gib": as_float(
            selected.get("peak_memory_allocated_gib")
        ),
        "inference_peak_memory_reserved_gib": as_float(
            selected.get("peak_memory_reserved_gib")
        ),
        "kv_cache_bytes_per_token_per_layer": as_float(
            selected["attention_accounting"].get("kv_cache_bytes_per_token_per_layer")
        ),
        "measured_kv_cache_bytes": as_float(
            selected["decode"].get("measured_kv_cache_bytes")
        ),
        "measured_kv_cache_bytes_per_token_per_layer": as_float(
            selected["decode"].get("measured_kv_cache_bytes_per_token_per_layer")
        ),
    }


def fetch_wandb_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required with --fetch_wandb") from exc
    run = wandb.Api(timeout=60).run(path)
    return {
        "wandb_path": path,
        "wandb_state": run.state,
        "wandb_url": run.url,
        "wandb_summary_step": as_float(dict(run.summary).get("_step")),
    }


def extract_report_metadata(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("model_config", {})
    args = report.get("args", {})
    accounting = report.get("attention_accounting", {})
    if not isinstance(config, dict):
        config = {}
    if not isinstance(args, dict):
        args = {}
    if not isinstance(accounting, dict):
        accounting = {}
    return {
        "attention_type": config.get("attention_type"),
        "seed": args.get("seed"),
        "parameters": as_float(report.get("parameters")),
        "attention_parameters": as_float(accounting.get("total_parameters")),
        "qkv_parameters": as_float(accounting.get("qkv_parameters")),
        "generator_parameters": as_float(accounting.get("generator_parameters")),
        "kv_cache_bytes_per_token_per_layer": as_float(
            accounting.get("kv_cache_bytes_per_token_per_layer")
        ),
        "attention_score_flops": as_float(accounting.get("attention_score_flops")),
    }


def summarize_manifest_run(
    entry: dict[str, Any],
    manifest_dir: Path,
    *,
    skip_slurm: bool,
    fetch_wandb: bool,
) -> dict[str, Any]:
    name = str(entry["name"])
    run_dir = resolve_optional_path(manifest_dir, entry.get("run_dir"))
    if run_dir is None:
        raise SystemExit(f"{name}: run_dir is required")
    _, metrics_path, report = resolve_run(run_dir)
    rows = read_jsonl(metrics_path)
    summary = summarize_run(name, rows, report, list(DEFAULT_THRESHOLDS))
    summary["group"] = str(entry.get("group", name))
    summary.update(extract_report_metadata(report))
    if entry.get("seed") is not None:
        summary["seed"] = entry["seed"]

    evaluation_path = resolve_optional_path(manifest_dir, entry.get("full_eval_json"))
    summary.update(read_full_evaluation(evaluation_path))
    inference_path = resolve_optional_path(manifest_dir, entry.get("inference_json"))
    summary.update(
        read_inference_benchmark(inference_path, entry.get("primary_context_length"))
    )
    if not skip_slurm:
        summary.update(
            query_slurm_jobs(
                [str(job_id) for job_id in entry.get("slurm_job_ids", [])],
                fallback_gpus=entry.get("gpus_per_job"),
            )
        )
    if fetch_wandb:
        summary.update(fetch_wandb_metadata(entry.get("wandb_path")))
    return summary


def aggregate_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["group"]), []).append(run)
    output = []
    for group, members in sorted(grouped.items()):
        row: dict[str, Any] = {"group": group, "seeds": len(members)}
        for metric in GROUP_METRICS:
            values = [as_float(member.get(metric)) for member in members]
            finite = [value for value in values if value is not None and math.isfinite(value)]
            row[f"{metric}_mean"] = statistics.mean(finite) if finite else None
            row[f"{metric}_std"] = statistics.stdev(finite) if len(finite) > 1 else None
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_outputs(output_dir: Path, runs: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = aggregate_groups(runs)
    group_columns = ["group", "seeds"] + [
        column
        for metric in GROUP_METRICS
        for column in (f"{metric}_mean", f"{metric}_std")
    ]
    (output_dir / "paper_runs.json").write_text(
        json.dumps(runs, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "paper_groups.json").write_text(
        json.dumps(groups, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "paper_runs.csv", runs, RUN_COLUMNS)
    write_csv(output_dir / "paper_groups.csv", groups, group_columns)
    (output_dir / "paper_runs.md").write_text(
        markdown_table(runs, RUN_COLUMNS) + "\n", encoding="utf-8"
    )
    (output_dir / "paper_groups.md").write_text(
        markdown_table(groups, group_columns) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("schema_version") != 1:
        raise SystemExit("manifest schema_version must be 1")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("manifest must contain a non-empty runs list")
    runs = [
        summarize_manifest_run(
            entry,
            args.manifest.parent,
            skip_slurm=args.skip_slurm,
            fetch_wandb=args.fetch_wandb,
        )
        for entry in entries
    ]
    write_outputs(args.output_dir, runs)
    print(markdown_table(runs, RUN_COLUMNS))


if __name__ == "__main__":
    main()
