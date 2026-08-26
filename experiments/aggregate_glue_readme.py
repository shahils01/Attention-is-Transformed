#!/usr/bin/env python3
"""Aggregate the selected three-seed GLUE runs for the HF README."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


METHODS = ("mha", "gqa", "collaborative", "gt_residual")
METHOD_LABELS = {
    "mha": "MHA",
    "gqa": "GQA",
    "collaborative": "Collaborative MHA",
    "gt_residual": "GT-MHA residual",
}
TASKS = ("mnli", "qnli", "sst2", "cola", "mrpc", "stsb", "qqp", "rte", "wnli")
TASK_LABELS = {
    "mnli": "MNLI-m (accuracy)",
    "qnli": "QNLI (accuracy)",
    "sst2": "SST-2 (accuracy)",
    "cola": "CoLA (MCC)",
    "mrpc": "MRPC (F1/accuracy average)",
    "stsb": "STS-B (Pearson/Spearman average)",
    "qqp": "QQP (F1/accuracy average)",
    "rte": "RTE (accuracy)",
    "wnli": "WNLI (accuracy)",
}
EIGHT_TASKS = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte")
RUN_RE = re.compile(
    r"_glue_(mnli|qnli|sst2|cola|mrpc|stsb|qqp|rte|wnli)_ftseed(42|43|44)_4ep_(.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arai3", type=Path, required=True)
    parser.add_argument("--anayak2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def method_for(run_name: str, source: str) -> str | None:
    if run_name.startswith("bert_base_mha_"):
        return "mha"
    if run_name.startswith("bert_base_gqa_"):
        return "gqa"
    if run_name.startswith("bert_base_collaborative_"):
        return "collaborative"
    if run_name.startswith("bert_base_gt_mha_residual_"):
        return "gt_residual"
    if run_name.startswith("checkpoint-100000_glue_"):
        return "mha" if source == "arai3" else "gqa"
    return None


def version_priority(suffix: str) -> tuple[int, int]:
    recovery = 1 if "recovery" in suffix else 0
    match = re.search(r"(?:^|_)v(\d+)$", suffix)
    version = int(match.group(1)) if match else 0
    return recovery, version


def metric_name(task: str) -> str:
    if task == "cola":
        return "eval_matthews_correlation"
    if task in ("mrpc", "stsb", "qqp"):
        return "eval_combined_score"
    return "eval_accuracy"


def best_validation(task: str, record: dict) -> tuple[float, float]:
    metric = metric_name(task)
    evaluations = [
        row
        for row in record.get("log_history", [])
        if metric in row and "eval_loss" in row
    ]
    if not evaluations:
        metrics = record["metrics"]
        return float(metrics[metric]) * 100, float(metrics["eval_loss"])
    # Select the epoch by the task's primary validation metric. If the metric
    # ties, prefer the lower validation loss. Report that same epoch's loss.
    best = max(
        evaluations,
        key=lambda row: (float(row[metric]), -float(row["eval_loss"])),
    )
    return float(best[metric]) * 100, float(best["eval_loss"])


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def formatted(values: list[float], digits: int) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def load_selected(arai3: Path, anayak2: Path):
    selected = {}
    for source, path in (("arai3", arai3), ("anayak2", anayak2)):
        for record in json.loads(path.read_text()):
            run_name = record["run_name"]
            match = RUN_RE.search(run_name)
            method = method_for(run_name, source)
            if match is None or method is None:
                continue
            task, seed_text, suffix = match.groups()
            key = method, task, int(seed_text)
            candidate = (version_priority(suffix), record)
            if key not in selected or candidate[0] > selected[key][0]:
                selected[key] = candidate
    return {key: candidate[1] for key, candidate in selected.items()}


def main() -> None:
    args = parse_args()
    selected = load_selected(args.arai3, args.anayak2)
    missing = [
        (method, task, seed)
        for method in METHODS
        for task in TASKS
        for seed in (42, 43, 44)
        if (method, task, seed) not in selected
    ]

    scores = {}
    losses = {}
    run_names = {}
    for key, record in selected.items():
        method, task, seed = key
        scores[key], losses[key] = best_validation(task, record)
        run_names[key] = record["run_name"]

    summary = {"missing": missing, "scores": {}, "losses": {}, "runs": run_names}
    for method in METHODS:
        summary["scores"][method] = {}
        summary["losses"][method] = {}
        for task in TASKS:
            values = [scores[(method, task, seed)] for seed in (42, 43, 44) if (method, task, seed) in scores]
            loss_values = [losses[(method, task, seed)] for seed in (42, 43, 44) if (method, task, seed) in losses]
            summary["scores"][method][task] = values
            summary["losses"][method][task] = loss_values

    if not missing:
        for method in METHODS:
            per_seed_8 = [
                statistics.mean(scores[(method, task, seed)] for task in EIGHT_TASKS)
                for seed in (42, 43, 44)
            ]
            per_seed_9 = [
                statistics.mean(scores[(method, task, seed)] for task in TASKS)
                for seed in (42, 43, 44)
            ]
            summary["scores"][method]["average_8"] = per_seed_8
            summary["scores"][method]["average_9"] = per_seed_9

        score_rows = list(TASKS) + ["average_8", "average_9"]
        print("| Task | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
        print("| --- | " + " | ".join("---:" for _ in METHODS) + " |")
        for task in score_rows:
            values_by_method = {m: summary["scores"][m][task] for m in METHODS}
            means = {m: statistics.mean(v) for m, v in values_by_method.items()}
            best = max(means, key=means.get)
            label = TASK_LABELS.get(task, "8-task GLUE average (excludes WNLI)" if task == "average_8" else "9-task macro average")
            cells = []
            for method in METHODS:
                cell = formatted(values_by_method[method], 2)
                cells.append(f"**{cell}**" if method == best else cell)
            print("| " + label + " | " + " | ".join(cells) + " |")

        print("\n| Task | " + " | ".join(METHOD_LABELS[m] + " eval loss" for m in METHODS) + " |")
        print("| --- | " + " | ".join("---:" for _ in METHODS) + " |")
        for task in TASKS:
            values_by_method = {m: summary["losses"][m][task] for m in METHODS}
            means = {m: statistics.mean(v) for m, v in values_by_method.items()}
            best = min(means, key=means.get)
            cells = []
            for method in METHODS:
                cell = formatted(values_by_method[method], 4)
                cells.append(f"**{cell}**" if method == best else cell)
            print("| " + TASK_LABELS[task].split(" (")[0] + " | " + " | ".join(cells) + " |")

    serializable_runs = {"|".join(map(str, key)): value for key, value in run_names.items()}
    summary["runs"] = serializable_runs
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if missing:
        print("Missing runs:")
        for method, task, seed in missing:
            print(f"- {method} {task} seed {seed}")


if __name__ == "__main__":
    main()
