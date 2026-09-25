#!/usr/bin/env python3
"""Extract per-head GT-MHA transformation diagnostics from a saved checkpoint.

The script strictly reconstructs the trained model, evaluates the deployed
query--key and value transformations, and writes both head-level measurements
and robust summaries.  It intentionally performs all linear algebra in float64
on CPU; the checkpoint's training precision is recorded but is not appropriate
for stable SVD and determinant diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--model-name", default="GT-MHA residual")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
        kwargs["mmap"] = True
    return torch.load(str(path), **kwargs)


def tokenizer_size(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("stoi"), dict):
        return len(payload["stoi"])
    if isinstance(payload, dict) and isinstance(payload.get("vocab"), list):
        return len(payload["vocab"])
    if isinstance(payload, dict) and isinstance(payload.get("vocabulary"), list):
        return len(payload["vocabulary"])
    if isinstance(payload, list):
        return len(payload)
    raise ValueError(f"Cannot infer vocabulary size from {path}")


def matrix_rows(
    matrix_batch: torch.Tensor,
    *,
    layer: int,
    module_name: str,
    pathway: str,
    metric_mode: str,
) -> list[dict[str, Any]]:
    matrices = matrix_batch.detach().to(device="cpu", dtype=torch.float64)
    dimension = matrices.shape[-1]
    eye = torch.eye(dimension, dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for head, matrix in enumerate(matrices):
        singular_values = torch.linalg.svdvals(matrix)
        sigma_max = singular_values.max().item()
        sigma_min = singular_values.min().item()
        condition = math.inf if sigma_min == 0.0 else sigma_max / sigma_min
        determinant_sign, logabsdet = torch.linalg.slogdet(matrix)
        rows.append(
            {
                "layer": layer,
                "module": module_name,
                "head": head,
                "pathway": pathway,
                "mapping": metric_mode,
                "dimension": dimension,
                "distance_from_identity": (
                    torch.linalg.matrix_norm(matrix - eye, ord="fro").item()
                    / math.sqrt(dimension)
                ),
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "condition_number": condition,
                "log10_condition_number": (
                    math.inf if not math.isfinite(condition) else math.log10(condition)
                ),
                "determinant_sign": determinant_sign.item(),
                "log_abs_determinant": logabsdet.item(),
            }
        )
    return rows


def quantile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return torch.quantile(tensor, q).item()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "distance_from_identity",
        "sigma_min",
        "sigma_max",
        "condition_number",
        "log10_condition_number",
        "log_abs_determinant",
    )
    summary: dict[str, Any] = {"count": len(rows)}
    for field in fields:
        finite = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
        summary[field] = {
            "finite_count": len(finite),
            "median": quantile(finite, 0.5) if finite else None,
            "q25": quantile(finite, 0.25) if finite else None,
            "q75": quantile(finite, 0.75) if finite else None,
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
        }
    signs = [float(row["determinant_sign"]) for row in rows]
    summary["determinant_sign_counts"] = {
        "negative": sum(sign < 0 for sign in signs),
        "zero": sum(sign == 0 for sign in signs),
        "positive": sum(sign > 0 for sign in signs),
    }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_src = args.repo_root / "src"
    sys.path.insert(0, str(repo_src))
    from lgma.attention import LieGeneratedMetricAttention  # noqa: PLC0415
    from lgma.transformer import TinyTransformerLM  # noqa: PLC0415

    observed_sha256 = sha256(args.checkpoint)
    if args.expected_sha256 and observed_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: {observed_sha256} != {args.expected_sha256}"
        )

    checkpoint = load_checkpoint(args.checkpoint)
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain model_config and model_state")
    config = dict(checkpoint["model_config"])
    model = TinyTransformerLM(vocab_size=tokenizer_size(args.tokenizer), **config)
    load_result = model.load_state_dict(checkpoint["model_state"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Strict load unexpectedly returned {load_result}")
    model.float().eval()

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LieGeneratedMetricAttention)
    ]
    if len(modules) != config.get("num_layers"):
        raise RuntimeError(
            f"Found {len(modules)} GT-MHA modules; expected {config.get('num_layers')}"
        )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for layer, (name, module) in enumerate(modules, start=1):
            rows.extend(
                matrix_rows(
                    module.compute_metrics(),
                    layer=layer,
                    module_name=name,
                    pathway="query_key",
                    metric_mode=module.metric_mode,
                )
            )
            rows.extend(
                matrix_rows(
                    module.compute_value_transforms(),
                    layer=layer,
                    module_name=name,
                    pathway="value",
                    metric_mode=module.value_transform_mode,
                )
            )

    csv_path = args.output_dir / "per_head_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, int | str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["pathway"], "all")].append(row)
        groups[(row["pathway"], int(row["layer"]))].append(row)

    summaries: dict[str, Any] = {}
    for (pathway, layer), group_rows in groups.items():
        summaries.setdefault(pathway, {})[str(layer)] = summarize(group_rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_training_precision": "BF16 (per manuscript protocol)",
        "diagnostic_precision": "float64",
        "model_config": config,
        "attention_module_count": len(modules),
        "attention_modules": [name for name, _ in modules],
        "row_count": len(rows),
        "summaries": summaries,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
