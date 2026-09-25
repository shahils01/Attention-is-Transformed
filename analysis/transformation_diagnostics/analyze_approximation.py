#!/usr/bin/env python3
"""Compare first- and second-order maps with exp(A) on learned GT-MHA generators.

For every attention layer and head, this script reconstructs the learned
query--key and value generators A_h and evaluates all three maps on that same
matrix:

    T1(A) = I + A
    T2(A) = I + A + A^2 / 2
    Texp(A) = exp(A)

The resulting discrepancies are counterfactual diagnostics of the learned
generator regime.  They should not be interpreted as a controlled comparison
of separately trained checkpoints.
"""

from __future__ import annotations

import argparse
import csv
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

from analyze_checkpoint import quantile, sha256, tokenizer_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
        kwargs["mmap"] = True
    return torch.load(str(path), **kwargs)


def value_head_generators(module: Any) -> torch.Tensor:
    """Reproduce the value-path A_h used by compute_value_transforms()."""
    if module.generator_type == "diagonal":
        diagonal = torch.einsum(
            "hm,md->hd",
            module.value_theta_weights(),
            module._maybe_normalize_generators(module.value_generators),
        )
        return torch.diag_embed(module.value_beta * diagonal)

    generators = module._dense_value_generators()
    head_generators = torch.einsum(
        "hm,mde->hde", module.value_theta_weights(), generators
    )
    head_generators = module.value_beta * head_generators
    if module.generator_type == "symmetric":
        head_generators = 0.5 * (
            head_generators + head_generators.transpose(-1, -2)
        )
    return head_generators


def approximation_rows(
    generator_batch: torch.Tensor,
    *,
    layer: int,
    module_name: str,
    pathway: str,
    trained_mapping: str,
) -> list[dict[str, Any]]:
    generators = generator_batch.detach().to(device="cpu", dtype=torch.float64)
    dimension = generators.shape[-1]
    eye = torch.eye(dimension, dtype=torch.float64)
    rows: list[dict[str, Any]] = []

    for head, generator in enumerate(generators):
        first = eye + generator
        second = first + 0.5 * (generator @ generator)
        exact = torch.linalg.matrix_exp(generator)
        exact_norm = torch.linalg.matrix_norm(exact, ord="fro").item()
        first_error = torch.linalg.matrix_norm(first - exact, ord="fro").item()
        second_error = torch.linalg.matrix_norm(second - exact, ord="fro").item()
        generator_fro = torch.linalg.matrix_norm(generator, ord="fro").item()
        generator_spectral = torch.linalg.matrix_norm(generator, ord=2).item()
        determinant_sign = torch.linalg.slogdet(first).sign.item()

        rows.append(
            {
                "layer": layer,
                "module": module_name,
                "head": head,
                "pathway": pathway,
                "trained_mapping": trained_mapping,
                "dimension": dimension,
                "generator_fro_normalized": generator_fro / math.sqrt(dimension),
                "generator_spectral_norm": generator_spectral,
                "first_order_absolute_error": first_error,
                "first_order_relative_error": first_error / exact_norm,
                "second_order_absolute_error": second_error,
                "second_order_relative_error": second_error / exact_norm,
                "second_to_first_error_ratio": (
                    second_error / first_error if first_error > 0.0 else 0.0
                ),
                "first_order_determinant_sign": determinant_sign,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "generator_fro_normalized",
        "generator_spectral_norm",
        "first_order_absolute_error",
        "first_order_relative_error",
        "second_order_absolute_error",
        "second_order_relative_error",
        "second_to_first_error_ratio",
    )
    summary: dict[str, Any] = {"count": len(rows)}
    for field in fields:
        finite = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
        summary[field] = {
            "median": quantile(finite, 0.5),
            "q25": quantile(finite, 0.25),
            "q75": quantile(finite, 0.75),
            "min": min(finite),
            "max": max(finite),
        }
    signs = [float(row["first_order_determinant_sign"]) for row in rows]
    summary["first_order_determinant_sign_counts"] = {
        "negative": sum(sign < 0 for sign in signs),
        "zero": sum(sign == 0 for sign in signs),
        "positive": sum(sign > 0 for sign in signs),
    }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo_root / "src"))

    from lgma.attention import LieGeneratedMetricAttention  # noqa: PLC0415
    from lgma.transformer import TinyTransformerLM  # noqa: PLC0415

    observed_sha256 = sha256(args.checkpoint)
    if args.expected_sha256 and observed_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: {observed_sha256} != {args.expected_sha256}"
        )

    checkpoint = load_checkpoint(args.checkpoint)
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
                approximation_rows(
                    module.compute_head_generators(),
                    layer=layer,
                    module_name=name,
                    pathway="query_key",
                    trained_mapping=module.metric_mode,
                )
            )
            rows.extend(
                approximation_rows(
                    value_head_generators(module),
                    layer=layer,
                    module_name=name,
                    pathway="value",
                    trained_mapping=module.value_transform_mode,
                )
            )

    csv_path = args.output_dir / "per_head_approximation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int | str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pathway"], "all")].append(row)
        grouped[(row["pathway"], int(row["layer"]))].append(row)

    summaries: dict[str, Any] = {}
    for (pathway, layer), group_rows in grouped.items():
        summaries.setdefault(pathway, {})[str(layer)] = summarize(group_rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_step": checkpoint.get("step"),
        "diagnostic_precision": "float64",
        "model_config": config,
        "attention_module_count": len(modules),
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
