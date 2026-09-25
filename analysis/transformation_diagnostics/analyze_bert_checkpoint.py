#!/usr/bin/env python3
"""Extract per-head GT-MHA diagnostics from a BERT masked-LM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from analyze_checkpoint import matrix_rows, sha256, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-parameter-count", type=int, default=96_128_826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.checkpoint_dir / "model.safetensors"
    config_path = args.checkpoint_dir / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint requires model.safetensors and config.json: {args.checkpoint_dir}"
        )

    observed_sha256 = sha256(weights_path)
    if observed_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: {observed_sha256} != {args.expected_sha256}"
        )

    sys.path.insert(0, str(args.repo_root / "src"))
    from lgma.attention import LieGeneratedMetricAttention  # noqa: PLC0415
    from lgma.bert import replace_bert_self_attention  # noqa: PLC0415
    from safetensors.torch import load_model  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForMaskedLM  # noqa: PLC0415

    config = AutoConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if config.num_hidden_layers != 12 or config.num_attention_heads != 12:
        raise RuntimeError(
            "Expected a 12-layer, 12-head BERT checkpoint; found "
            f"{config.num_hidden_layers} layers and {config.num_attention_heads} heads"
        )
    model = AutoModelForMaskedLM.from_config(config)
    audit = replace_bert_self_attention(
        model,
        attention_type="gt_mha_residual",
        num_base_heads=4,
        num_generators=8,
        generator_mixing="softmax",
        use_sdpa=False,
        fuse_base_qkv=True,
        initialize_from_mha=False,
        enforce_paper_gt_mha=True,
    )
    load_model(model, str(weights_path), strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != args.expected_parameter_count:
        raise RuntimeError(
            f"Parameter-count mismatch: {parameter_count} != {args.expected_parameter_count}"
        )
    model.float().eval()

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LieGeneratedMetricAttention)
    ]
    if len(modules) != config.num_hidden_layers:
        raise RuntimeError(
            f"Found {len(modules)} GT-MHA modules; expected {config.num_hidden_layers}"
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
        "model_name": "BERT-base GT-MHA residual C=4 G=8 H=12",
        "pretraining_seed": args.seed,
        "checkpoint_step": 100_000,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_training_precision": "BF16 (per training protocol)",
        "diagnostic_precision": "float64",
        "parameter_count": parameter_count,
        "model_config": config.to_dict(),
        "gt_mha_construction": {
            "attention_type": "gt_mha_residual",
            "num_base_heads": 4,
            "num_generators": 8,
            "num_attention_heads": 12,
            "fuse_base_qkv": True,
        },
        "replacement_audit": audit,
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
