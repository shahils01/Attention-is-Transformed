#!/usr/bin/env python3
"""Evaluate saved MHA heads on the frozen GT-MHA diversity inputs.

The attention JSD and pre-output cosine distance use the same definitions as
analyze_head_diversity.py.  The latter is retained for protocol consistency,
but MHA heads have independent value bases, so projected contribution distance
is also reported in the common model space.  BERT can additionally run a
held-out single-head ablation diagnostic on a subset of the same inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import analyze_head_diversity as shared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("bert", "tinystories"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-precision", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--bert-validation-file", type=Path)
    parser.add_argument("--bert-examples", type=int, default=1000)
    parser.add_argument("--bert-seed", type=int)
    parser.add_argument("--ablation-examples", type=int, default=0)
    parser.add_argument("--tinystories-data-dir", type=Path)
    parser.add_argument("--causal-query-start", type=int, default=16)
    return parser.parse_args()


def split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    batch, time, width = tensor.shape
    if width % heads:
        raise ValueError("Projection width is not divisible by head count")
    return tensor.reshape(batch, time, heads, width // heads).transpose(1, 2)


def projected_contributions(outputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Map each head output through its block of the shared output projection."""
    heads = outputs.shape[1]
    head_width = outputs.shape[-1]
    if weight.shape[1] != heads * head_width:
        raise ValueError("Output projection shape does not match head outputs")
    blocks = weight.float().reshape(weight.shape[0], heads, head_width)
    return torch.einsum("bhtd,ohd->bhto", outputs.float(), blocks)


def nearest_neighbor_mean(matrix: torch.Tensor) -> torch.Tensor:
    heads = matrix.shape[-1]
    diagonal = torch.eye(heads, dtype=torch.bool, device=matrix.device)
    return matrix.masked_fill(diagonal, float("inf")).min(dim=-1).values.mean(dim=-1)


class MhaCollector:
    def __init__(
        self,
        attention_modules: list[torch.nn.Module],
        output_projections: list[torch.nn.Linear],
        *,
        task: str,
        query_start: int,
    ) -> None:
        if len(attention_modules) != len(output_projections):
            raise ValueError("One output projection is required for each attention layer")
        self.task = task
        self.query_start = query_start
        self.current_ids: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.matrix_sums: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.matrix_counts: list[int] = []
        self.max_context_error: list[float] = [0.0] * len(attention_modules)
        self.expected_context: dict[int, torch.Tensor] = {}
        self.handles = []
        self.metrics = (
            "attention_jsd",
            "output_cosine_distance",
            "projected_contribution_cosine_distance",
        )
        for layer, (module, projection) in enumerate(
            zip(attention_modules, output_projections), start=1
        ):
            heads = (
                module.num_attention_heads if task == "bert" else module.num_heads
            )
            for metric in self.metrics:
                self.matrix_sums[metric].append(
                    torch.zeros(heads, heads, dtype=torch.float64)
                )
            self.matrix_counts.append(0)
            self.handles.append(module.register_forward_pre_hook(self._capture(layer, module, projection)))
            self.handles.append(projection.register_forward_pre_hook(self._verify(layer)))

    def _capture(self, layer: int, module: torch.nn.Module, projection: torch.nn.Linear):
        def hook(_module, args):
            x = args[0]
            heads = module.num_attention_heads if self.task == "bert" else module.num_heads
            if self.task == "bert":
                q = split_heads(module.query(x), heads)
                k = split_heads(module.key(x), heads)
                v = split_heads(module.value(x), heads)
                causal = False
            else:
                q = split_heads(module.q_proj(x), heads)
                k = split_heads(module.k_proj(x), heads)
                v = split_heads(module.v_proj(x), heads)
                causal = bool(module.causal)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            if causal:
                time = scores.shape[-1]
                mask = torch.ones(time, time, dtype=torch.bool, device=scores.device).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            outputs = torch.matmul(attention, v)
            self.expected_context[layer] = outputs.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            matrices = {
                "attention_jsd": shared.attention_jsd_matrix(attention, self.query_start),
                "output_cosine_distance": shared.output_cosine_distance_matrix(outputs, self.query_start),
                "projected_contribution_cosine_distance": shared.output_cosine_distance_matrix(
                    projected_contributions(outputs, projection.weight), self.query_start
                ),
            }
            upper = torch.triu(torch.ones(heads, heads, dtype=torch.bool, device=x.device), diagonal=1)
            if len(self.current_ids) != x.shape[0]:
                raise RuntimeError("Batch identifiers do not match attention batch")
            for index, example_id in enumerate(self.current_ids):
                row: dict[str, Any] = {"example_id": example_id, "layer": layer}
                for metric, matrix in matrices.items():
                    row[f"{metric}_overall"] = float(matrix[index][upper].mean().cpu())
                row["attention_jsd_nearest_neighbor"] = float(
                    nearest_neighbor_mean(matrices["attention_jsd"][index]).cpu()
                )
                self.rows.append(row)
            for metric, matrix in matrices.items():
                self.matrix_sums[metric][layer - 1] += matrix.double().sum(0).cpu()
            self.matrix_counts[layer - 1] += x.shape[0]
        return hook

    def _verify(self, layer: int):
        def hook(_projection, args):
            expected = self.expected_context.pop(layer)
            actual = args[0]
            error = float((actual.float() - expected.float()).abs().max().cpu())
            self.max_context_error[layer - 1] = max(self.max_context_error[layer - 1], error)
            tolerance = 0.08 if actual.dtype == torch.bfloat16 else 0.001
            if error > tolerance:
                raise RuntimeError(
                    f"Layer {layer}: reconstructed MHA context differs from model by {error:.6g}"
                )
        return hook

    def set_batch(self, ids: list[str]) -> None:
        self.current_ids = ids

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.expected_context:
            raise RuntimeError("Missing output-projection verification")

    def write(self, output_dir: Path, protocol: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "per_example_layer.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        matrices = {
            metric: np.stack(
                [(matrix / count).numpy() for matrix, count in zip(self.matrix_sums[metric], self.matrix_counts)]
            )
            for metric in self.metrics
        }
        np.savez_compressed(output_dir / "mean_pairwise_matrices.npz", **matrices)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol,
            "summary": shared.summarize(self.rows),
            "max_context_reconstruction_error_by_layer": self.max_context_error,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output_dir": str(output_dir), "summary": report["summary"]["all_layers"]}, indent=2), flush=True)


def original_bert_inputs(tokenizer, count: int, validation_path: Path):
    if validation_path.suffix == ".arrow":
        from datasets import Dataset

        texts = Dataset.from_file(str(validation_path))["text"]
    elif validation_path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        texts = parquet.read_table(validation_path, columns=["text"])["text"].to_pylist()
    else:
        raise ValueError(f"Unsupported BERT validation file: {validation_path}")
    originals: list[list[int]] = []
    special_masks: list[list[int]] = []
    for start in range(0, len(texts), 1000):
        encoded = tokenizer(texts[start:start + 1000], return_special_tokens_mask=True)
        joined_ids = sum(encoded["input_ids"], [])
        joined_special = sum(encoded["special_tokens_mask"], [])
        usable = len(joined_ids) // 128 * 128
        for offset in range(0, usable, 128):
            originals.append(joined_ids[offset:offset + 128])
            special_masks.append(joined_special[offset:offset + 128])
            if len(originals) == count:
                return originals, special_masks
    raise RuntimeError(f"Only constructed {len(originals)} BERT validation sequences")


def bert_ablation_labels(originals: list[list[int]], special_masks: list[list[int]]):
    labels = []
    for index, (tokens, special) in enumerate(zip(originals, special_masks)):
        generator = torch.Generator().manual_seed(shared.BERT_MASK_SEED + index)
        probability = torch.full((len(tokens),), 0.15, dtype=torch.float32)
        probability.masked_fill_(torch.tensor(special, dtype=torch.bool), 0.0)
        selected = torch.bernoulli(probability, generator=generator).bool()
        labels.append([token if selected[position] else -100 for position, token in enumerate(tokens)])
    return labels


def evaluate_bert_masked_loss(model, inputs, labels, batch_size: int) -> float:
    total_loss = 0.0
    total_masked = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            ids = torch.tensor(inputs[start:start + batch_size], dtype=torch.long, device=device)
            target = torch.tensor(labels[start:start + batch_size], dtype=torch.long, device=device)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            total_loss += float(F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                ignore_index=-100, reduction="sum",
            ).cpu())
            total_masked += int((target != -100).sum().cpu())
    return total_loss / total_masked


def run_bert_ablation(model, inputs, labels, batch_size: int, output_dir: Path) -> None:
    if not inputs:
        return
    baseline = evaluate_bert_masked_loss(model, inputs, labels, batch_size)
    results = []
    for layer_index, layer in enumerate(model.bert.encoder.layer):
        dense = layer.attention.output.dense
        heads = layer.attention.self.num_attention_heads
        width = dense.in_features // heads
        for head_index in range(heads):
            def zero_head(_module, args, *, start=head_index * width, end=(head_index + 1) * width):
                hidden = args[0].clone()
                hidden[..., start:end] = 0
                return (hidden, *args[1:])
            handle = dense.register_forward_pre_hook(zero_head)
            try:
                loss = evaluate_bert_masked_loss(model, inputs, labels, batch_size)
            finally:
                handle.remove()
            results.append({
                "layer": layer_index + 1, "head": head_index + 1,
                "masked_nll": loss, "delta_masked_nll": loss - baseline,
            })
            print(f"ablation layer={layer_index + 1} head={head_index + 1} delta_nll={loss - baseline:.6f}", flush=True)
    (output_dir / "single_head_ablation.json").write_text(json.dumps({
        "baseline_masked_nll": baseline,
        "examples": len(inputs),
        "masked_tokens": sum(token != -100 for row in labels for token in row),
        "intervention": "zero each head at the attention output projection input; no retraining",
        "heads": results,
    }, indent=2) + "\n", encoding="utf-8")


def run_bert(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    from safetensors.torch import load_model

    if args.bert_validation_file is None:
        raise ValueError("--bert-validation-file is required")
    weights = args.checkpoint / "model.safetensors"
    checkpoint_sha = shared.sha256(weights)
    if checkpoint_sha != args.checkpoint_sha256:
        raise RuntimeError("BERT checkpoint SHA-256 mismatch")
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForMaskedLM.from_config(config)
    load_model(model, str(weights), strict=True)
    model = model.cuda().eval()
    attention_modules = [layer.attention.self for layer in model.bert.encoder.layer]
    output_projections = [layer.attention.output.dense for layer in model.bert.encoder.layer]
    if len(attention_modules) != config.num_hidden_layers:
        raise RuntimeError("Unexpected BERT layer count")
    masked_inputs = shared.build_bert_validation_inputs(
        tokenizer, args.bert_examples, args.bert_validation_file
    )
    collector = MhaCollector(attention_modules, output_projections, task="bert", query_start=0)
    with torch.inference_mode():
        for start in range(0, len(masked_inputs), args.batch_size):
            rows = masked_inputs[start:start + args.batch_size]
            collector.set_batch([f"wikitext_validation_{index:04d}" for index in range(start, start + len(rows))])
            tokens = torch.tensor(rows, dtype=torch.long, device="cuda")
            if args.inference_precision == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.bert(input_ids=tokens, attention_mask=torch.ones_like(tokens))
            else:
                model.bert(input_ids=tokens, attention_mask=torch.ones_like(tokens))
    collector.close()
    protocol = {
        "task": "BERT pretraining on WikiText-103",
        "model": "BERT-base MHA H=12",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 100000,
        "pretraining_seed": args.bert_seed,
        "source_root": str(args.source_root),
        "validation_file": str(args.bert_validation_file),
        "validation_file_sha256": shared.sha256(args.bert_validation_file),
        "examples": len(masked_inputs),
        "inputs": "first fixed-length packed WikiText-103 validation examples with the deterministic 15% BERT 80/10/10 mask used during validation",
        "sequence_length": 128,
        "validation_mask_seed": shared.BERT_MASK_SEED,
        "attention_metric": "pairwise Jensen-Shannon divergence divided by ln(2)",
        "output_metric": "pairwise cosine distance after aggregation, before output projection; MHA head coordinate frames are independent",
        "projected_metric": "pairwise cosine distance between per-head contributions after output-projection blocks",
        "aggregation": "per-example, per-layer means over unordered head pairs",
        "precision": "BF16 model inference; float32 diversity calculations" if args.inference_precision == "bf16" else "FP32 model inference and diversity calculations",
    }
    collector.write(args.output_dir, protocol)
    if args.ablation_examples:
        originals, special_masks = original_bert_inputs(tokenizer, args.ablation_examples, args.bert_validation_file)
        reconstructed = shared.fixed_mlm_inputs(
            originals, special_masks, mask_token_id=int(tokenizer.mask_token_id), vocab_size=len(tokenizer)
        )
        if reconstructed != masked_inputs[:args.ablation_examples]:
            raise RuntimeError("Ablation inputs do not match the frozen diversity inputs")
        labels = bert_ablation_labels(originals, special_masks)
        run_bert_ablation(model, reconstructed, labels, args.batch_size, args.output_dir)


def run_tinystories(args: argparse.Namespace) -> None:
    if args.tinystories_data_dir is None:
        raise ValueError("--tinystories-data-dir is required")
    sys.path.insert(0, str(args.source_root / "src"))
    from lgma.baselines import StandardMultiheadAttention
    from lgma.transformer import TinyTransformerLM

    checkpoint_sha = shared.sha256(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise RuntimeError("TinyStories checkpoint SHA-256 mismatch")
    prompts, vocabulary, stoi = shared.select_tinystories_prompts(args.tinystories_data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = dict(checkpoint["model_config"])
    if config.get("attention_type") != "mha":
        raise RuntimeError("Expected an MHA TinyStories checkpoint")
    model = TinyTransformerLM(vocab_size=len(vocabulary), **config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    gc.collect()
    model = model.cuda().eval()
    modules = [module for module in model.modules() if isinstance(module, StandardMultiheadAttention)]
    if len(modules) != config["num_layers"]:
        raise RuntimeError("Unexpected TinyStories MHA layer count")
    collector = MhaCollector(modules, [module.out_proj for module in modules],
                             task="tinystories", query_start=args.causal_query_start)
    grouped = defaultdict(list)
    for prompt in prompts:
        grouped[prompt["prefix_characters"]].append(prompt)
    with torch.inference_mode():
        for length in sorted(grouped):
            for start in range(0, len(grouped[length]), args.batch_size):
                rows = grouped[length][start:start + args.batch_size]
                collector.set_batch([row["prompt_id"] for row in rows])
                tokens = torch.tensor([[stoi[char] for char in row["prefix"]] for row in rows],
                                      dtype=torch.long, device="cuda")
                if args.inference_precision == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(tokens)
                else:
                    model(tokens)
    collector.close()
    protocol = {
        "task": "TinyStoriesV2",
        "model": "MHA H=16",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "source_root": str(args.source_root),
        "data_dir": str(args.tinystories_data_dir),
        "examples": len(prompts),
        "inputs": "same 200 frozen validation prefixes used for generation evaluation",
        "prompt_selection_seed": shared.PROMPT_SELECTION_SEED,
        "prefix_length_range": [min(row["prefix_characters"] for row in prompts), max(row["prefix_characters"] for row in prompts)],
        "causal_query_start": args.causal_query_start,
        "attention_metric": "pairwise Jensen-Shannon divergence divided by ln(2)",
        "output_metric": "pairwise cosine distance after aggregation, before output projection; MHA head coordinate frames are independent",
        "projected_metric": "pairwise cosine distance between per-head contributions after output-projection blocks",
        "aggregation": "per-example, per-layer means over unordered head pairs",
        "precision": "BF16 model inference; float32 diversity calculations" if args.inference_precision == "bf16" else "FP32 model inference and diversity calculations",
    }
    collector.write(args.output_dir, protocol)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The submitted analysis requires one CUDA device")
    if args.batch_size <= 0 or args.bert_examples <= 0:
        raise ValueError("Batch size and example count must be positive")
    if args.ablation_examples < 0 or args.ablation_examples > args.bert_examples:
        raise ValueError("Ablation examples must be between zero and BERT example count")
    torch.manual_seed(shared.BERT_MASK_SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.task == "bert":
        run_bert(args)
    else:
        run_tinystories(args)


if __name__ == "__main__":
    main()
