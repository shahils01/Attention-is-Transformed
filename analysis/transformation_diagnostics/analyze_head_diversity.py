#!/usr/bin/env python3
"""Functional head-diversity analysis for GT-MHA checkpoints.

The script measures two complementary quantities at every layer:

1. normalized pairwise Jensen--Shannon divergence between attention
   distributions (where heads attend); and
2. pairwise cosine distance between per-head representations after attention
   aggregation and the learned value transform, but before the shared output
   projection (what heads contribute).

For multi-orbit GT-MHA, pairwise statistics are separated into within-base
and across-base comparisons.  The script supports the frozen TinyStories
generation prefixes and the fixed masked WikiText-103 validation inputs used
for BERT evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


TINYSTORIES_ROOT = Path("/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_full_20260915")
TINYSTORIES_DATA = Path("/work/hdd/bifg/arai3/lgma_data/tinystories")
TINYSTORIES_CHECKPOINT = TINYSTORIES_ROOT / "GT-MHA residual.pt"
TINYSTORIES_CHECKPOINT_SHA256 = (
    "a3b36ada91bb42b22b6df4da2b36be5cd3333f5a473e857c326aa06323ae06f2"
)
TINYSTORIES_TRAIN_SHA256 = (
    "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443"
)
TINYSTORIES_VALID_SHA256 = (
    "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
)
TINYSTORIES_DEV_STORY_IDS = {
    4319,
    1076,
    11823,
    8143,
    18790,
    207,
    6254,
    18123,
    2325,
    19976,
    23645,
    2356,
    17347,
    21373,
    10444,
    21087,
    1620,
    15611,
    7598,
    15817,
}

BERT_SOURCE_ROOT = TINYSTORIES_ROOT / "source"
BERT_DATA = Path("/work/hdd/bifg/arai3/lgma_data/wikitext-103-raw-v1")
BERT_CHECKPOINTS = {
    43: Path(
        "/work/hdd/bifg/arai3/lgma_runs/"
        "bert_base_gt_mha_residual_b4g8h12_random_seed43_explicit_fuseqkv_v2/"
        "checkpoint-100000"
    ),
}
BERT_CHECKPOINT_SHAS = {
    43: "242a3d784f41bf492055b50a15c36eba29f98537fabbfae01c83490b4a5370f3",
}

PROMPT_SELECTION_SEED = 20260919
BERT_MASK_SEED = 17029
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("tinystories", "bert"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bert-seed", type=int, default=43)
    parser.add_argument("--bert-examples", type=int, default=1000)
    parser.add_argument("--expected-num-bases", type=int, default=4)
    parser.add_argument("--bert-checkpoint-dir", type=Path)
    parser.add_argument("--bert-checkpoint-sha256")
    parser.add_argument("--bert-source-root", type=Path)
    parser.add_argument("--bert-validation-file", type=Path)
    parser.add_argument(
        "--inference-precision",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--tinystories-checkpoint", type=Path)
    parser.add_argument("--tinystories-checkpoint-sha256")
    parser.add_argument("--tinystories-source-root", type=Path)
    parser.add_argument("--tinystories-data-dir", type=Path)
    parser.add_argument(
        "--causal-query-start",
        type=int,
        default=16,
        help="Ignore earlier causal queries whose support is mechanically small.",
    )
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def pair_masks(
    num_heads: int,
    num_bases: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    upper = torch.triu(
        torch.ones(num_heads, num_heads, dtype=torch.bool, device=device), diagonal=1
    )
    heads_per_base = num_heads // num_bases
    base_ids = torch.arange(num_heads, device=device) // heads_per_base
    same_base = base_ids[:, None] == base_ids[None, :]
    return {
        "overall": upper,
        "within_base": upper & same_base,
        "across_base": upper & ~same_base,
    }


def attention_jsd_matrix(attn: torch.Tensor, query_start: int) -> torch.Tensor:
    """Normalized JSD matrices with shape [batch, heads, heads]."""
    probabilities = attn.float()[:, :, query_start:, :]
    batch, heads, _, _ = probabilities.shape
    output = torch.zeros(batch, heads, heads, device=attn.device, dtype=torch.float32)
    head_i, head_j = torch.triu_indices(heads, heads, offset=1, device=attn.device)
    normalization = math.log(2.0)
    for start in range(0, head_i.numel(), 32):
        i = head_i[start : start + 32]
        j = head_j[start : start + 32]
        p = probabilities[:, i, :, :]
        q = probabilities[:, j, :, :]
        mixture = 0.5 * (p + q)
        log_mixture = mixture.clamp_min(EPS).log()
        kl_p = torch.where(
            p > 0,
            p * (p.clamp_min(EPS).log() - log_mixture),
            torch.zeros_like(p),
        ).sum(dim=-1)
        kl_q = torch.where(
            q > 0,
            q * (q.clamp_min(EPS).log() - log_mixture),
            torch.zeros_like(q),
        ).sum(dim=-1)
        values = (0.5 * (kl_p + kl_q)).mean(dim=-1) / normalization
        output[:, i, j] = values
        output[:, j, i] = values
    return output


def output_cosine_distance_matrix(
    outputs: torch.Tensor,
    query_start: int,
) -> torch.Tensor:
    """Cosine-distance matrices for [batch, heads, time, head_dim] outputs."""
    normalized = F.normalize(outputs.float()[:, :, query_start:, :], dim=-1, eps=1e-8)
    normalized = normalized.transpose(1, 2)
    similarity = torch.matmul(normalized, normalized.transpose(-1, -2)).mean(dim=1)
    distance = 1.0 - similarity
    distance = 0.5 * (distance + distance.transpose(-1, -2))
    distance.diagonal(dim1=-2, dim2=-1).zero_()
    return distance


class DiversityCollector:
    def __init__(
        self,
        modules: list[torch.nn.Module],
        *,
        query_start: int,
        example_ids: list[str],
        hook_modules: list[torch.nn.Module] | None = None,
    ) -> None:
        self.modules = modules
        self.hook_modules = modules if hook_modules is None else hook_modules
        if len(self.hook_modules) != len(self.modules):
            raise ValueError("hook_modules and modules must have the same length")
        self.query_start = query_start
        self.current_ids: list[str] = []
        self.example_ids = example_ids
        self.rows: list[dict[str, Any]] = []
        self.matrix_sums: dict[str, list[torch.Tensor]] = {
            "attention_jsd": [],
            "output_cosine_distance": [],
        }
        self.matrix_counts: list[int] = [0] * len(modules)
        self.handles = []
        for layer, (module, hook_module) in enumerate(
            zip(modules, self.hook_modules), start=1
        ):
            self.matrix_sums["attention_jsd"].append(
                torch.zeros(module.num_heads, module.num_heads, dtype=torch.float64)
            )
            self.matrix_sums["output_cosine_distance"].append(
                torch.zeros(module.num_heads, module.num_heads, dtype=torch.float64)
            )
            self.handles.append(
                hook_module.register_forward_pre_hook(
                    self._make_hook(layer, module), with_kwargs=True
                )
            )

    def _make_hook(self, layer: int, attention_module: torch.nn.Module):
        def hook(_hook_module, args, kwargs):
            x = args[0]
            module = attention_module
            q, k, v = module._project(x, context=kwargs.get("context"))
            raw_outputs, attn = module._explicit_attention(
                q,
                k,
                v,
                kwargs.get("attn_mask"),
                kwargs.get("key_padding_mask"),
            )
            transformed_outputs = module._apply_value_transforms_to_outputs(raw_outputs)
            jsd = attention_jsd_matrix(attn, self.query_start)
            output_distance = output_cosine_distance_matrix(
                transformed_outputs, self.query_start
            )
            if jsd.shape[0] != len(self.current_ids):
                raise RuntimeError(
                    "Collector batch identifiers do not match hook batch"
                )
            masks = pair_masks(
                module.num_heads, module.num_base_heads, device=jsd.device
            )
            for row_index, example_id in enumerate(self.current_ids):
                row: dict[str, Any] = {
                    "example_id": example_id,
                    "layer": layer,
                }
                for group, mask in masks.items():
                    if mask.any():
                        row[f"attention_jsd_{group}"] = float(
                            jsd[row_index][mask].mean().cpu()
                        )
                        row[f"output_cosine_distance_{group}"] = float(
                            output_distance[row_index][mask].mean().cpu()
                        )
                self.rows.append(row)
            self.matrix_sums["attention_jsd"][layer - 1] += jsd.double().sum(0).cpu()
            self.matrix_sums["output_cosine_distance"][layer - 1] += (
                output_distance.double().sum(0).cpu()
            )
            self.matrix_counts[layer - 1] += jsd.shape[0]

        return hook

    def set_batch(self, example_ids: list[str]) -> None:
        self.current_ids = example_ids

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def mean_matrices(self) -> dict[str, np.ndarray]:
        results = {}
        for metric, matrices in self.matrix_sums.items():
            results[metric] = np.stack(
                [
                    (matrix / count).numpy()
                    for matrix, count in zip(matrices, self.matrix_counts)
                ]
            )
        return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sorted(key for key in rows[0] if key not in {"example_id", "layer"})
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)

    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "q1": float(np.quantile(array, 0.25)),
            "q3": float(np.quantile(array, 0.75)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "all_layers": {
            metric: stats([float(row[metric]) for row in rows if metric in row])
            for metric in metrics
        },
        "by_layer": {
            str(layer): {
                metric: stats(
                    [float(row[metric]) for row in layer_rows if metric in row]
                )
                for metric in metrics
            }
            for layer, layer_rows in sorted(by_layer.items())
        },
    }


def write_outputs(
    output_dir: Path,
    collector: DiversityCollector,
    protocol: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_example_layer.jsonl").open("w", encoding="utf-8") as handle:
        for row in collector.rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    matrices = collector.mean_matrices()
    np.savez_compressed(output_dir / "mean_pairwise_matrices.npz", **matrices)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "summary": summarize(collector.rows),
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
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


def select_tinystories_prompts(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    train_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    validation_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    if sha256(train_path) != TINYSTORIES_TRAIN_SHA256:
        raise RuntimeError("TinyStories training-corpus hash mismatch")
    if sha256(validation_path) != TINYSTORIES_VALID_SHA256:
        raise RuntimeError("TinyStories validation-corpus hash mismatch")

    validation_text = validation_path.read_text(encoding="utf-8")
    stories = validation_text.split("<|endoftext|>")
    development_hashes = {
        canonical_hash(stories[index]) for index in TINYSTORIES_DEV_STORY_IDS
    }
    candidates: dict[str, tuple[int, int, int, str]] = {}
    for index, story in enumerate(stories):
        cut = next(
            (
                position
                for position in range(120, min(161, len(story)))
                if story[position].isspace()
            ),
            None,
        )
        if len(story) < 256 or cut is None:
            continue
        digest = canonical_hash(story)
        if digest in development_hashes or digest in candidates:
            continue
        candidates[digest] = (len(story), index, cut, digest)

    train_matches: set[str] = set()
    vocabulary: set[str] = set(validation_text)
    with train_path.open(encoding="utf-8") as handle:
        pending = ""
        while chunk := handle.read(8 * 1024 * 1024):
            vocabulary.update(chunk)
            parts = (pending + chunk).split("<|endoftext|>")
            pending = parts.pop()
            for story in parts:
                digest = canonical_hash(story)
                if digest in candidates:
                    train_matches.add(digest)
        if pending.strip():
            digest = canonical_hash(pending)
            if digest in candidates:
                train_matches.add(digest)

    eligible = sorted(
        row for digest, row in candidates.items() if digest not in train_matches
    )
    rng = random.Random(PROMPT_SELECTION_SEED)
    prompts: list[dict[str, Any]] = []
    for quartile in range(4):
        pool = eligible[
            len(eligible) * quartile // 4 : len(eligible) * (quartile + 1) // 4
        ]
        for _, story_id, cut, _ in rng.sample(pool, 50):
            story = stories[story_id]
            prompts.append(
                {
                    "prompt_id": f"eval_{len(prompts) + 1:03d}",
                    "story_id": story_id,
                    "story_sha256": hashlib.sha256(story.encode()).hexdigest(),
                    "prefix": story[:cut],
                    "prefix_characters": cut,
                }
            )
    if (
        len(prompts) != 200
        or prompts[0]["story_id"] != 6116
        or prompts[-1]["story_id"] != 10599
    ):
        raise RuntimeError("Frozen TinyStories prompt reconstruction failed")
    vocab = sorted(vocabulary)
    return prompts, vocab, {character: index for index, character in enumerate(vocab)}


def run_tinystories(args: argparse.Namespace) -> None:
    source_root = args.tinystories_source_root or (TINYSTORIES_ROOT / "source")
    data_dir = args.tinystories_data_dir or TINYSTORIES_DATA
    checkpoint_path = args.tinystories_checkpoint or TINYSTORIES_CHECKPOINT
    expected_checkpoint_sha = (
        args.tinystories_checkpoint_sha256 or TINYSTORIES_CHECKPOINT_SHA256
    )
    sys.path.insert(
        0, str(source_root / "src" if (source_root / "src").is_dir() else source_root)
    )
    from lgma.attention import LieGeneratedMetricAttention  # noqa: PLC0415
    from lgma.transformer import TinyTransformerLM  # noqa: PLC0415

    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError("TinyStories checkpoint hash mismatch")
    prompts, vocabulary, stoi = select_tinystories_prompts(data_dir)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = dict(checkpoint["model_config"])
    model = TinyTransformerLM(vocab_size=len(vocabulary), **config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 250000))
    del checkpoint
    gc.collect()
    # Keep checkpoint parameters in FP32 and use autocast for the forward pass,
    # matching the established TinyStories evaluation path.
    model = model.to(device="cuda").eval()
    modules = [
        module
        for module in model.modules()
        if isinstance(module, LieGeneratedMetricAttention)
    ]
    if len(modules) != config["num_layers"]:
        raise RuntimeError("Unexpected number of TinyStories GT-MHA layers")
    if any(module.num_base_heads != args.expected_num_bases for module in modules):
        raise RuntimeError(
            f"TinyStories checkpoint is not the expected C={args.expected_num_bases} model"
        )

    collector = DiversityCollector(
        modules,
        query_start=args.causal_query_start,
        example_ids=[row["prompt_id"] for row in prompts],
    )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        grouped[int(prompt["prefix_characters"])].append(prompt)

    with torch.inference_mode():
        for length in sorted(grouped):
            group = grouped[length]
            for start in range(0, len(group), args.batch_size):
                rows = group[start : start + args.batch_size]
                collector.set_batch([row["prompt_id"] for row in rows])
                tokens = torch.tensor(
                    [[stoi[character] for character in row["prefix"]] for row in rows],
                    dtype=torch.long,
                    device="cuda",
                )
                if args.inference_precision == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model(tokens)
                else:
                    _ = model(tokens)
                del tokens
    collector.close()
    protocol = {
        "task": "TinyStoriesV2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "source_root": str(source_root),
        "data_dir": str(data_dir),
        "model": f"GT-MHA residual C={args.expected_num_bases} G=8 H=16",
        "examples": len(prompts),
        "inputs": "same 200 frozen validation prefixes used for generation evaluation",
        "prompt_selection_seed": PROMPT_SELECTION_SEED,
        "prefix_length_range": [
            min(row["prefix_characters"] for row in prompts),
            max(row["prefix_characters"] for row in prompts),
        ],
        "causal_query_start": args.causal_query_start,
        "attention_metric": "pairwise Jensen-Shannon divergence divided by ln(2)",
        "output_metric": (
            "pairwise cosine distance after aggregation and value transformation, "
            "before output projection"
        ),
        "aggregation": "per-example, per-layer means over unordered head pairs",
        "precision": (
            "BF16 model inference; float32 diversity calculations"
            if args.inference_precision == "bf16"
            else "FP32 model inference and diversity calculations"
        ),
    }
    write_outputs(args.output_dir, collector, protocol)


def fixed_mlm_inputs(
    input_ids: list[list[int]],
    special_masks: list[list[int]],
    *,
    mask_token_id: int,
    vocab_size: int,
) -> list[list[int]]:
    masked: list[list[int]] = []
    for example_index, (ids, special_mask) in enumerate(zip(input_ids, special_masks)):
        generator = torch.Generator().manual_seed(BERT_MASK_SEED + example_index)
        inputs = torch.tensor(ids, dtype=torch.long)
        probability = torch.full(inputs.shape, 0.15, dtype=torch.float32)
        probability.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), 0.0)
        selected = torch.bernoulli(probability, generator=generator).bool()
        replaced = (
            torch.bernoulli(torch.full(inputs.shape, 0.8), generator=generator).bool()
            & selected
        )
        inputs[replaced] = mask_token_id
        random_replaced = (
            torch.bernoulli(torch.full(inputs.shape, 0.5), generator=generator).bool()
            & selected
            & ~replaced
        )
        random_tokens = torch.randint(
            vocab_size, inputs.shape, generator=generator, dtype=torch.long
        )
        inputs[random_replaced] = random_tokens[random_replaced]
        masked.append(inputs.tolist())
    return masked


def build_bert_validation_inputs(
    tokenizer,
    count: int,
    validation_path: Path,
) -> list[list[int]]:
    if validation_path.suffix == ".parquet":
        import pyarrow.parquet as parquet  # noqa: PLC0415

        texts = parquet.read_table(validation_path, columns=["text"])[
            "text"
        ].to_pylist()
    elif validation_path.suffix == ".arrow":
        from datasets import Dataset  # noqa: PLC0415

        texts = Dataset.from_file(str(validation_path))["text"]
    else:
        raise ValueError(f"Unsupported BERT validation file: {validation_path}")
    packed_ids: list[list[int]] = []
    packed_special: list[list[int]] = []
    map_batch_size = 1000
    sequence_length = 128
    for start in range(0, len(texts), map_batch_size):
        encoded = tokenizer(
            texts[start : start + map_batch_size],
            return_special_tokens_mask=True,
        )
        joined_ids = sum(encoded["input_ids"], [])
        joined_special = sum(encoded["special_tokens_mask"], [])
        usable = (len(joined_ids) // sequence_length) * sequence_length
        for offset in range(0, usable, sequence_length):
            packed_ids.append(joined_ids[offset : offset + sequence_length])
            packed_special.append(joined_special[offset : offset + sequence_length])
            if len(packed_ids) == count:
                return fixed_mlm_inputs(
                    packed_ids,
                    packed_special,
                    mask_token_id=int(tokenizer.mask_token_id),
                    vocab_size=len(tokenizer),
                )
    raise RuntimeError(f"Only constructed {len(packed_ids)} BERT validation sequences")


def run_bert(args: argparse.Namespace) -> None:
    seed = args.bert_seed
    checkpoint_dir = args.bert_checkpoint_dir or BERT_CHECKPOINTS.get(seed)
    expected_checkpoint_sha = args.bert_checkpoint_sha256 or BERT_CHECKPOINT_SHAS.get(
        seed
    )
    if checkpoint_dir is None or expected_checkpoint_sha is None:
        raise ValueError(
            f"BERT seed {seed} requires --bert-checkpoint-dir and "
            "--bert-checkpoint-sha256 on this host"
        )
    source_root = args.bert_source_root or BERT_SOURCE_ROOT
    validation_path = args.bert_validation_file or (
        BERT_DATA / "validation-00000-of-00001.parquet"
    )
    weights_path = checkpoint_dir / "model.safetensors"
    checkpoint_sha = sha256(weights_path)
    if checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError("BERT checkpoint hash mismatch")

    sys.path.insert(0, str(source_root / "src"))
    from lgma.attention import LieGeneratedMetricAttention  # noqa: PLC0415
    from lgma.bert import replace_bert_self_attention  # noqa: PLC0415
    from safetensors.torch import load_model  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer  # noqa: PLC0415

    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    validation_inputs = build_bert_validation_inputs(
        tokenizer, args.bert_examples, validation_path
    )
    model = AutoModelForMaskedLM.from_config(config)
    replacement_options = {
        "attention_type": "gt_mha_residual",
        "num_base_heads": args.expected_num_bases,
        "num_generators": 8,
        "generator_mixing": "softmax",
        "use_sdpa": True,
        "fuse_base_qkv": True,
        "initialize_from_mha": False,
        "enforce_paper_gt_mha": args.expected_num_bases == 4,
    }
    supported_options = inspect.signature(replace_bert_self_attention).parameters
    replace_bert_self_attention(
        model,
        **{
            name: value
            for name, value in replacement_options.items()
            if name in supported_options
        },
    )
    load_model(model, str(weights_path), strict=True)
    # Keep checkpoint parameters in FP32 and use autocast for the forward pass,
    # matching the BERT validation configuration.
    model = model.to(device="cuda").eval()
    adapters = [
        module
        for module in model.modules()
        if isinstance(
            getattr(module, "gt_attention", None), LieGeneratedMetricAttention
        )
    ]
    modules = [adapter.gt_attention for adapter in adapters]
    if len(modules) != config.num_hidden_layers:
        raise RuntimeError("Unexpected number of BERT GT-MHA layers")
    if any(module.num_base_heads != args.expected_num_bases for module in modules):
        raise RuntimeError(
            f"BERT checkpoint is not the expected C={args.expected_num_bases} model"
        )

    example_ids = [
        f"wikitext_validation_{index:04d}" for index in range(args.bert_examples)
    ]
    collector = DiversityCollector(
        modules,
        query_start=0,
        example_ids=example_ids,
        hook_modules=adapters,
    )
    with torch.inference_mode():
        for start in range(0, len(validation_inputs), args.batch_size):
            batch = validation_inputs[start : start + args.batch_size]
            ids = example_ids[start : start + len(batch)]
            collector.set_batch(ids)
            tokens = torch.tensor(batch, dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(tokens)
            if args.inference_precision == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.bert(input_ids=tokens, attention_mask=attention_mask)
            else:
                _ = model.bert(input_ids=tokens, attention_mask=attention_mask)
            del tokens, attention_mask
    collector.close()
    protocol = {
        "task": "BERT pretraining on WikiText-103",
        "checkpoint": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 100000,
        "pretraining_seed": seed,
        "source_root": str(source_root),
        "validation_file": str(validation_path),
        "validation_file_sha256": sha256(validation_path),
        "model": (f"BERT-base GT-MHA residual C={args.expected_num_bases} G=8 H=12"),
        "examples": args.bert_examples,
        "inputs": (
            "first fixed-length packed WikiText-103 validation examples with the "
            "deterministic 15% BERT 80/10/10 mask used during validation"
        ),
        "sequence_length": 128,
        "validation_mask_seed": BERT_MASK_SEED,
        "attention_metric": "pairwise Jensen-Shannon divergence divided by ln(2)",
        "output_metric": (
            "pairwise cosine distance after aggregation and value transformation, "
            "before output projection"
        ),
        "aggregation": "per-example, per-layer means over unordered head pairs",
        "precision": (
            "BF16 model inference; float32 diversity calculations"
            if args.inference_precision == "bf16"
            else "FP32 model inference and diversity calculations"
        ),
    }
    write_outputs(args.output_dir, collector, protocol)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Head-diversity analysis requires a CUDA device")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    torch.manual_seed(17029)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.task == "tinystories":
        run_tinystories(args)
    else:
        run_bert(args)


if __name__ == "__main__":
    main()
