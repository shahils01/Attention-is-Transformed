from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from tinystories_runtime import load_tinystories_checkpoint, precision_context

from lgma.accounting import attention_accounting, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TinyStories checkpoint prefill and uncached autoregressive decoding."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument("--context_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--decode_tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--profile_flops",
        action="store_true",
        help=(
            "Use the PyTorch profiler once per context length to report FLOPs from "
            "operators that expose FLOP formulas. Fused/custom kernels may be omitted."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def latency_summary(seconds: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(seconds) * 1000.0,
        "p95_ms": percentile(seconds, 0.95) * 1000.0,
        "min_ms": min(seconds) * 1000.0,
    }


@torch.no_grad()
def prefill_once(model, input_ids: torch.Tensor, device: torch.device, precision: str) -> None:
    with precision_context(device, precision):
        model(input_ids)


@torch.no_grad()
def decode_once(
    model,
    prompt: torch.Tensor,
    context_length: int,
    decode_tokens: int,
    device: torch.device,
    precision: str,
) -> None:
    ids = prompt
    for _ in range(decode_tokens):
        context = ids[:, -context_length:]
        with precision_context(device, precision):
            logits = model(context)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)


def profile_prefill_flops(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    precision: str,
) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        prefill_once(model, input_ids, device, precision)
        synchronize(device)
    return int(sum(event.flops or 0 for event in profile.key_averages()))


def benchmark_context(
    model,
    encoded: torch.Tensor,
    context_length: int,
    batch_size: int,
    decode_tokens: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    precision: str,
    profile_flops: bool = False,
) -> dict[str, Any]:
    prompt = encoded[:context_length].unsqueeze(0).repeat(batch_size, 1).to(device)
    for _ in range(warmup):
        prefill_once(model, prompt, device, precision)
        decode_once(model, prompt, context_length, min(decode_tokens, 2), device, precision)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefill_times = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        prefill_once(model, prompt, device, precision)
        synchronize(device)
        prefill_times.append(time.perf_counter() - started)

    decode_times = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        decode_once(model, prompt, context_length, decode_tokens, device, precision)
        synchronize(device)
        decode_times.append(time.perf_counter() - started)

    prefill_median = statistics.median(prefill_times)
    decode_median = statistics.median(decode_times)
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]
    accounting = attention_accounting(
        model.first_attention,
        sequence_length=context_length,
        batch_size=batch_size,
        dtype=dtype,
    )
    result: dict[str, Any] = {
        "context_length": context_length,
        "batch_size": batch_size,
        "prefill": {
            **latency_summary(prefill_times),
            "tokens_per_second": batch_size * context_length / max(prefill_median, 1e-12),
        },
        "decode": {
            **latency_summary(decode_times),
            "generated_tokens": decode_tokens,
            "tokens_per_second": batch_size * decode_tokens / max(decode_median, 1e-12),
            "median_ms_per_token": decode_median * 1000.0 / decode_tokens,
            "mode": "full_context_recompute_no_kv_cache",
        },
        "attention_accounting": accounting.__dict__,
    }
    if profile_flops:
        result["profiled_prefill_flops"] = profile_prefill_flops(
            model, prompt, device, precision
        )
        result["profiled_flops_note"] = (
            "PyTorch operator-reported FLOPs; unsupported fused/custom kernels may be omitted."
        )
    if device.type == "cuda":
        result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["peak_memory_allocated_gib"] = result["peak_memory_allocated_bytes"] / 1024**3
        result["peak_memory_reserved_gib"] = result["peak_memory_reserved_bytes"] / 1024**3
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "context_length",
        "batch_size",
        "prefill_median_ms",
        "prefill_p95_ms",
        "prefill_tokens_per_second",
        "decode_median_ms_per_token",
        "decode_tokens_per_second",
        "peak_memory_allocated_gib",
        "peak_memory_reserved_gib",
        "kv_cache_bytes_per_token_per_layer",
        "attention_score_flops",
        "profiled_prefill_flops",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "context_length": result["context_length"],
                    "batch_size": result["batch_size"],
                    "prefill_median_ms": result["prefill"]["median_ms"],
                    "prefill_p95_ms": result["prefill"]["p95_ms"],
                    "prefill_tokens_per_second": result["prefill"]["tokens_per_second"],
                    "decode_median_ms_per_token": result["decode"]["median_ms_per_token"],
                    "decode_tokens_per_second": result["decode"]["tokens_per_second"],
                    "peak_memory_allocated_gib": result.get("peak_memory_allocated_gib"),
                    "peak_memory_reserved_gib": result.get("peak_memory_reserved_gib"),
                    "kv_cache_bytes_per_token_per_layer": result["attention_accounting"][
                        "kv_cache_bytes_per_token_per_layer"
                    ],
                    "attention_score_flops": result["attention_accounting"][
                        "attention_score_flops"
                    ],
                    "profiled_prefill_flops": result.get("profiled_prefill_flops"),
                }
            )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.decode_tokens <= 0 or args.repeats <= 0:
        raise SystemExit("--batch_size, --decode_tokens, and --repeats must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")

    device = torch.device(args.device)
    model, _, _, val_encoded, config, step = load_tinystories_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
    )
    model_context = int(config["context_length"])
    context_lengths = args.context_lengths or sorted(
        {length for length in (128, 256, model_context) if length <= model_context}
    )
    invalid = [length for length in context_lengths if length <= 0 or length > model_context]
    if invalid:
        raise SystemExit(
            f"context lengths must be in [1, {model_context}] for this checkpoint: {invalid}"
        )
    if val_encoded.numel() < max(context_lengths):
        raise SystemExit("validation text is shorter than the largest context length")

    results = [
        benchmark_context(
            model,
            val_encoded,
            context_length,
            args.batch_size,
            args.decode_tokens,
            args.warmup,
            args.repeats,
            device,
            args.precision,
            args.profile_flops,
        )
        for context_length in context_lengths
    ]
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "attention_type": config["attention_type"],
        "parameters": count_parameters(model),
        "device": str(device),
        "precision": args.precision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "decode_mode": "full_context_recompute_no_kv_cache",
        "profile_flops": args.profile_flops,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.csv is not None:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
