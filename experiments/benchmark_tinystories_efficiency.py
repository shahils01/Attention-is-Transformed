from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tinystories_runtime import load_tinystories_checkpoint, precision_context
from lgma.accounting import attention_accounting, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled TinyStories efficiency benchmark for prefill, cached decode, "
            "sliding-window decode, KV-cache storage, and peak GPU memory."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--prefill_context_lengths", type=int, nargs="+", default=[128, 256, 448, 512]
    )
    parser.add_argument(
        "--decode_context_lengths", type=int, nargs="+", default=[128, 256, 448]
    )
    parser.add_argument("--decode_tokens", type=int, default=64)
    parser.add_argument("--sliding_context_length", type=int, default=512)
    parser.add_argument("--sliding_decode_tokens", type=int, default=128)
    parser.add_argument(
        "--skip_sliding",
        action="store_true",
        help="Skip full-window recomputation; use for the primary KV-cached decoding benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--fuse_base_qkv", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fold_value_transform_into_output",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--sdpa_gqa_mode", choices=["auto", "native", "expand"], default=None
    )
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def timing_summary(seconds: list[float], work_items: int) -> dict[str, Any]:
    median = statistics.median(seconds)
    return {
        "raw_seconds": seconds,
        "median_ms": median * 1000.0,
        "p95_ms": percentile(seconds, 0.95) * 1000.0,
        "min_ms": min(seconds) * 1000.0,
        "throughput_tokens_per_second": work_items / max(median, 1e-12),
    }


def cache_num_bytes(past_key_values: Any) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer_cache in past_key_values
        for tensor in layer_cache
    )


def cuda_memory(device: torch.device, baseline_allocated: int = 0) -> dict[str, int | float]:
    if device.type != "cuda":
        return {}
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return {
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "incremental_peak_allocated_bytes": max(0, allocated - baseline_allocated),
        "peak_allocated_gib": allocated / 1024**3,
        "peak_reserved_gib": reserved / 1024**3,
        "incremental_peak_allocated_gib": max(0, allocated - baseline_allocated) / 1024**3,
    }


def reset_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


@torch.inference_mode()
def prefill(model, prompt: torch.Tensor, device: torch.device, precision: str):
    with precision_context(device, precision):
        return model(prompt, use_cache=True)


@torch.inference_mode()
def cached_decode(
    model,
    supplied_tokens: torch.Tensor,
    past_key_values: Any,
    device: torch.device,
    precision: str,
):
    logits = None
    for position in range(supplied_tokens.shape[1]):
        with precision_context(device, precision):
            logits, past_key_values = model(
                supplied_tokens[:, position : position + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
    return logits, past_key_values


@torch.inference_mode()
def sliding_decode(
    model,
    prompt: torch.Tensor,
    supplied_tokens: torch.Tensor,
    device: torch.device,
    precision: str,
):
    window = prompt
    logits = None
    for position in range(supplied_tokens.shape[1]):
        window = torch.cat((window[:, 1:], supplied_tokens[:, position : position + 1]), dim=1)
        with precision_context(device, precision):
            logits = model(window, use_cache=False)
    return logits


def timed_repetitions(
    setup: Callable[[], tuple[Any, ...]],
    workload: Callable[..., Any],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        args = setup()
        result = workload(*args)
        del args, result
    synchronize(device)
    durations = []
    for _ in range(repeats):
        args = setup()
        synchronize(device)
        started = time.perf_counter()
        result = workload(*args)
        synchronize(device)
        durations.append(time.perf_counter() - started)
        del args, result
    return durations


def profile_prefill_flops(
    model, prompt: torch.Tensor, device: torch.device, precision: str
) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        result = prefill(model, prompt, device, precision)
        synchronize(device)
    del result
    return int(sum(event.flops or 0 for event in profile.key_averages()))


def analytical_attention_flops(attention, batch_size: int, sequence_length: int) -> dict[str, int]:
    """Count dominant multiply-add FLOPs for one attention layer.

    The estimate counts dense projections, query--key products, attention--value
    products, and dense GT-MHA transformations. It intentionally excludes
    softmax, masking, normalization, elementwise operations, and matrix
    exponentiation; those omissions are reported explicitly in the output.
    """
    q_out = int(attention.q_proj.out_features)
    k_out = int(attention.k_proj.out_features)
    v_out = int(attention.v_proj.out_features)
    d_model = int(attention.q_proj.in_features)
    heads = int(attention.num_heads)
    score_dim = int(getattr(attention, "base_dim", attention.head_dim))
    value_dim = int(getattr(attention, "value_dim", attention.head_dim))
    projection = 2 * batch_size * sequence_length * d_model * (q_out + k_out + v_out)
    score = 2 * batch_size * heads * sequence_length * sequence_length * score_dim
    aggregation = 2 * batch_size * heads * sequence_length * sequence_length * value_dim
    output = (
        2
        * batch_size
        * sequence_length
        * int(attention.out_proj.in_features)
        * int(attention.out_proj.out_features)
    )
    query_transform = 0
    value_transform = 0
    transform_construction = 0
    if attention.__class__.__name__ == "LieGeneratedMetricAttention":
        generators = int(attention.num_generators)
        query_transform = 2 * batch_size * sequence_length * heads * score_dim * score_dim
        transform_construction += 2 * heads * generators * score_dim * score_dim
        if attention.value_transform_mode != "none":
            transform_construction += 2 * heads * generators * value_dim * value_dim
            if attention.fold_value_transform_into_output:
                value_transform = 2 * heads * value_dim * value_dim * d_model
            else:
                value_transform = (
                    2 * batch_size * sequence_length * heads * value_dim * value_dim
                )
    components = {
        "qkv_projection": projection,
        "query_transform_application": query_transform,
        "score_products": score,
        "attention_value_products": aggregation,
        "value_transform_or_fold": value_transform,
        "output_projection": output,
        "transform_construction": transform_construction,
    }
    components["dominant_total"] = sum(components.values())
    return components


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("batch_size/repeats must be positive and warmup non-negative")
    if args.decode_tokens <= 0 or args.sliding_decode_tokens <= 0:
        raise SystemExit("decode token counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    overrides = {
        key: value
        for key, value in {
            "fuse_base_qkv": args.fuse_base_qkv,
            "fold_value_transform_into_output": args.fold_value_transform_into_output,
            "sdpa_gqa_mode": args.sdpa_gqa_mode,
        }.items()
        if value is not None
    }
    model, _, _, val_encoded, config, step = load_tinystories_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        model_config_overrides=overrides,
    )
    model.eval()
    context_limit = int(config["context_length"])
    requested = set(args.prefill_context_lengths + args.decode_context_lengths)
    if not args.skip_sliding:
        requested.add(args.sliding_context_length)
    if min(requested) <= 0 or max(requested) > context_limit:
        raise SystemExit(f"all context lengths must be in [1, {context_limit}]")
    if any(length + args.decode_tokens > context_limit for length in args.decode_context_lengths):
        raise SystemExit("cached prompt length plus decode tokens exceeds model context")
    if val_encoded.numel() < max(requested):
        raise SystemExit("validation text is shorter than the requested context")

    base_prompt = val_encoded[: max(requested)].to(device)
    prompt_by_length = {
        length: base_prompt[:length].unsqueeze(0).repeat(args.batch_size, 1)
        for length in requested
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    common_decode = torch.randint(
        0,
        int(model.vocab_size),
        (args.batch_size, args.decode_tokens),
        device=device,
        generator=generator,
    )
    common_sliding = None
    if not args.skip_sliding:
        common_sliding = torch.randint(
            0,
            int(model.vocab_size),
            (args.batch_size, args.sliding_decode_tokens),
            device=device,
            generator=generator,
        )

    prefill_results = []
    for length in args.prefill_context_lengths:
        prompt = prompt_by_length[length]
        durations = timed_repetitions(
            lambda prompt=prompt: (model, prompt, device, args.precision),
            prefill,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        baseline = reset_memory(device)
        prefill_logits, cache = prefill(model, prompt, device, args.precision)
        synchronize(device)
        memory = cuda_memory(device, baseline)
        cache_bytes = cache_num_bytes(cache)
        del prefill_logits, cache
        row: dict[str, Any] = {
            "context_length": length,
            **timing_summary(durations, args.batch_size * length),
            "memory": memory,
            "measured_kv_cache_bytes": cache_bytes,
            "analytical_attention_flops_per_layer": analytical_attention_flops(
                model.first_attention, args.batch_size, length
            ),
        }
        if args.profile_flops:
            row["profiled_operator_flops"] = profile_prefill_flops(
                model, prompt, device, args.precision
            )
        prefill_results.append(row)

    decode_results = []
    final_cache_bytes = None
    for length in args.decode_context_lengths:
        prompt = prompt_by_length[length]

        def setup_decode(prompt=prompt):
            _, cache = prefill(model, prompt, device, args.precision)
            return model, common_decode, cache, device, args.precision

        durations = timed_repetitions(
            setup_decode,
            cached_decode,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        setup_logits, setup_cache = prefill(model, prompt, device, args.precision)
        del setup_logits
        baseline = reset_memory(device)
        _, final_cache = cached_decode(
            model, common_decode, setup_cache, device, args.precision
        )
        synchronize(device)
        memory = cuda_memory(device, baseline)
        final_cache_bytes = cache_num_bytes(final_cache)
        del setup_cache, final_cache
        decode_results.append(
            {
                "context_length": length,
                "decode_tokens": args.decode_tokens,
                **timing_summary(durations, args.batch_size * args.decode_tokens),
                "median_ms_per_decode_step": (
                    statistics.median(durations) * 1000.0 / args.decode_tokens
                ),
                "memory": memory,
                "measured_final_kv_cache_bytes": final_cache_bytes,
            }
        )

    sliding_payload = None
    if not args.skip_sliding:
        sliding_prompt = prompt_by_length[args.sliding_context_length]
        sliding_durations = timed_repetitions(
            lambda: (model, sliding_prompt, common_sliding, device, args.precision),
            sliding_decode,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        baseline = reset_memory(device)
        sliding_result = sliding_decode(
            model, sliding_prompt, common_sliding, device, args.precision
        )
        synchronize(device)
        sliding_memory = cuda_memory(device, baseline)
        del sliding_result
        sliding_payload = {
            "context_length": args.sliding_context_length,
            "decode_tokens": args.sliding_decode_tokens,
            **timing_summary(
                sliding_durations, args.batch_size * args.sliding_decode_tokens
            ),
            "median_ms_per_decode_step": (
                statistics.median(sliding_durations)
                * 1000.0
                / args.sliding_decode_tokens
            ),
            "memory": sliding_memory,
            "mode": "full_window_recomputation_after_each_supplied_token",
        }

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    accounting = attention_accounting(
        model.first_attention,
        sequence_length=max(requested),
        batch_size=args.batch_size,
        dtype=dtype,
    )
    theoretical_cache = (
        accounting.kv_cache_bytes_per_token_per_layer
        * args.batch_size
        * len(model.blocks)
    )
    payload = {
        "schema_version": 1,
        "protocol": "paper_controlled_efficiency_v1",
        "model_label": args.model_label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "attention_type": config["attention_type"],
        "model_config": config,
        "model_parameters": count_parameters(model),
        "batch_size": args.batch_size,
        "precision": args.precision,
        "warmup_repetitions": args.warmup,
        "timed_repetitions": args.repeats,
        "fixed_token_seed": args.seed,
        "optimized_execution": overrides,
        "prefill": prefill_results,
        "cached_decode": decode_results,
        "sliding_window_decode": sliding_payload,
        "attention_accounting": accounting.__dict__,
        "theoretical_kv_cache_bytes_per_cached_token_all_layers_batch": theoretical_cache,
        "hardware": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "device_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "notes": [
            "All methods receive the same fixed supplied decode-token stream.",
            "Checkpoint loading is outside every timed region.",
            "Peak memory includes model weights; incremental peak subtracts allocated bytes immediately before the phase.",
            "Profiled FLOPs include only operators for which PyTorch exposes FLOP formulas.",
            "Analytical attention FLOPs count dominant multiply-add operations but omit softmax, masking, normalization, elementwise work, and matrix exponentiation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
