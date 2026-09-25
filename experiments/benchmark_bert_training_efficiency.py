from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.bert import bert_parameter_counts, load_bert_masked_lm


METHODS = {
    "MHA": "mha",
    "GQA": "gqa",
    "MQA": "mqa",
    "Collaborative MHA": "collaborative",
    "GT-MHA": "gt_mha_residual",
    "GT-MHA (quadratic)": "gt_mha_quadratic",
    "GT-MHA (exact exponential)": "gt_mha_exact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched single-GPU BERT MLM training throughput and peak-memory benchmark."
    )
    parser.add_argument("--model_name_or_path", default="google-bert/bert-base-uncased")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use_sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PyTorch SDPA for GT-MHA when attention weights are not requested.",
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


def precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model(
    method: str,
    model_name_or_path: str,
    device: torch.device,
    use_sdpa: bool,
):
    attention_type = METHODS[method]
    model, audit = load_bert_masked_lm(
        model_name_or_path,
        attention_type=attention_type,
        initialization="random",
        num_kv_heads=4,
        num_base_heads=4,
        num_generators=8,
        generator_mixing="softmax",
        theta_init="random_sphere",
        theta_init_scale=0.02,
        generator_init_scale=0.02,
        use_sdpa=use_sdpa,
        fuse_base_qkv=True,
        sdpa_gqa_mode="auto",
        enforce_paper_gt_mha=True,
    )
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    return model, audit


def make_inputs(
    model,
    batch_size: int,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(
        999,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=device,
    )
    labels = input_ids.clone()
    selected = torch.rand(
        (batch_size, sequence_length), generator=generator, device=device
    ) < 0.15
    labels.masked_fill_(~selected, -100)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_optimizer(model) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": 0.0, "betas": (0.9, 0.999), "eps": 1e-8}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


def optimizer_step(
    model,
    optimizer,
    inputs: dict[str, torch.Tensor],
    accumulation_steps: int,
    device: torch.device,
    precision: str,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for _ in range(accumulation_steps):
        with precision_context(device, precision):
            loss = model(**inputs).loss / accumulation_steps
        scaler.scale(loss).backward()
        loss_value += float(loss.detach())
    scaler.step(optimizer)
    scaler.update()
    return loss_value


def profile_step_flops(step) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return int(sum(event.flops or 0 for event in profile.key_averages()))


def benchmark_method(args: argparse.Namespace, method: str, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model, audit = build_model(
        method,
        args.model_name_or_path,
        device,
        use_sdpa=args.use_sdpa,
    )
    inputs = make_inputs(model, args.batch_size, args.sequence_length, args.seed, device)
    optimizer = make_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.precision == "fp16")
    )

    def step() -> float:
        return optimizer_step(
            model,
            optimizer,
            inputs,
            args.gradient_accumulation_steps,
            device,
            args.precision,
            scaler,
        )

    for _ in range(args.warmup):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    loss = 0.0
    for _ in range(args.repeats):
        synchronize(device)
        started = time.perf_counter()
        loss = step()
        synchronize(device)
        durations.append(time.perf_counter() - started)

    median = statistics.median(durations)
    tokens_per_optimizer_step = (
        args.batch_size * args.sequence_length * args.gradient_accumulation_steps
    )
    result: dict[str, Any] = {
        "method": method,
        "attention_type": METHODS[method],
        "parameter_counts": bert_parameter_counts(model),
        "replacement_audit": audit,
        "batch_size_per_device": args.batch_size,
        "sequence_length": args.sequence_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "raw_step_seconds": durations,
        "median_step_ms": median * 1000.0,
        "p95_step_ms": percentile(durations, 0.95) * 1000.0,
        "tokens_per_second": tokens_per_optimizer_step / max(median, 1e-12),
        "sequences_per_second": (
            args.batch_size * args.gradient_accumulation_steps / max(median, 1e-12)
        ),
        "last_loss": loss,
        "gradient_checkpointing": True,
        "requested_gt_mha_sdpa": args.use_sdpa,
        "model_attention_implementation": getattr(
            model.config, "_attn_implementation", None
        ),
        "attention_module_classes": sorted(
            {
                layer.attention.self.__class__.__name__
                for layer in model.bert.encoder.layer
            }
        ),
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
        )
    if args.profile_flops:
        result["profiled_operator_flops_per_optimizer_step"] = profile_step_flops(step)
        result["profiled_flops_note"] = (
            "PyTorch operator-reported FLOPs; unsupported fused/custom operations may be omitted."
        )

    del inputs, optimizer, scaler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)
    return result


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.sequence_length,
        args.gradient_accumulation_steps,
        args.repeats,
    ) <= 0 or args.warmup < 0:
        raise SystemExit("batch/length/accumulation/repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    results = [benchmark_method(args, method, device) for method in args.methods]
    payload = {
        "schema_version": 1,
        "protocol": "bert_matched_single_gpu_training_v2",
        "methods_in_execution_order": args.methods,
        "precision": args.precision,
        "requested_gt_mha_sdpa": args.use_sdpa,
        "warmup_optimizer_steps": args.warmup,
        "timed_optimizer_steps": args.repeats,
        "fixed_input_seed": args.seed,
        "hardware": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        },
        "results": results,
        "notes": [
            "All methods use the same fixed synthetic token and mask tensors.",
            "Model construction and input allocation are outside timed regions.",
            "The benchmark uses one GPU to isolate per-device implementation efficiency; it is not a four-GPU scaling result.",
            "Throughput includes forward, backward, and AdamW update time with two accumulated microbatches.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
