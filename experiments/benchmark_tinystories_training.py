from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.checkpointing import load_full_checkpoint
from lgma.transformer import TinyTransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark checkpoint-resumed TinyStories training steps without "
            "writing checkpoints or changing the source run."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--fuse_base_qkv", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fold_value_transform_into_output",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sdpa_gqa_mode", choices=["auto", "native", "expand"], default="auto"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="default",
    )
    return parser.parse_args()


def precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("batch_size/repeats must be positive and warmup non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    checkpoint = load_full_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(config, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint must contain model_config and model_state")
    config = dict(config)
    config.update(
        {
            "fuse_base_qkv": args.fuse_base_qkv,
            "fold_value_transform_into_output": args.fold_value_transform_into_output,
            "sdpa_gqa_mode": args.sdpa_gqa_mode,
        }
    )
    trained_context = int(config["context_length"])
    context_length = args.context_length or trained_context
    if context_length <= 0 or context_length > trained_context:
        raise SystemExit(f"context_length must be in [1, {trained_context}]")
    token_weight = model_state.get("token_embedding.weight")
    if not torch.is_tensor(token_weight):
        raise SystemExit("checkpoint model_state is missing token_embedding.weight")
    vocab_size = int(token_weight.shape[0])

    model = TinyTransformerLM(vocab_size=vocab_size, **config).to(device)
    model.load_state_dict(model_state)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = 0.0
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.precision == "fp16")
    )
    if args.compile:
        model.compile(backend="inductor", mode=args.compile_mode)

    generator = torch.Generator(device=device).manual_seed(20_260_813)
    input_ids = torch.randint(
        vocab_size,
        (args.batch_size, context_length),
        device=device,
        generator=generator,
    )
    targets = torch.randint(
        vocab_size,
        (args.batch_size, context_length),
        device=device,
        generator=generator,
    )

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, args.precision):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

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

    median_seconds = statistics.median(durations)
    tokens_per_step = args.batch_size * context_length
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "batch_size": args.batch_size,
        "context_length": context_length,
        "precision": args.precision,
        "warmup_steps": args.warmup,
        "measured_steps": args.repeats,
        "median_step_ms": median_seconds * 1000.0,
        "mean_step_ms": statistics.mean(durations) * 1000.0,
        "tokens_per_second": tokens_per_step / max(median_seconds, 1e-12),
        "loss": loss,
        "fuse_base_qkv": args.fuse_base_qkv,
        "fold_value_transform_into_output": args.fold_value_transform_into_output,
        "sdpa_gqa_mode": args.sdpa_gqa_mode,
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
    }
    if device.type == "cuda":
        result["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
