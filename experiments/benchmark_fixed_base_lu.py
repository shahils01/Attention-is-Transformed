from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma import attention as attention_module
from lgma.lu_kernel import triton
from lgma.transformer import TinyTransformerLM, build_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched fused and unfused fixed-base LU attention training on CUDA.",
    )
    parser.add_argument("--scope", choices=("layer", "model"), default="layer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--model-width", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--bases", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12, help="Transformer layers in model scope")
    parser.add_argument("--vocab-size", type=int, default=230, help="synthetic vocabulary in model scope")
    parser.add_argument("--dropout", type=float, help="defaults to 0.1 for model scope and 0 for layer scope")
    parser.add_argument("--lu-init-scale", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=10, help="warmup steps per variant")
    parser.add_argument("--repeats", type=int, default=40, help="measured steps per variant")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fuse-base-qkv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sdpa-gqa-mode", choices=("auto", "native", "expand"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.model_width, args.head_dim, args.bases) <= 0:
        parser.error("batch size, sequence length, model width, head dim, and bases must be positive")
    if args.heads <= args.bases or args.heads % args.bases != 0:
        parser.error("heads must be a multiple of bases with at least two heads per base")
    if not math.isfinite(args.lu_init_scale) or args.lu_init_scale < 0:
        parser.error("lu init scale must be finite and nonnegative")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("warmup must be positive and repeats must be at least two")
    if args.layers < 1 or args.vocab_size < 2:
        parser.error("layers must be positive and vocabulary must have at least two entries")
    if args.dropout is None:
        args.dropout = 0.1 if args.scope == "model" else 0.0
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    return args


def layer_step(layer, inputs, upstream, dtype):
    layer.zero_grad(set_to_none=True)
    inputs.grad = None
    with torch.autocast(inputs.device.type, dtype=dtype):
        output = layer(inputs)
    output.backward(upstream)
    return {"output": output.detach(), "input_grad": inputs.grad}


def model_step(model, inputs, targets, optimizer, scaler, dtype):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(inputs.device.type, dtype=dtype):
        logits, loss = model(inputs, targets=targets)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return {"output": logits.detach(), "loss": loss.detach()}


def verify_dispatch_and_parity(modules, steps, dtype, device, expected_fusions):
    dispatch = {"fused": {"forward": 0, "backward": 0}, "unfused": {"forward": 0, "backward": 0}}
    implementation = attention_module.fused_lu_queries

    def run_verified(label):
        def backward_hook(grad_inputs, grad_outputs):
            dispatch[label]["backward"] += 1

        def forward_hook(*args, **kwargs):
            result = implementation(*args, **kwargs)
            if result is not None:
                node = result.grad_fn
                if node is None or "FusedLUQueries" not in node.name():
                    raise RuntimeError("CUDA LU expansion did not produce the expected custom autograd operation")
                dispatch[label]["forward"] += 1
                node.register_hook(backward_hook)
            return result

        with patch.object(attention_module, "fused_lu_queries", forward_hook):
            return steps[label]()

    rng_state = torch.cuda.get_rng_state(device)
    actual = run_verified("fused")
    torch.cuda.set_rng_state(rng_state, device)
    expected = run_verified("unfused")
    expected_dispatch = {
        "fused": {"forward": expected_fusions, "backward": expected_fusions},
        "unfused": {"forward": 0, "backward": 0},
    }
    if dispatch != expected_dispatch:
        raise RuntimeError(f"Expected CUDA LU dispatch {expected_dispatch}. Observed {dispatch}.")
    if actual["output"].dtype != dtype or expected["output"].dtype != dtype:
        raise RuntimeError(f"Expected both attention outputs in {dtype}")

    gradients = {
        label: {name: parameter.grad for name, parameter in layer.named_parameters()}
        for label, layer in modules.items()
    }
    pairs = {
        **{name: (result, expected[name]) for name, result in actual.items()},
        **{
            f"{name}.grad": (grad, gradients["unfused"][name])
            for name, grad in gradients["fused"].items()
        },
    }
    tolerance = 0.035 if dtype == torch.bfloat16 else 0.012
    errors = {}
    for name, (result, reference) in pairs.items():
        if result is None or reference is None:
            raise RuntimeError(f"Missing gradient for {name}")
        result, reference = result.float(), reference.float()
        norm = reference.norm().item()
        error = (result - reference).norm().item() / max(norm, torch.finfo(torch.float32).tiny)
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError(f"Fused LU parity failed for {name}. Relative L2 error {error:.6g} exceeds {tolerance}.")
        errors[name] = error
    return {"dispatch": dispatch, "relative_l2_error": errors, "relative_l2_tolerance": tolerance}


def benchmark(args: argparse.Namespace, device: torch.device) -> dict:
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    settings = {
        "d_model": args.model_width,
        "num_heads": args.heads,
        "num_base_heads": args.bases,
        "head_dim": args.head_dim,
        "num_generators": 0,
        "metric_mode": "lu",
        "value_transform": "none",
        "lu_init_scale": args.lu_init_scale,
        "fuse_base_qkv": args.fuse_base_qkv,
        "sdpa_gqa_mode": args.sdpa_gqa_mode,
        "dropout": args.dropout,
        "causal": True,
    }
    shape = (args.batch_size, args.sequence_length)
    full_model = args.scope == "model"
    build_module = (
        partial(
            TinyTransformerLM, attention_type="gt_mha_lu", vocab_size=args.vocab_size,
            context_length=args.sequence_length, num_layers=args.layers,
        )
        if full_model else partial(build_attention, "gt_mha_lu")
    )
    modules = {
        label: build_module(**settings, fuse_lu_queries=(label == "fused")).to(device).train()
        for label in ("fused", "unfused")
    }
    modules["unfused"].load_state_dict(modules["fused"].state_dict(), strict=True)
    optimizers = {}
    if full_model:
        tokens = torch.randint(args.vocab_size, shape, device=device)
        targets = torch.randint(args.vocab_size, shape, device=device)
        # AdamW builds and updates its moments while all timed trials retain identical weights.
        optimizers = {
            label: torch.optim.AdamW(module.parameters(), lr=0.0)
            for label, module in modules.items()
        }
        scalers = {
            label: (
                torch.cuda.amp.GradScaler(init_scale=128.0, growth_interval=args.warmup + args.repeats + 2)
                if dtype == torch.float16 else None
            )
            for label in modules
        }
        steps = {
            label: partial(model_step, model, tokens, targets, optimizers[label], scalers[label], dtype)
            for label, model in modules.items()
        }
    else:
        shape += (args.model_width,)
        input_data = torch.randn(shape, device=device)
        upstream = torch.randn(shape, device=device, dtype=dtype)
        steps = {
            label: partial(layer_step, layer, input_data.clone().requires_grad_(), upstream, dtype)
            for label, layer in modules.items()
        }
    expected_fusions = args.layers if full_model else 1
    parity = verify_dispatch_and_parity(modules, steps, dtype, device, expected_fusions)

    def run_pair(index, events=None):
        order = ("fused", "unfused") if index % 2 == 0 else ("unfused", "fused")
        rng_state = torch.cuda.get_rng_state(device) if args.dropout else None
        for position, label in enumerate(order):
            if position == 1 and rng_state is not None:
                torch.cuda.set_rng_state(rng_state, device)
            if events is None:
                steps[label]()
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                steps[label]()
                end.record()
                events[label].append((start, end))

    for index in range(args.warmup):
        run_pair(index)
    torch.cuda.synchronize(device)
    events = {label: [] for label in modules}
    for index in range(args.repeats):
        run_pair(index, events)
    torch.cuda.synchronize(device)
    for label, optimizer in optimizers.items():
        parameter = next(modules[label].parameters())
        steps_completed = int(optimizer.state.get(parameter, {}).get("step", 0))
        if steps_completed != 1 + args.warmup + args.repeats:
            raise RuntimeError(f"AdamW skipped training steps in the {label} trial")

    performance = {}
    for label, pairs in events.items():
        milliseconds = [start.elapsed_time(end) for start, end in pairs]
        median = statistics.median(milliseconds)
        performance[label] = {
            "median_ms": median,
            "mean_ms": statistics.mean(milliseconds),
            "p95_ms": sorted(milliseconds)[math.ceil(0.95 * len(milliseconds)) - 1],
            "tokens_per_second": 1000 * args.batch_size * args.sequence_length / median,
            "samples_ms": milliseconds,
        }
    return {
        "scope": args.scope,
        "workload": (
            "causal language model with cross entropy, backward, and AdamW"
            if full_model else "single causal attention layer forward and backward"
        ),
        "attention_type": "gt_mha_lu",
        "input_shape": shape,
        "attention": settings,
        "layers": expected_fusions,
        "vocab_size": args.vocab_size if full_model else None,
        "optimizer": {"name": "AdamW", "lr": 0.0} if full_model else None,
        "precision": args.precision,
        "parameter_dtype": "float32",
        "seed": args.seed,
        "warmup_steps_per_variant": args.warmup,
        "measured_steps_per_variant": args.repeats,
        "total_parameters": sum(parameter.numel() for parameter in modules["fused"].parameters()),
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "parity": parity,
        "timing": performance,
        "speedup": performance["unfused"]["median_ms"] / performance["fused"]["median_ms"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("An NVIDIA CUDA device is required to measure the fused LU path")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise SystemExit("Fixed-base LU fusion requires NVIDIA compute capability 8.0 or greater")
    if triton is None:
        raise SystemExit("Triton is required to measure the fused LU path")
    with torch.cuda.device(device):
        result = benchmark(args, device)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
