from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.accounting import attention_accounting, count_parameters
from lgma.attention import LieGeneratedMetricAttention
from lgma.diagnostics import (
    attention_cosine_similarity,
    mean_off_diagonal,
    metric_cosine_similarity,
    metric_delta_cosine_similarity,
    metric_diversity_loss,
)
from lgma.synthetic import CharTokenizer, make_lm_batch
from lgma.transformer import TinyTransformerLM, load_model_config


ATTENTION_TYPES = ["mha", "mqa", "gqa", "shared_identity", "lgma"]
GENERATOR_TYPES = ["full", "diagonal", "symmetric"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Character-level text LM runner with LGMA/MHA diagnostics."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--val_data_path",
        default=None,
        help="Optional validation text file. If omitted, validation batches use data_path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config. CLI model args override config values.",
    )
    parser.add_argument("--attention", choices=ATTENTION_TYPES, default=None)
    parser.add_argument("--generator_type", choices=GENERATOR_TYPES, default=None)
    parser.add_argument("--num_generators", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--theta_init_scale",
        type=float,
        default=None,
        help="LGMA head-coordinate initialization scale.",
    )
    parser.add_argument(
        "--generator_init_scale",
        type=float,
        default=None,
        help="LGMA generator initialization scale before division by sqrt(head_dim).",
    )
    parser.add_argument(
        "--metric_diversity_weight",
        type=float,
        default=0.0,
        help="Weight for off-diagonal metric cosine diversity regularization.",
    )
    parser.add_argument(
        "--metric_diversity_squared",
        action="store_true",
        help="Use squared off-diagonal metric cosine regularization.",
    )
    parser.add_argument(
        "--metric_diversity_on_full_metric",
        action="store_true",
        help="Regularize full M_h cosine instead of default delta metric cosine over M_h - I.",
    )
    parser.add_argument(
        "--non_causal",
        action="store_true",
        help="Disable causal masking. Text LM runs are causal by default.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    return torch.device(device_arg)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.cuda.amp.GradScaler(enabled=enabled)


def write_jsonl(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.config is None:
        config: dict[str, object] = {}
    else:
        config = load_model_config(args.config)

    defaults: dict[str, object] = {
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 8,
        "head_dim": 16,
        "attention_type": "lgma",
        "num_generators": 4,
        "generator_type": "full",
        "context_length": 128,
        "dropout": 0.0,
        "num_kv_heads": None,
        "causal": not args.non_causal,
        "theta_init_scale": 0.02,
        "generator_init_scale": 0.02,
    }
    merged = {**defaults, **config}

    overrides = {
        "attention_type": args.attention,
        "generator_type": args.generator_type,
        "num_generators": args.num_generators,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "context_length": args.context_length,
        "num_kv_heads": args.num_kv_heads,
        "dropout": args.dropout,
        "theta_init_scale": args.theta_init_scale,
        "generator_init_scale": args.generator_init_scale,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    merged["causal"] = not args.non_causal
    if merged["attention_type"] != "lgma":
        merged["num_generators"] = 0
    return merged


def build_tokenizer(train_text: str, val_text: str | None) -> CharTokenizer:
    if val_text is None:
        return CharTokenizer(train_text)
    return CharTokenizer(train_text + val_text)


@torch.no_grad()
def evaluate_text_loss(
    model: TinyTransformerLM,
    encoded: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        batch = make_lm_batch(encoded, batch_size, seq_len, device=device)
        _, loss = model(batch.input_ids, batch.targets)
        losses.append(loss.detach())
    model.train()
    return float(torch.stack(losses).mean().cpu())


def compute_metric_diversity_regularizer(
    model: TinyTransformerLM,
    squared: bool,
    use_delta: bool,
    device: torch.device,
) -> torch.Tensor:
    lgma_layers = [
        module for module in model.modules() if isinstance(module, LieGeneratedMetricAttention)
    ]
    if not lgma_layers:
        return torch.zeros((), device=device)
    losses = [
        metric_diversity_loss(
            module.compute_metrics(),
            squared=squared,
            use_delta=use_delta,
        )
        for module in lgma_layers
    ]
    return torch.stack(losses).mean()


def add_attention_diagnostics(report: dict[str, object], model: TinyTransformerLM, batch) -> None:
    first_attn = model.first_attention
    with torch.no_grad():
        seq_len = batch.input_ids.shape[1]
        device = batch.input_ids.device
        x = model.blocks[0].norm1(
            model.token_embedding(batch.input_ids)
            + model.position_embedding(torch.arange(seq_len, device=device))[None, :, :]
        )
        _, attn = first_attn(x, need_weights=True)
        attention_similarity = attention_cosine_similarity(attn)
        report["attention_diversity_mean_cosine"] = float(attention_similarity.mean())
        report["attention_diversity_offdiag_mean_cosine"] = float(
            mean_off_diagonal(attention_similarity)
        )
        if hasattr(first_attn, "compute_metrics"):
            metrics = first_attn.compute_metrics()
            metric_similarity = metric_cosine_similarity(metrics)
            metric_delta_similarity = metric_delta_cosine_similarity(metrics)
            report["metric_diversity_mean_cosine"] = float(metric_similarity.mean())
            report["metric_diversity_offdiag_mean_cosine"] = float(
                mean_off_diagonal(metric_similarity)
            )
            report["metric_delta_diversity_mean_cosine"] = float(
                metric_delta_similarity.mean()
            )
            report["metric_delta_diversity_offdiag_mean_cosine"] = float(
                mean_off_diagonal(metric_delta_similarity)
            )


def save_checkpoint(
    path: Path,
    model: TinyTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    config: dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "model_config": config,
        "args": vars(args),
    }
    torch.save(payload, path)
    print(json.dumps({"event": "checkpoint_saved", "step": step, "path": str(path)}))


def load_checkpoint(
    path: Path,
    model: TinyTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    step = int(checkpoint.get("step", 0))
    print(json.dumps({"event": "checkpoint_loaded", "step": step, "path": str(path)}))
    return step


def build_report(
    args: argparse.Namespace,
    config: dict[str, object],
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    train_encoded: torch.Tensor,
    val_encoded: torch.Tensor,
    device: torch.device,
    final_loss: float,
    final_diversity_loss: float,
) -> dict[str, object]:
    seq_len = int(config["context_length"])
    first_attn = model.first_attention
    validation_loss = evaluate_text_loss(
        model,
        val_encoded,
        args.batch_size,
        seq_len,
        device,
        args.eval_batches,
    )
    report: dict[str, object] = {
        "data_path": args.data_path,
        "val_data_path": args.val_data_path,
        "config": args.config,
        "model_config": config,
        "vocab_size": tokenizer.vocab_size,
        "train_characters": int(train_encoded.numel()),
        "validation_characters": int(val_encoded.numel()),
        "parameters": count_parameters(model),
        "attention_accounting": attention_accounting(
            first_attn, sequence_length=seq_len, batch_size=args.batch_size
        ).__dict__,
        "final_loss": final_loss,
        "final_perplexity": math.exp(min(float(final_loss), 20.0)),
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        "final_metric_diversity_loss": final_diversity_loss,
        "metric_diversity_weight": args.metric_diversity_weight,
        "metric_diversity_squared": args.metric_diversity_squared,
        "metric_diversity_on_delta": not args.metric_diversity_on_full_metric,
        "device": str(device),
        "precision": args.precision,
        "grad_accum_steps": args.grad_accum_steps,
    }
    diagnostic_batch = make_lm_batch(val_encoded, args.batch_size, seq_len, device=device)
    add_attention_diagnostics(report, model, diagnostic_batch)
    return report


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad_accum_steps must be positive")
    if args.log_every < 0 or args.eval_every < 0 or args.save_every < 0:
        raise SystemExit("--log_every, --eval_every, and --save_every must be non-negative")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_text = Path(args.data_path).read_text(encoding="utf-8")
    val_text = (
        Path(args.val_data_path).read_text(encoding="utf-8")
        if args.val_data_path is not None
        else None
    )
    tokenizer = build_tokenizer(train_text, val_text)
    train_encoded = tokenizer.encode(train_text)
    val_encoded = tokenizer.encode(val_text) if val_text is not None else train_encoded

    config = model_config_from_args(args)
    seq_len = int(config["context_length"])
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config).to(device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise SystemExit("--compile requires a PyTorch version with torch.compile")
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device, args.precision)

    last_loss = None
    last_diversity_loss = 0.0
    use_delta = not args.metric_diversity_on_full_metric

    output_dir = args.output_dir
    metrics_path = output_dir / "metrics.jsonl" if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = load_checkpoint(args.resume_checkpoint, model, optimizer, scaler, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    last_log_time = time.perf_counter()
    last_log_step = start_step
    tokens_per_step = args.batch_size * seq_len * args.grad_accum_steps

    for step_idx in range(start_step, args.steps):
        step = step_idx + 1
        optimizer.zero_grad(set_to_none=True)

        task_losses = []
        diversity_losses = []
        for _ in range(args.grad_accum_steps):
            batch = make_lm_batch(train_encoded, args.batch_size, seq_len, device=device)
            with autocast_context(device, args.precision):
                _, task_loss = model(batch.input_ids, batch.targets)
                diversity_loss = torch.zeros((), device=device)
                if args.metric_diversity_weight != 0.0:
                    diversity_loss = compute_metric_diversity_regularizer(
                        model,
                        squared=args.metric_diversity_squared,
                        use_delta=use_delta,
                        device=device,
                    )
                loss = task_loss + args.metric_diversity_weight * diversity_loss
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            task_losses.append(task_loss.detach())
            diversity_losses.append(diversity_loss.detach())

        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(torch.stack(task_losses).mean().cpu())
        last_diversity_loss = float(torch.stack(diversity_losses).mean().cpu())

        should_log = args.log_every > 0 and (step == 1 or step % args.log_every == 0)
        should_eval = args.eval_every > 0 and step % args.eval_every == 0
        is_final = step == args.steps

        if should_log or should_eval or is_final:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-12)
            steps_since_log = max(step - last_log_step, 1)
            tokens_per_second = tokens_per_step * steps_since_log / elapsed
            payload: dict[str, object] = {
                "step": step,
                "loss": last_loss,
                "perplexity": math.exp(min(last_loss, 20.0)),
                "metric_diversity_loss": last_diversity_loss,
                "tokens_per_second": tokens_per_second,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if device.type == "cuda":
                payload["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            if should_eval or is_final:
                validation_loss = evaluate_text_loss(
                    model,
                    val_encoded,
                    args.batch_size,
                    seq_len,
                    device,
                    args.eval_batches,
                )
                payload["validation_loss"] = validation_loss
                payload["validation_perplexity"] = math.exp(min(validation_loss, 20.0))
            print(json.dumps(payload))
            write_jsonl(metrics_path, payload)
            last_log_time = now
            last_log_step = step

        if args.save_every > 0 and output_dir is not None and step % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_step_{step}.pt",
                model,
                optimizer,
                scaler,
                step,
                config,
                args,
            )

    if last_loss is None:
        raise SystemExit("--steps must be greater than resume checkpoint step")

    report = build_report(
        args,
        config,
        model,
        tokenizer,
        train_encoded,
        val_encoded,
        device,
        last_loss,
        last_diversity_loss,
    )
    if device.type == "cuda":
        report["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    if output_dir is not None:
        final_report_path = output_dir / "final_report.json"
        final_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_checkpoint(
            output_dir / "checkpoint_final.pt",
            model,
            optimizer,
            scaler,
            args.steps,
            config,
            args,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
