from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.accounting import attention_accounting, count_parameters
from lgma.attention import LieGeneratedMetricAttention
from lgma.diagnostics import (
    attention_cosine_similarity,
    centered_attention_cosine_similarity,
    grouped_gradient_norms,
    grouped_similarity_stats,
    induced_bilinear_forms,
    induced_metric_cosine_similarity,
    mean_off_diagonal,
    metric_cosine_similarity,
    metric_delta_cosine_similarity,
    metric_diversity_loss,
    metric_distance_from_identity,
    score_cosine_similarity,
)
from lgma.synthetic import CharTokenizer, make_lm_batch
from lgma.tracking import finish_wandb, init_wandb_run, log_wandb
from lgma.transformer import LGMA_ATTENTION_TYPES, TinyTransformerLM, load_model_config


ATTENTION_TYPES = [
    "mha",
    "mqa",
    "gqa",
    "shared_identity",
    "lgma",
    "lgma_v2",
    "lgma_residual",
    "lgma_unconstrained",
    "lgma_value_diag",
    "lgma_multibase",
    "lgma_multibase_value_diag",
]
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
    parser.add_argument(
        "--no_ddp",
        action="store_true",
        help="Disable automatic DDP even when launched with torchrun.",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--base_dim", type=int, default=None)
    parser.add_argument("--value_dim", type=int, default=None)
    parser.add_argument("--num_base_heads", type=int, default=None)
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
    parser.add_argument("--metric_mode", choices=["exp", "residual", "unconstrained"], default=None)
    parser.add_argument("--metric_beta", type=float, default=None)
    parser.add_argument("--theta_init", choices=["random_sphere", "circle"], default=None)
    parser.add_argument("--logit_scale_mode", choices=["sqrt_dim", "rms_metric"], default=None)
    parser.add_argument(
        "--learn_head_temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--value_transform", choices=["none", "diag"], default=None)
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
        "--induced_metric_diversity_weight",
        type=float,
        default=0.0,
        help="Weight for off-diagonal induced B_h cosine diversity regularization.",
    )
    parser.add_argument(
        "--diagnostic_every",
        type=int,
        default=0,
        help="Emit attention diagnostics every N steps. Final report always includes diagnostics.",
    )
    parser.add_argument(
        "--diagnostic_batches",
        type=int,
        default=1,
        help="Number of batches used for attention diagnostic logging.",
    )
    parser.add_argument(
        "--non_causal",
        action="store_true",
        help="Disable causal masking. Text LM runs are causal by default.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="Enable Weights & Biases logging to this project.",
    )
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument(
        "--wandb_tags",
        default=None,
        help="Comma-separated W&B tags, for example `tinystories,lgma,b2`.",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Use `disabled` to force no W&B logging even if a project is set.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=Path,
        default=None,
        help="Optional W&B run directory. Defaults to output_dir when set.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    return torch.device(device_arg)


def distributed_env_enabled(args: argparse.Namespace) -> bool:
    return not args.no_ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    if not distributed_env_enabled(args):
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise SystemExit("DDP requires CUDA. Launch with GPUs and --device cuda.")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


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


def distributed_mean(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced = reduced / dist.get_world_size()
    return reduced


def effective_attention_config(module) -> dict[str, object]:
    keys = [
        "head_dim",
        "base_dim",
        "value_dim",
        "num_heads",
        "num_base_heads",
        "generated_heads_per_base",
        "num_generators",
        "generator_type",
        "metric_mode",
        "metric_beta",
        "theta_init",
        "logit_scale_mode",
        "learn_head_temperature",
        "value_transform",
    ]
    return {key: getattr(module, key) for key in keys if hasattr(module, key)}


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
        "base_dim": None,
        "value_dim": None,
        "num_base_heads": 1,
        "metric_mode": "exp",
        "metric_beta": 1.0,
        "theta_init": "random_sphere",
        "logit_scale_mode": "sqrt_dim",
        "learn_head_temperature": False,
        "value_transform": "none",
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
        "base_dim": args.base_dim,
        "value_dim": args.value_dim,
        "num_base_heads": args.num_base_heads,
        "context_length": args.context_length,
        "num_kv_heads": args.num_kv_heads,
        "dropout": args.dropout,
        "theta_init_scale": args.theta_init_scale,
        "generator_init_scale": args.generator_init_scale,
        "metric_mode": args.metric_mode,
        "metric_beta": args.metric_beta,
        "theta_init": args.theta_init,
        "logit_scale_mode": args.logit_scale_mode,
        "learn_head_temperature": args.learn_head_temperature,
        "value_transform": args.value_transform,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    merged["causal"] = not args.non_causal
    if merged["attention_type"] not in LGMA_ATTENTION_TYPES:
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


def compute_diversity_regularizers(
    model: TinyTransformerLM,
    metric_weight: float,
    induced_weight: float,
    squared: bool,
    use_delta: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    lgma_layers = [
        module for module in model.modules() if isinstance(module, LieGeneratedMetricAttention)
    ]
    if not lgma_layers:
        zero = torch.zeros((), device=device)
        return zero, zero
    metric_loss = torch.zeros((), device=device)
    induced_loss = torch.zeros((), device=device)
    if metric_weight != 0.0:
        metric_loss = torch.stack(
            [
                metric_diversity_loss(
                    module.compute_metrics(),
                    squared=squared,
                    use_delta=use_delta,
                )
                for module in lgma_layers
            ]
        ).mean()
    if induced_weight != 0.0:
        induced_loss = torch.stack(
            [
                metric_diversity_loss(
                    induced_bilinear_forms(module),
                    squared=squared,
                    use_delta=False,
                )
                for module in lgma_layers
            ]
        ).mean()
    return metric_loss, induced_loss


def add_attention_diagnostics(
    report: dict[str, object],
    model: TinyTransformerLM,
    encoded: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    diagnostic_batches: int,
) -> None:
    model = unwrap_model(model)
    first_attn = model.first_attention
    with torch.no_grad():
        attention_sims = []
        centered_sims = []
        score_sims = []
        for _ in range(max(1, diagnostic_batches)):
            batch = make_lm_batch(encoded, batch_size, seq_len, device=device)
            x = model.blocks[0].norm1(
                model.token_embedding(batch.input_ids)
                + model.position_embedding(torch.arange(seq_len, device=device))[None, :, :]
            )
            _, attn = first_attn(x, need_weights=True)
            attention_sims.append(attention_cosine_similarity(attn))
            centered_sims.append(centered_attention_cosine_similarity(attn))
            if hasattr(first_attn, "compute_scores"):
                q, k, _ = first_attn._project(x)
                score_sims.append(score_cosine_similarity(first_attn.compute_scores(q, k)))

        attention_similarity = torch.stack(attention_sims).mean(dim=0)
        centered_similarity = torch.stack(centered_sims).mean(dim=0)
        report["attention_diversity_mean_cosine"] = float(attention_similarity.mean())
        report["attention_diversity_offdiag_mean_cosine"] = float(
            mean_off_diagonal(attention_similarity)
        )
        report["centered_attention_diversity_mean_cosine"] = float(
            centered_similarity.mean()
        )
        report["centered_attention_diversity_offdiag_mean_cosine"] = float(
            mean_off_diagonal(centered_similarity)
        )
        if hasattr(first_attn, "num_base_heads") and first_attn.num_base_heads > 1:
            report.update(
                {
                    f"attention_{key}": float(value)
                    for key, value in grouped_similarity_stats(
                        attention_similarity,
                        first_attn.num_base_heads,
                        first_attn.generated_heads_per_base,
                    ).items()
                }
            )
            report.update(
                {
                    f"centered_attention_{key}": float(value)
                    for key, value in grouped_similarity_stats(
                        centered_similarity,
                        first_attn.num_base_heads,
                        first_attn.generated_heads_per_base,
                    ).items()
                }
            )
        if score_sims:
            score_similarity = torch.stack(score_sims).mean(dim=0)
            report["score_diversity_mean_cosine"] = float(score_similarity.mean())
            report["score_diversity_offdiag_mean_cosine"] = float(
                mean_off_diagonal(score_similarity)
            )
            if hasattr(first_attn, "num_base_heads") and first_attn.num_base_heads > 1:
                report.update(
                    {
                        f"score_{key}": float(value)
                        for key, value in grouped_similarity_stats(
                            score_similarity,
                            first_attn.num_base_heads,
                            first_attn.generated_heads_per_base,
                        ).items()
                    }
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
            identity_distance = metric_distance_from_identity(metrics)
            report["metric_identity_distance_mean"] = float(identity_distance.mean())
            report["metric_identity_distance_max"] = float(identity_distance.max())
            induced_similarity = induced_metric_cosine_similarity(first_attn, metrics)
            report["induced_metric_diversity_mean_cosine"] = float(induced_similarity.mean())
            report["induced_metric_diversity_offdiag_mean_cosine"] = float(
                mean_off_diagonal(induced_similarity)
            )
            if hasattr(first_attn, "num_base_heads") and first_attn.num_base_heads > 1:
                report.update(
                    {
                        f"metric_{key}": float(value)
                        for key, value in grouped_similarity_stats(
                            metric_similarity,
                            first_attn.num_base_heads,
                            first_attn.generated_heads_per_base,
                        ).items()
                    }
                )
                report.update(
                    {
                        f"induced_metric_{key}": float(value)
                        for key, value in grouped_similarity_stats(
                            induced_similarity,
                            first_attn.num_base_heads,
                            first_attn.generated_heads_per_base,
                        ).items()
                    }
                )


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    config: dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_model = unwrap_model(model)
    payload = {
        "step": step,
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "model_config": config,
        "args": vars(args),
    }
    torch.save(payload, path)
    print(json.dumps({"event": "checkpoint_saved", "step": step, "path": str(path)}))


def load_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
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
    final_induced_diversity_loss: float,
    gradient_norms: dict[str, float],
) -> dict[str, object]:
    seq_len = int(config["context_length"])
    base_model = unwrap_model(model)
    first_attn = base_model.first_attention
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
        "parameters": count_parameters(base_model),
        "effective_attention_config": effective_attention_config(first_attn),
        "attention_accounting": attention_accounting(
            first_attn, sequence_length=seq_len, batch_size=args.batch_size
        ).__dict__,
        "final_loss": final_loss,
        "final_perplexity": math.exp(min(float(final_loss), 20.0)),
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        "final_metric_diversity_loss": final_diversity_loss,
        "final_induced_metric_diversity_loss": final_induced_diversity_loss,
        "metric_diversity_weight": args.metric_diversity_weight,
        "induced_metric_diversity_weight": args.induced_metric_diversity_weight,
        "metric_diversity_squared": args.metric_diversity_squared,
        "metric_diversity_on_delta": not args.metric_diversity_on_full_metric,
        "device": str(device),
        "precision": args.precision,
        "grad_accum_steps": args.grad_accum_steps,
        "gradient_norms": gradient_norms,
    }
    add_attention_diagnostics(
        report,
        base_model,
        val_encoded,
        args.batch_size,
        seq_len,
        device,
        args.diagnostic_batches,
    )
    return report


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad_accum_steps must be positive")
    if args.log_every < 0 or args.eval_every < 0 or args.save_every < 0:
        raise SystemExit("--log_every, --eval_every, and --save_every must be non-negative")

    ddp_enabled, rank, local_rank, world_size = setup_distributed(args)
    main_process = is_main_process(rank)
    try:
        torch.manual_seed(args.seed + rank)
        if ddp_enabled:
            if args.device != "cuda":
                raise SystemExit("DDP requires --device cuda")
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
        if args.compile and ddp_enabled:
            raise SystemExit("--compile is not enabled for DDP in this runner")

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
        base_model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config).to(device)
        if args.compile:
            if not hasattr(torch, "compile"):
                raise SystemExit("--compile requires a PyTorch version with torch.compile")
            base_model = torch.compile(base_model)
        if ddp_enabled:
            model = DistributedDataParallel(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=(
                    args.metric_diversity_weight != 0.0
                    or args.induced_metric_diversity_weight != 0.0
                ),
            )
        else:
            model = base_model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scaler = make_grad_scaler(device, args.precision)

        last_loss = None
        last_diversity_loss = 0.0
        last_induced_diversity_loss = 0.0
        last_grad_norms: dict[str, float] = {}
        use_delta = not args.metric_diversity_on_full_metric

        output_dir = args.output_dir
        metrics_path = output_dir / "metrics.jsonl" if output_dir is not None else None
        if output_dir is not None and main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = None
        if main_process:
            first_attn = unwrap_model(model).first_attention
            wandb_run = init_wandb_run(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                mode=args.wandb_mode,
                output_dir=args.wandb_dir or output_dir,
                config={
                    "args": vars(args),
                    "model_config": config,
                    "effective_attention_config": effective_attention_config(first_attn),
                    "attention_accounting": attention_accounting(
                        first_attn,
                        sequence_length=seq_len,
                        batch_size=args.batch_size,
                    ).__dict__,
                    "parameters": count_parameters(unwrap_model(model)),
                    "vocab_size": tokenizer.vocab_size,
                    "train_characters": int(train_encoded.numel()),
                    "validation_characters": int(val_encoded.numel()),
                    "distributed": ddp_enabled,
                    "world_size": world_size,
                },
            )

        start_step = 0
        if args.resume_checkpoint is not None:
            start_step = load_checkpoint(args.resume_checkpoint, model, optimizer, scaler, device)
        if ddp_enabled:
            dist.barrier()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        last_log_time = time.perf_counter()
        last_log_step = start_step
        tokens_per_step = args.batch_size * seq_len * args.grad_accum_steps * world_size

        for step_idx in range(start_step, args.steps):
            step = step_idx + 1
            optimizer.zero_grad(set_to_none=True)

            task_losses = []
            diversity_losses = []
            induced_diversity_losses = []
            for accum_idx in range(args.grad_accum_steps):
                batch = make_lm_batch(train_encoded, args.batch_size, seq_len, device=device)
                sync_context = (
                    model.no_sync()
                    if ddp_enabled and accum_idx < args.grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context(device, args.precision):
                        _, task_loss = model(batch.input_ids, batch.targets)
                        diversity_loss, induced_diversity_loss = compute_diversity_regularizers(
                            model,
                            metric_weight=args.metric_diversity_weight,
                            induced_weight=args.induced_metric_diversity_weight,
                            squared=args.metric_diversity_squared,
                            use_delta=use_delta,
                            device=device,
                        )
                        loss = (
                            task_loss
                            + args.metric_diversity_weight * diversity_loss
                            + args.induced_metric_diversity_weight * induced_diversity_loss
                        )
                        loss = loss / args.grad_accum_steps
                    scaler.scale(loss).backward()
                task_losses.append(task_loss.detach())
                diversity_losses.append(diversity_loss.detach())
                induced_diversity_losses.append(induced_diversity_loss.detach())

            if device.type == "cuda" and args.precision == "fp16":
                scaler.unscale_(optimizer)
            first_attention = unwrap_model(model).first_attention
            if isinstance(first_attention, LieGeneratedMetricAttention):
                last_grad_norms = grouped_gradient_norms(first_attention)
            if args.max_grad_norm > 0:
                if not (device.type == "cuda" and args.precision == "fp16"):
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            mean_task_loss = distributed_mean(torch.stack(task_losses).mean(), ddp_enabled)
            mean_diversity_loss = distributed_mean(
                torch.stack(diversity_losses).mean(), ddp_enabled
            )
            mean_induced_diversity_loss = distributed_mean(
                torch.stack(induced_diversity_losses).mean(), ddp_enabled
            )
            last_loss = float(mean_task_loss.cpu())
            last_diversity_loss = float(mean_diversity_loss.cpu())
            last_induced_diversity_loss = float(mean_induced_diversity_loss.cpu())

            should_log = args.log_every > 0 and (step == 1 or step % args.log_every == 0)
            should_eval = args.eval_every > 0 and step % args.eval_every == 0
            should_diagnose = args.diagnostic_every > 0 and step % args.diagnostic_every == 0
            is_final = step == args.steps

            if (should_log or should_eval or should_diagnose or is_final) and main_process:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-12)
                steps_since_log = max(step - last_log_step, 1)
                tokens_per_second = tokens_per_step * steps_since_log / elapsed
                payload: dict[str, object] = {
                    "step": step,
                    "loss": last_loss,
                    "perplexity": math.exp(min(last_loss, 20.0)),
                    "metric_diversity_loss": last_diversity_loss,
                    "induced_metric_diversity_loss": last_induced_diversity_loss,
                    "tokens_per_second": tokens_per_second,
                    "lr": optimizer.param_groups[0]["lr"],
                    "rank": rank,
                    "world_size": world_size,
                }
                payload.update(last_grad_norms)
                if device.type == "cuda":
                    payload["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                if should_eval or is_final:
                    validation_loss = evaluate_text_loss(
                        unwrap_model(model),
                        val_encoded,
                        args.batch_size,
                        seq_len,
                        device,
                        args.eval_batches,
                    )
                    payload["validation_loss"] = validation_loss
                    payload["validation_perplexity"] = math.exp(min(validation_loss, 20.0))
                if should_diagnose:
                    add_attention_diagnostics(
                        payload,
                        unwrap_model(model),
                        val_encoded,
                        args.batch_size,
                        seq_len,
                        device,
                        args.diagnostic_batches,
                    )
                print(json.dumps(payload))
                write_jsonl(metrics_path, payload)
                log_wandb(wandb_run, payload, step=step)
                last_log_time = now
                last_log_step = step

            if (
                args.save_every > 0
                and output_dir is not None
                and step % args.save_every == 0
                and main_process
            ):
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

        if main_process:
            report = build_report(
                args,
                config,
                unwrap_model(model),
                tokenizer,
                train_encoded,
                val_encoded,
                device,
                last_loss,
                last_diversity_loss,
                last_induced_diversity_loss,
                last_grad_norms,
            )
            report["distributed"] = ddp_enabled
            report["world_size"] = world_size
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
            log_wandb(wandb_run, {"final": report}, step=args.steps)
            print(json.dumps(report, indent=2))
            finish_wandb(wandb_run)
            wandb_run = None
        if ddp_enabled:
            dist.barrier()
    finally:
        if "wandb_run" in locals() and wandb_run is not None:
            finish_wandb(wandb_run)
        cleanup_distributed(ddp_enabled if "ddp_enabled" in locals() else False)


if __name__ == "__main__":
    main()
