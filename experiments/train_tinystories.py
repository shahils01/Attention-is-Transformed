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
from lgma.checkpointing import load_full_checkpoint
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
    metric_condition_number,
    metric_distance_from_identity,
    metric_singular_values,
    score_cosine_similarity,
)
from lgma.packed_data import PackedTokenSplit, load_packed_token_corpus
from lgma.synthetic import CharTokenizer, SyntheticBatch, make_lm_batch
from lgma.tracking import finish_wandb, init_wandb_run, log_wandb
from lgma.transformer import LGMA_ATTENTION_TYPES, TinyTransformerLM, load_model_config


ATTENTION_TYPES = [
    "mha",
    "mqa",
    "gqa",
    "collaborative",
    "shared_identity",
    "lgma",
    "lgma_v2",
    "lgma_residual",
    "lgma_quad",
    "lgma_unconstrained",
    "lgma_value_diag",
    "lgma_multibase",
    "lgma_multibase_value_diag",
]
GENERATOR_TYPES = ["full", "diagonal", "symmetric"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text LM runner with LGMA/MHA diagnostics."
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_path")
    data_group.add_argument(
        "--packed_data_dir",
        type=Path,
        help="Packed-token corpus produced by experiments/prepare_fineweb.py.",
    )
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
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=0,
        help=(
            "Keep only the newest N checkpoint_step_*.pt files. "
            "Use 0 to retain all periodic checkpoints."
        ),
    )
    parser.add_argument(
        "--milestone_checkpoint_every",
        type=int,
        default=0,
        help=(
            "Never prune periodic checkpoints whose step is divisible by N. "
            "Use 0 to disable milestone retention."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="Learning-rate schedule after optional warmup.",
    )
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        default=None,
        help=(
            "Optional step horizon used only by the learning-rate schedule. "
            "Defaults to --steps, allowing a shorter run to match a longer "
            "reference run's schedule."
        ),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Linearly warm up from 0 to --lr over this many optimizer steps.",
    )
    parser.add_argument(
        "--lr_hold_steps",
        type=int,
        default=0,
        help="Keep --lr constant for this many steps after warmup before decay.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Final LR for cosine decay. Ignored by the constant schedule.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_backend",
        choices=["eager", "aot_eager", "inductor"],
        default="inductor",
        help=(
            "torch.compile backend. Use eager to test TorchDynamo capture, "
            "aot_eager to additionally test AOTAutograd, or inductor for the "
            "fully optimized training path. Only used with --compile."
        ),
    )
    parser.add_argument(
        "--compile_mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default=None,
        help="Optional torch.compile optimization mode; only valid with the inductor backend.",
    )
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
    parser.add_argument(
        "--fuse_base_qkv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Issue one checkpoint-compatible concatenated base QKV projection.",
    )
    parser.add_argument(
        "--fold_value_transform_into_output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fold head-specific value transforms into output-projection blocks.",
    )
    parser.add_argument(
        "--sdpa_gqa_mode",
        choices=["auto", "native", "expand"],
        default=None,
        help="Select native GQA SDPA or temporary K/V expansion for performance testing.",
    )
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
        "--stabilize_generators",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Make dense metric and Lie-value generator bases trace zero.",
    )
    parser.add_argument(
        "--normalize_generators",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize each metric and Lie-value generator to unit Frobenius norm.",
    )
    parser.add_argument(
        "--head_generator_symmetric_cap",
        type=float,
        default=None,
        help=(
            "Optional Frobenius-norm cap on sym(A_h) before constructing the metric. "
            "For exp metrics, log(C) guarantees singular values in [1/C, C]."
        ),
    )
    parser.add_argument(
        "--metric_mode",
        choices=["exp", "residual", "quadratic", "unconstrained"],
        default=None,
    )
    parser.add_argument("--metric_beta", type=float, default=None)
    parser.add_argument("--metric_clip_min", type=float, default=None)
    parser.add_argument("--metric_clip_max", type=float, default=None)
    parser.add_argument(
        "--value_beta",
        type=float,
        default=None,
        help="Optional separate gain for value Lie transforms. Defaults to metric_beta.",
    )
    parser.add_argument("--theta_init", choices=["random_sphere", "circle"], default=None)
    parser.add_argument("--logit_scale_mode", choices=["sqrt_dim", "rms_metric"], default=None)
    parser.add_argument(
        "--learn_head_temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--value_transform",
        choices=[
            "none",
            "diag",
            "lie",
            "lie_exp",
            "lie_residual",
            "lie_quadratic",
            "unconstrained",
        ],
        default=None,
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
        "--induced_metric_diversity_weight",
        type=float,
        default=0.0,
        help="Weight for off-diagonal induced B_h cosine diversity regularization.",
    )
    parser.add_argument(
        "--ah_norm_weight",
        type=float,
        default=0.0,
        help="Weight for A_h Frobenius norm regularization.",
    )
    parser.add_argument(
        "--ah_norm_max",
        type=float,
        default=0.0,
        help="If positive, only penalize A_h norms above this radius.",
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


def learning_rate_for_step(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    total_steps: int,
    warmup_steps: int,
    hold_steps: int,
    schedule: str,
) -> float:
    if step <= 0:
        raise ValueError("step must be positive")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    hold_end = warmup_steps + hold_steps
    if step <= hold_end:
        return base_lr
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"unsupported lr schedule: {schedule}")

    decay_steps = max(total_steps - hold_end, 1)
    decay_step = min(max(step - hold_end, 0), decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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
        "stabilize_generators",
        "normalize_generators",
        "head_generator_symmetric_cap",
        "metric_mode",
        "metric_beta",
        "metric_clip_min",
        "metric_clip_max",
        "value_beta",
        "theta_init",
        "logit_scale_mode",
        "learn_head_temperature",
        "value_transform",
        "fuse_base_qkv",
        "fold_value_transform_into_output",
        "sdpa_gqa_mode",
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
        "stabilize_generators": True,
        "normalize_generators": False,
        "head_generator_symmetric_cap": None,
        "base_dim": None,
        "value_dim": None,
        "num_base_heads": 1,
        "metric_mode": "exp",
        "metric_beta": 1.0,
        "metric_clip_min": None,
        "metric_clip_max": None,
        "value_beta": None,
        "theta_init": "random_sphere",
        "logit_scale_mode": "sqrt_dim",
        "learn_head_temperature": False,
        "value_transform": "none",
        "fuse_base_qkv": False,
        "fold_value_transform_into_output": False,
        "sdpa_gqa_mode": "auto",
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
        "stabilize_generators": args.stabilize_generators,
        "normalize_generators": args.normalize_generators,
        "head_generator_symmetric_cap": args.head_generator_symmetric_cap,
        "metric_mode": args.metric_mode,
        "metric_beta": args.metric_beta,
        "metric_clip_min": args.metric_clip_min,
        "metric_clip_max": args.metric_clip_max,
        "value_beta": args.value_beta,
        "theta_init": args.theta_init,
        "logit_scale_mode": args.logit_scale_mode,
        "learn_head_temperature": args.learn_head_temperature,
        "value_transform": args.value_transform,
        "fuse_base_qkv": args.fuse_base_qkv,
        "fold_value_transform_into_output": args.fold_value_transform_into_output,
        "sdpa_gqa_mode": args.sdpa_gqa_mode,
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


def make_data_batch(
    data: torch.Tensor | PackedTokenSplit,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> SyntheticBatch:
    if isinstance(data, PackedTokenSplit):
        return data.sample_batch(
            batch_size,
            seq_len,
            device=device,
            generator=generator,
        )
    return make_lm_batch(
        data,
        batch_size,
        seq_len,
        device=device,
        generator=generator,
    )


@torch.no_grad()
def evaluate_text_loss(
    model: TinyTransformerLM,
    encoded: torch.Tensor | PackedTokenSplit,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_batches: int,
) -> float:
    model.eval()
    losses = []
    generator = None
    if isinstance(encoded, PackedTokenSplit):
        generator = torch.Generator().manual_seed(17_029)
    for _ in range(eval_batches):
        batch = make_data_batch(
            encoded,
            batch_size,
            seq_len,
            device,
            generator=generator,
        )
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
    if metric_weight == 0.0 and induced_weight == 0.0:
        zero = torch.zeros((), device=device)
        return zero, zero
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


def compute_ah_norm_regularizer(
    model: TinyTransformerLM,
    ah_norm_weight: float,
    ah_norm_max: float,
    device: torch.device,
) -> torch.Tensor:
    if ah_norm_weight == 0.0:
        return torch.zeros((), device=device)
    lgma_layers = [
        module for module in model.modules() if isinstance(module, LieGeneratedMetricAttention)
    ]
    if not lgma_layers:
        return torch.zeros((), device=device)
    penalties = []
    for module in lgma_layers:
        ah = module.compute_head_generators()
        norms = ah.float().reshape(ah.shape[0], -1).norm(dim=-1)
        if ah_norm_max > 0.0:
            penalties.append((norms - ah_norm_max).clamp_min(0.0).square().mean())
        else:
            penalties.append(norms.square().mean())
    return torch.stack(penalties).mean().to(device=device)


def add_attention_diagnostics(
    report: dict[str, object],
    model: TinyTransformerLM,
    encoded: torch.Tensor | PackedTokenSplit,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    diagnostic_batches: int,
) -> None:
    model = unwrap_model(model)
    first_attn = model.first_attention
    with torch.no_grad():
        generator = None
        if isinstance(encoded, PackedTokenSplit):
            generator = torch.Generator().manual_seed(91_337)
        attention_sims = []
        centered_sims = []
        score_sims = []
        for _ in range(max(1, diagnostic_batches)):
            batch = make_data_batch(
                encoded,
                batch_size,
                seq_len,
                device,
                generator=generator,
            )
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
            condition_numbers = metric_condition_number(metrics)
            report["metric_condition_number_mean"] = float(condition_numbers.mean())
            report["metric_condition_number_max"] = float(condition_numbers.max())
            singular_values = metric_singular_values(metrics)
            report["metric_singular_value_min"] = float(singular_values.min())
            report["metric_singular_value_max"] = float(singular_values.max())
            ah = first_attn.compute_head_generators()
            ah_norms = ah.float().reshape(ah.shape[0], -1).norm(dim=-1)
            report["ah_fro_norm_mean"] = float(ah_norms.mean())
            report["ah_fro_norm_max"] = float(ah_norms.max())
            cap_norms = []
            cap_active = []
            for module in model.modules():
                if not isinstance(module, LieGeneratedMetricAttention):
                    continue
                pre_cap_norms = module.pre_cap_head_generator_symmetric_norms()
                if pre_cap_norms is None:
                    continue
                cap_norms.append(pre_cap_norms)
                cap_active.append(pre_cap_norms > module.head_generator_symmetric_cap)
            if cap_norms:
                all_pre_cap_norms = torch.cat(cap_norms)
                all_cap_active = torch.cat(cap_active)
                report["head_generator_symmetric_pre_cap_norm_mean"] = float(
                    all_pre_cap_norms.mean()
                )
                report["head_generator_symmetric_pre_cap_norm_max"] = float(
                    all_pre_cap_norms.max()
                )
                report["head_generator_symmetric_cap_active_fraction"] = float(
                    all_cap_active.float().mean()
                )
                report["head_generator_symmetric_cap_active_heads"] = int(
                    all_cap_active.sum()
                )
                report["head_generator_symmetric_cap_total_heads"] = int(
                    all_cap_active.numel()
                )
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
    data_generator_states: list[torch.Tensor] | None = None,
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
        "data_generator_states": data_generator_states,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    print(json.dumps({"event": "checkpoint_saved", "step": step, "path": str(path)}))


def prune_old_checkpoints(
    output_dir: Path,
    keep: int,
    milestone_every: int = 0,
) -> None:
    if keep <= 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    prefix = "checkpoint_step_"
    suffix = ".pt"
    for path in output_dir.glob(f"{prefix}*{suffix}"):
        step_text = path.name[len(prefix) : -len(suffix)]
        if step_text.isdigit():
            checkpoints.append((int(step_text), path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    latest_steps = {step for step, _ in checkpoints[:keep]}
    for step, path in checkpoints:
        if step in latest_steps:
            continue
        if milestone_every > 0 and step % milestone_every == 0:
            continue
        path.unlink()
        print(json.dumps({"event": "checkpoint_pruned", "step": step, "path": str(path)}))


def load_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    data_generator: torch.Generator | None = None,
    data_rank: int = 0,
) -> int:
    # Trainer checkpoints contain optimizer/config objects in addition to tensors.
    # PyTorch 2.6+ defaults torch.load to weights_only=True, so explicitly opt in
    # to loading the full payload created by save_checkpoint above.
    checkpoint = load_full_checkpoint(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    data_generator_states = checkpoint.get("data_generator_states")
    if data_generator is not None and data_generator_states is not None:
        if data_rank >= len(data_generator_states):
            raise ValueError(
                f"checkpoint has {len(data_generator_states)} data sampler states, "
                f"but rank {data_rank} was requested"
            )
        data_generator.set_state(data_generator_states[data_rank])
    elif (
        data_generator is not None
        and data_rank == 0
        and checkpoint.get("data_generator_state") is not None
    ):
        # Backward compatibility with single-state checkpoints created before
        # per-rank DDP sampler state was recorded.
        data_generator.set_state(checkpoint["data_generator_state"])
    step = int(checkpoint.get("step", 0))
    print(json.dumps({"event": "checkpoint_loaded", "step": step, "path": str(path)}))
    return step


def gather_data_generator_states(
    data_generator: torch.Generator,
    ddp_enabled: bool,
    rank: int,
    world_size: int,
) -> list[torch.Tensor] | None:
    local_state = data_generator.get_state()
    if not ddp_enabled:
        return [local_state]
    gathered: list[torch.Tensor | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    dist.gather_object(local_state, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather every DDP data sampler state")
    return [state for state in gathered if state is not None]


def build_report(
    args: argparse.Namespace,
    config: dict[str, object],
    model: TinyTransformerLM,
    tokenizer,
    train_encoded: torch.Tensor | PackedTokenSplit,
    val_encoded: torch.Tensor | PackedTokenSplit,
    device: torch.device,
    final_loss: float,
    final_diversity_loss: float,
    final_induced_diversity_loss: float,
    final_ah_norm_loss: float,
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
        "packed_data_dir": str(args.packed_data_dir) if args.packed_data_dir else None,
        "data_backend": "packed_uint16" if args.packed_data_dir else "character_text",
        "config": args.config,
        "model_config": config,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train_encoded.numel()),
        "validation_tokens": int(val_encoded.numel()),
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
        "final_ah_norm_loss": final_ah_norm_loss,
        "metric_diversity_weight": args.metric_diversity_weight,
        "induced_metric_diversity_weight": args.induced_metric_diversity_weight,
        "ah_norm_weight": args.ah_norm_weight,
        "ah_norm_max": args.ah_norm_max,
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
    if args.packed_data_dir is not None and args.val_data_path is not None:
        raise SystemExit("--val_data_path cannot be used with --packed_data_dir")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad_accum_steps must be positive")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup_steps must be non-negative")
    if args.lr_hold_steps < 0:
        raise SystemExit("--lr_hold_steps must be non-negative")
    if args.min_lr < 0:
        raise SystemExit("--min_lr must be non-negative")
    if args.min_lr > args.lr:
        raise SystemExit("--min_lr must be <= --lr")
    if args.lr_schedule_steps is not None and args.lr_schedule_steps <= 0:
        raise SystemExit("--lr_schedule_steps must be positive")
    if args.log_every < 0 or args.eval_every < 0 or args.save_every < 0:
        raise SystemExit("--log_every, --eval_every, and --save_every must be non-negative")
    if args.keep_last_checkpoints < 0:
        raise SystemExit("--keep_last_checkpoints must be non-negative")
    if args.milestone_checkpoint_every < 0:
        raise SystemExit("--milestone_checkpoint_every must be non-negative")
    if args.metric_clip_min is not None and args.metric_clip_min < 0:
        raise SystemExit("--metric_clip_min must be non-negative")
    if (
        args.head_generator_symmetric_cap is not None
        and args.head_generator_symmetric_cap <= 0
    ):
        raise SystemExit("--head_generator_symmetric_cap must be positive")
    if (
        args.metric_clip_min is not None
        and args.metric_clip_max is not None
        and args.metric_clip_min > args.metric_clip_max
    ):
        raise SystemExit("--metric_clip_min must be <= --metric_clip_max")
    if args.ah_norm_weight < 0:
        raise SystemExit("--ah_norm_weight must be non-negative")
    if args.ah_norm_max < 0:
        raise SystemExit("--ah_norm_max must be non-negative")

    ddp_enabled, rank, local_rank, world_size = setup_distributed(args)
    main_process = is_main_process(rank)
    try:
        torch.manual_seed(args.seed + rank)
        # Keep data ordering independent of architecture initialization and save
        # its state in checkpoints for exact continuation after preemption.
        data_generator = torch.Generator().manual_seed(args.seed + rank)
        if ddp_enabled:
            if args.device != "cuda":
                raise SystemExit("DDP requires --device cuda")
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = resolve_device(args.device)
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
        if args.packed_data_dir is not None:
            corpus = load_packed_token_corpus(args.packed_data_dir)
            tokenizer = corpus.tokenizer
            train_encoded = corpus.train
            val_encoded = corpus.validation
            data_backend = "packed_uint16"
        else:
            train_text = Path(args.data_path).read_text(encoding="utf-8")
            val_text = (
                Path(args.val_data_path).read_text(encoding="utf-8")
                if args.val_data_path is not None
                else None
            )
            tokenizer = build_tokenizer(train_text, val_text)
            train_encoded = tokenizer.encode(train_text)
            val_encoded = tokenizer.encode(val_text) if val_text is not None else train_encoded
            data_backend = "character_text"

        config = model_config_from_args(args)
        seq_len = int(config["context_length"])
        base_model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config).to(device)
        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scaler = make_grad_scaler(device, args.precision)

        last_loss = None
        last_diversity_loss = 0.0
        last_induced_diversity_loss = 0.0
        last_ah_norm_loss = 0.0
        last_grad_norms: dict[str, float] = {}
        use_delta = not args.metric_diversity_on_full_metric

        output_dir = args.output_dir
        metrics_path = output_dir / "metrics.jsonl" if output_dir is not None else None
        if output_dir is not None and main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = None
        if main_process:
            first_attn = base_model.first_attention
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
                    "parameters": count_parameters(base_model),
                    "vocab_size": tokenizer.vocab_size,
                    "data_backend": data_backend,
                    "train_tokens": int(train_encoded.numel()),
                    "validation_tokens": int(val_encoded.numel()),
                    "data_sampling": "uniform_random_with_replacement",
                    "data_seed": args.seed + rank,
                    "distributed": ddp_enabled,
                    "world_size": world_size,
                },
            )

        start_step = 0
        if args.resume_checkpoint is not None:
            start_step = load_checkpoint(
                args.resume_checkpoint,
                base_model,
                optimizer,
                scaler,
                device,
                data_generator,
                rank,
            )
        if ddp_enabled:
            dist.barrier()

        # Compile after restoring the checkpoint. This avoids charging lazy
        # compiler work to checkpoint loading and keeps checkpoint/optimizer
        # state tied to the original module and Parameter objects.
        if args.compile:
            if not hasattr(base_model, "compile"):
                raise SystemExit("--compile requires a PyTorch version with torch.compile")
            if args.compile_mode is not None and args.compile_backend != "inductor":
                raise SystemExit("--compile_mode requires --compile_backend inductor")
            compile_kwargs: dict[str, object] = {"backend": args.compile_backend}
            if args.compile_mode is not None:
                compile_kwargs["mode"] = args.compile_mode
            base_model.compile(**compile_kwargs)

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

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        training_start_time = time.perf_counter()
        last_log_time = training_start_time
        last_log_step = start_step
        tokens_per_step = args.batch_size * seq_len * args.grad_accum_steps * world_size
        effective_global_batch = args.batch_size * args.grad_accum_steps * world_size

        for step_idx in range(start_step, args.steps):
            step = step_idx + 1
            should_log = args.log_every > 0 and (step == 1 or step % args.log_every == 0)
            should_eval = args.eval_every > 0 and step % args.eval_every == 0
            should_diagnose = args.diagnostic_every > 0 and step % args.diagnostic_every == 0
            is_final = step == args.steps
            should_report = should_log or should_eval or should_diagnose or is_final
            current_lr = learning_rate_for_step(
                step,
                base_lr=args.lr,
                min_lr=args.min_lr,
                total_steps=args.lr_schedule_steps or args.steps,
                warmup_steps=args.warmup_steps,
                hold_steps=args.lr_hold_steps,
                schedule=args.lr_schedule,
            )
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)

            task_losses = []
            diversity_losses = []
            induced_diversity_losses = []
            ah_norm_losses = []
            for accum_idx in range(args.grad_accum_steps):
                batch = make_data_batch(
                    train_encoded,
                    args.batch_size,
                    seq_len,
                    device,
                    generator=data_generator,
                )
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
                        ah_norm_loss = compute_ah_norm_regularizer(
                            model,
                            ah_norm_weight=args.ah_norm_weight,
                            ah_norm_max=args.ah_norm_max,
                            device=device,
                        )
                        loss = (
                            task_loss
                            + args.metric_diversity_weight * diversity_loss
                            + args.induced_metric_diversity_weight * induced_diversity_loss
                            + args.ah_norm_weight * ah_norm_loss
                        )
                        loss = loss / args.grad_accum_steps
                    scaler.scale(loss).backward()
                task_losses.append(task_loss.detach())
                diversity_losses.append(diversity_loss.detach())
                induced_diversity_losses.append(induced_diversity_loss.detach())
                ah_norm_losses.append(ah_norm_loss.detach())

            if device.type == "cuda" and args.precision == "fp16":
                scaler.unscale_(optimizer)
            first_attention = unwrap_model(model).first_attention
            if (
                should_report
                and main_process
                and isinstance(first_attention, LieGeneratedMetricAttention)
            ):
                last_grad_norms = grouped_gradient_norms(first_attention)
            if args.max_grad_norm > 0:
                if not (device.type == "cuda" and args.precision == "fp16"):
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if should_report:
                mean_task_loss = distributed_mean(
                    torch.stack(task_losses).mean(), ddp_enabled
                )
                mean_diversity_loss = distributed_mean(
                    torch.stack(diversity_losses).mean(), ddp_enabled
                )
                mean_induced_diversity_loss = distributed_mean(
                    torch.stack(induced_diversity_losses).mean(), ddp_enabled
                )
                mean_ah_norm_loss = distributed_mean(
                    torch.stack(ah_norm_losses).mean(), ddp_enabled
                )
                last_loss = float(mean_task_loss.cpu())
                last_diversity_loss = float(mean_diversity_loss.cpu())
                last_induced_diversity_loss = float(mean_induced_diversity_loss.cpu())
                last_ah_norm_loss = float(mean_ah_norm_loss.cpu())

            if should_report and main_process:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-12)
                elapsed_training_seconds = now - training_start_time
                steps_since_log = max(step - last_log_step, 1)
                tokens_per_second = tokens_per_step * steps_since_log / elapsed
                tokens_seen = step * tokens_per_step
                elapsed_training_hours = elapsed_training_seconds / 3600.0
                payload: dict[str, object] = {
                    "step": step,
                    "loss": last_loss,
                    "perplexity": math.exp(min(last_loss, 20.0)),
                    "metric_diversity_loss": last_diversity_loss,
                    "induced_metric_diversity_loss": last_induced_diversity_loss,
                    "ah_norm_loss": last_ah_norm_loss,
                    "tokens_seen": tokens_seen,
                    "tokens_per_step": tokens_per_step,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second / max(world_size, 1),
                    "effective_global_batch": effective_global_batch,
                    "elapsed_training_seconds": elapsed_training_seconds,
                    "elapsed_training_hours": elapsed_training_hours,
                    "gpu_hours": elapsed_training_hours * world_size,
                    "lr": optimizer.param_groups[0]["lr"],
                    "rank": rank,
                    "world_size": world_size,
                }
                payload.update(last_grad_norms)
                if device.type == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated(device)
                    payload["peak_memory_bytes"] = peak_memory
                    payload["peak_memory_gib"] = peak_memory / float(1024**3)
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
            ):
                data_generator_states = gather_data_generator_states(
                    data_generator,
                    ddp_enabled,
                    rank,
                    world_size,
                )
                if main_process:
                    save_checkpoint(
                        output_dir / f"checkpoint_step_{step}.pt",
                        model,
                        optimizer,
                        scaler,
                        step,
                        config,
                        args,
                        data_generator_states,
                    )
                    prune_old_checkpoints(
                        output_dir,
                        args.keep_last_checkpoints,
                        args.milestone_checkpoint_every,
                    )

        if last_loss is None:
            raise SystemExit("--steps must be greater than resume checkpoint step")

        final_data_generator_states = gather_data_generator_states(
            data_generator,
            ddp_enabled,
            rank,
            world_size,
        )
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
                last_ah_norm_loss,
                last_grad_norms,
            )
            report["distributed"] = ddp_enabled
            report["world_size"] = world_size
            report["tokens_per_step"] = tokens_per_step
            report["tokens_seen"] = args.steps * tokens_per_step
            report["effective_global_batch"] = effective_global_batch
            report["data_sampling"] = "uniform_random_with_replacement"
            report["data_seed"] = args.seed
            elapsed_training_seconds = time.perf_counter() - training_start_time
            report["elapsed_training_seconds"] = elapsed_training_seconds
            report["elapsed_training_hours"] = elapsed_training_seconds / 3600.0
            report["gpu_hours"] = report["elapsed_training_hours"] * world_size
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(device)
                report["peak_memory_bytes"] = peak_memory
                report["peak_memory_gib"] = peak_memory / float(1024**3)
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
                    final_data_generator_states,
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
