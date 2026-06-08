from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.accounting import attention_accounting, count_parameters
from lgma.diagnostics import (
    attention_cosine_similarity,
    centered_attention_cosine_similarity,
    grouped_gradient_norms,
    grouped_similarity_stats,
    induced_bilinear_forms,
    induced_metric_cosine_similarity,
    mean_off_diagonal,
    metric_delta_cosine_similarity,
    metric_diversity_loss,
    metric_distance_from_identity,
    metric_cosine_similarity,
    score_cosine_similarity,
)
from lgma.attention import LieGeneratedMetricAttention
from lgma.synthetic import make_synthetic_batch
from lgma.tracking import finish_wandb, init_wandb_run, log_wandb
from lgma.transformer import LGMA_ATTENTION_TYPES, TinyTransformerLM


SYNTHETIC_TASKS = [
    "copy",
    "reverse",
    "modular",
    "previous",
    "cumsum_mod",
    "multi_relation",
]
MULTI_RELATION_NAMES = ["copy", "reverse", "previous", "next"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny LM on synthetic tasks.")
    parser.add_argument(
        "--task",
        choices=SYNTHETIC_TASKS,
        default="copy",
    )
    parser.add_argument(
        "--attention",
        choices=[
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
        ],
        default="lgma",
    )
    parser.add_argument("--generator_type", choices=["full", "diagonal", "symmetric"], default="full")
    parser.add_argument("--num_generators", type=int, default=2)
    parser.add_argument("--base_dim", type=int, default=None)
    parser.add_argument("--value_dim", type=int, default=None)
    parser.add_argument("--num_base_heads", type=int, default=1)
    parser.add_argument("--metric_mode", choices=["exp", "residual", "unconstrained"], default="exp")
    parser.add_argument("--metric_beta", type=float, default=1.0)
    parser.add_argument("--theta_init", choices=["random_sphere", "circle"], default="random_sphere")
    parser.add_argument("--logit_scale_mode", choices=["sqrt_dim", "rms_metric"], default="sqrt_dim")
    parser.add_argument("--learn_head_temperature", action="store_true")
    parser.add_argument("--value_transform", choices=["none", "diag"], default="none")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=8,
        help="Number of fresh synthetic batches used for validation reporting.",
    )
    parser.add_argument(
        "--theta_init_scale",
        type=float,
        default=0.02,
        help="LGMA head-coordinate initialization scale. Try 0.25 or 0.5.",
    )
    parser.add_argument(
        "--generator_init_scale",
        type=float,
        default=0.02,
        help="LGMA generator initialization scale before division by sqrt(head_dim). Try 0.1 or 0.2.",
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
        "--diagnostic_every",
        type=int,
        default=0,
        help="Emit attention diagnostics every N training steps. Disabled when 0.",
    )
    parser.add_argument(
        "--diagnostic_batches",
        type=int,
        default=1,
        help="Number of batches used for diagnostic logging.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal attention. Default synthetic experiments are non-causal.",
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
        help="Comma-separated W&B tags, for example `synthetic,lgma,b2`.",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb_dir", type=Path, default=None)
    return parser.parse_args()


def lgma_layers(model: TinyTransformerLM) -> list[LieGeneratedMetricAttention]:
    return [
        module
        for module in model.modules()
        if isinstance(module, LieGeneratedMetricAttention)
    ]


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


def compute_diversity_regularizers(
    model: TinyTransformerLM,
    metric_weight: float,
    induced_weight: float,
    squared: bool,
    use_delta: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    layers = lgma_layers(model)
    metric_loss = torch.zeros((), device=device)
    induced_loss = torch.zeros((), device=device)
    if not layers:
        return metric_loss, induced_loss
    if metric_weight != 0.0:
        metric_loss = torch.stack(
            [
                metric_diversity_loss(
                    module.compute_metrics(),
                    squared=squared,
                    use_delta=use_delta,
                )
                for module in layers
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
                for module in layers
            ]
        ).mean()
    return metric_loss, induced_loss


@torch.no_grad()
def attention_diagnostics(
    model: TinyTransformerLM,
    task: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    diagnostic_batches: int,
) -> dict[str, float]:
    first_attn = model.first_attention
    attention_sims = []
    centered_sims = []
    score_sims = []
    for _ in range(max(1, diagnostic_batches)):
        batch = make_synthetic_batch(task, batch_size, seq_len, vocab_size, device=device)
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
    report = {
        "attention_diversity_mean_cosine": float(attention_similarity.mean()),
        "attention_diversity_offdiag_mean_cosine": float(mean_off_diagonal(attention_similarity)),
        "centered_attention_diversity_mean_cosine": float(centered_similarity.mean()),
        "centered_attention_diversity_offdiag_mean_cosine": float(
            mean_off_diagonal(centered_similarity)
        ),
    }
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
        report["score_diversity_offdiag_mean_cosine"] = float(mean_off_diagonal(score_similarity))
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
        report["metric_diversity_mean_cosine"] = float(metric_similarity.mean())
        report["metric_diversity_offdiag_mean_cosine"] = float(
            mean_off_diagonal(metric_similarity)
        )
        metric_delta_similarity = metric_delta_cosine_similarity(metrics)
        report["metric_delta_diversity_mean_cosine"] = float(metric_delta_similarity.mean())
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
    return report


@torch.no_grad()
def evaluate_loss(
    model: TinyTransformerLM,
    task: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    eval_batches: int,
    relation_id: int | None = None,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        relation_ids = None
        if relation_id is not None:
            relation_ids = torch.full((batch_size,), relation_id, device=device, dtype=torch.long)
        batch = make_synthetic_batch(
            task,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            relation_ids=relation_ids,
        )
        _, loss = model(batch.input_ids, batch.targets)
        losses.append(loss.detach())
    model.train()
    return float(torch.stack(losses).mean().cpu())


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = TinyTransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        attention_type=args.attention,
        num_generators=args.num_generators if args.attention in LGMA_ATTENTION_TYPES else 0,
        generator_type=args.generator_type,
        context_length=args.seq_len,
        causal=args.causal,
        theta_init_scale=args.theta_init_scale,
        generator_init_scale=args.generator_init_scale,
        base_dim=args.base_dim,
        value_dim=args.value_dim,
        metric_mode=args.metric_mode,
        metric_beta=args.metric_beta,
        theta_init=args.theta_init,
        logit_scale_mode=args.logit_scale_mode,
        learn_head_temperature=args.learn_head_temperature,
        value_transform=args.value_transform,
        num_base_heads=args.num_base_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    first_attn = model.first_attention
    wandb_run = init_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        output_dir=args.wandb_dir,
        config={
            "args": vars(args),
            "effective_attention_config": effective_attention_config(first_attn),
            "attention_accounting": attention_accounting(
                first_attn,
                sequence_length=args.seq_len,
                batch_size=args.batch_size,
            ).__dict__,
            "parameters": count_parameters(model),
        },
    )

    last_loss = None
    last_diversity_loss = 0.0
    last_induced_diversity_loss = 0.0
    last_grad_norms: dict[str, float] = {}
    for step in range(args.steps):
        batch = make_synthetic_batch(
            args.task,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
        )
        _, task_loss = model(batch.input_ids, batch.targets)
        diversity_loss, induced_diversity_loss = compute_diversity_regularizers(
            model,
            args.metric_diversity_weight,
            args.induced_metric_diversity_weight,
            args.metric_diversity_squared,
            not args.metric_diversity_on_full_metric,
            device,
        )
        loss = (
            task_loss
            + args.metric_diversity_weight * diversity_loss
            + args.induced_metric_diversity_weight * induced_diversity_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if isinstance(model.first_attention, LieGeneratedMetricAttention):
            last_grad_norms = grouped_gradient_norms(model.first_attention)
        optimizer.step()
        last_loss = float(task_loss.detach().cpu())
        last_diversity_loss = float(diversity_loss.detach().cpu())
        last_induced_diversity_loss = float(induced_diversity_loss.detach().cpu())
        should_diagnose = args.diagnostic_every > 0 and (step + 1) % args.diagnostic_every == 0
        if step == 0 or (step + 1) == args.steps or should_diagnose:
            payload = {
                "step": step + 1,
                "loss": last_loss,
                "metric_diversity_loss": last_diversity_loss,
                "induced_metric_diversity_loss": last_induced_diversity_loss,
            }
            payload.update(last_grad_norms)
            if should_diagnose:
                payload.update(
                    attention_diagnostics(
                        model,
                        args.task,
                        args.batch_size,
                        args.seq_len,
                        args.vocab_size,
                        device,
                        args.diagnostic_batches,
                    )
                )
            print(json.dumps(payload))
            log_wandb(wandb_run, payload, step=step + 1)

    first_attn = model.first_attention
    report = {
        "task": args.task,
        "attention": args.attention,
        "causal": args.causal,
        "theta_init_scale": args.theta_init_scale,
        "generator_init_scale": args.generator_init_scale,
        "base_dim": args.base_dim,
        "value_dim": args.value_dim,
        "num_base_heads": args.num_base_heads,
        "metric_mode": args.metric_mode,
        "metric_beta": args.metric_beta,
        "theta_init": args.theta_init,
        "logit_scale_mode": args.logit_scale_mode,
        "learn_head_temperature": args.learn_head_temperature,
        "value_transform": args.value_transform,
        "metric_diversity_weight": args.metric_diversity_weight,
        "induced_metric_diversity_weight": args.induced_metric_diversity_weight,
        "metric_diversity_squared": args.metric_diversity_squared,
        "metric_diversity_on_delta": not args.metric_diversity_on_full_metric,
        "parameters": count_parameters(model),
        "effective_attention_config": effective_attention_config(first_attn),
        "attention_accounting": attention_accounting(
            first_attn, sequence_length=args.seq_len, batch_size=args.batch_size
        ).__dict__,
        "final_loss": last_loss,
        "validation_loss": evaluate_loss(
            model,
            args.task,
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            args.eval_batches,
        ),
        "final_metric_diversity_loss": last_diversity_loss,
        "final_induced_metric_diversity_loss": last_induced_diversity_loss,
        "gradient_norms": last_grad_norms,
    }
    if args.task == "multi_relation":
        report["validation_loss_by_relation"] = {
            name: evaluate_loss(
                model,
                args.task,
                args.batch_size,
                args.seq_len,
                args.vocab_size,
                device,
                args.eval_batches,
                relation_id=idx,
            )
            for idx, name in enumerate(MULTI_RELATION_NAMES)
        }
    report.update(
        attention_diagnostics(
            model,
            args.task,
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            args.diagnostic_batches,
        )
    )
    log_wandb(wandb_run, {"final": report}, step=args.steps)
    print(json.dumps(report, indent=2))
    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
