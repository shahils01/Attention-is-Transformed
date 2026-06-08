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
    mean_off_diagonal,
    metric_delta_cosine_similarity,
    metric_diversity_loss,
    metric_cosine_similarity,
)
from lgma.attention import LieGeneratedMetricAttention
from lgma.synthetic import make_synthetic_batch
from lgma.transformer import TinyTransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny LM on synthetic tasks.")
    parser.add_argument(
        "--task",
        choices=["copy", "reverse", "modular", "previous", "cumsum_mod"],
        default="copy",
    )
    parser.add_argument("--attention", choices=["mha", "mqa", "gqa", "shared_identity", "lgma"], default="lgma")
    parser.add_argument("--generator_type", choices=["full", "diagonal", "symmetric"], default="full")
    parser.add_argument("--num_generators", type=int, default=2)
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
        "--causal",
        action="store_true",
        help="Use causal attention. Default synthetic experiments are non-causal.",
    )
    return parser.parse_args()


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
        num_generators=args.num_generators if args.attention == "lgma" else 0,
        generator_type=args.generator_type,
        context_length=args.seq_len,
        causal=args.causal,
        theta_init_scale=args.theta_init_scale,
        generator_init_scale=args.generator_init_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    last_loss = None
    last_diversity_loss = 0.0
    for step in range(args.steps):
        batch = make_synthetic_batch(
            args.task,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
        )
        _, task_loss = model(batch.input_ids, batch.targets)
        diversity_loss = torch.zeros((), device=device)
        if args.metric_diversity_weight != 0.0:
            lgma_layers = [
                module
                for module in model.modules()
                if isinstance(module, LieGeneratedMetricAttention)
            ]
            if lgma_layers:
                layer_losses = [
                    metric_diversity_loss(
                        module.compute_metrics(),
                        squared=args.metric_diversity_squared,
                        use_delta=not args.metric_diversity_on_full_metric,
                    )
                    for module in lgma_layers
                ]
                diversity_loss = torch.stack(layer_losses).mean()
        loss = task_loss + args.metric_diversity_weight * diversity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(task_loss.detach().cpu())
        last_diversity_loss = float(diversity_loss.detach().cpu())
        if step == 0 or (step + 1) == args.steps:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": last_loss,
                        "metric_diversity_loss": last_diversity_loss,
                    }
                )
            )

    first_attn = model.first_attention
    report = {
        "task": args.task,
        "attention": args.attention,
        "causal": args.causal,
        "theta_init_scale": args.theta_init_scale,
        "generator_init_scale": args.generator_init_scale,
        "metric_diversity_weight": args.metric_diversity_weight,
        "metric_diversity_squared": args.metric_diversity_squared,
        "metric_diversity_on_delta": not args.metric_diversity_on_full_metric,
        "parameters": count_parameters(model),
        "attention_accounting": attention_accounting(
            first_attn, sequence_length=args.seq_len, batch_size=args.batch_size
        ).__dict__,
        "final_loss": last_loss,
        "final_metric_diversity_loss": last_diversity_loss,
    }
    with torch.no_grad():
        batch = make_synthetic_batch(
            args.task, args.batch_size, args.seq_len, args.vocab_size, device=device
        )
        x = model.blocks[0].norm1(
            model.token_embedding(batch.input_ids)
            + model.position_embedding(torch.arange(args.seq_len, device=device))[None, :, :]
        )
        result = first_attn(x, need_weights=True)
        attn = result[1]
        attention_similarity = attention_cosine_similarity(attn)
        report["attention_diversity_mean_cosine"] = float(attention_similarity.mean())
        report["attention_diversity_offdiag_mean_cosine"] = float(
            mean_off_diagonal(attention_similarity)
        )
        if hasattr(first_attn, "compute_metrics"):
            metric_similarity = metric_cosine_similarity(first_attn.compute_metrics())
            report["metric_diversity_mean_cosine"] = float(metric_similarity.mean())
            report["metric_diversity_offdiag_mean_cosine"] = float(
                mean_off_diagonal(metric_similarity)
            )
            metric_delta_similarity = metric_delta_cosine_similarity(first_attn.compute_metrics())
            report["metric_delta_diversity_mean_cosine"] = float(
                metric_delta_similarity.mean()
            )
            report["metric_delta_diversity_offdiag_mean_cosine"] = float(
                mean_off_diagonal(metric_delta_similarity)
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
