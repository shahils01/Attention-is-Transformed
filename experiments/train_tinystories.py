from __future__ import annotations

import argparse
import json
import math
import sys
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    last_loss = None
    last_diversity_loss = 0.0
    use_delta = not args.metric_diversity_on_full_metric
    for step in range(args.steps):
        batch = make_lm_batch(train_encoded, args.batch_size, seq_len, device=device)
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
                        "perplexity": math.exp(min(last_loss, 20.0)),
                        "metric_diversity_loss": last_diversity_loss,
                    }
                )
            )

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
        "final_loss": last_loss,
        "final_perplexity": math.exp(min(float(last_loss), 20.0)),
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        "final_metric_diversity_loss": last_diversity_loss,
        "metric_diversity_weight": args.metric_diversity_weight,
        "metric_diversity_squared": args.metric_diversity_squared,
        "metric_diversity_on_delta": use_delta,
    }
    diagnostic_batch = make_lm_batch(val_encoded, args.batch_size, seq_len, device=device)
    add_attention_diagnostics(report, model, diagnostic_batch)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
