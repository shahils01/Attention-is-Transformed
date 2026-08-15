from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.accounting import attention_accounting, count_parameters
from lgma.diagnostics import (
    grouped_similarity_stats,
    mean_off_diagonal,
    metric_delta_cosine_similarity,
    metric_distance_from_identity,
)
from lgma.tracking import finish_wandb, init_wandb_run, log_wandb
from lgma.vla_data import XVLAMetaDataset
from lgma.vla_model import VLAPolicyConfig, VLATransformerPolicy, ee6d_continuous_loss, ee6d_loss


ATTENTION_CHOICES = (
    "mha",
    "collaborative",
    "shared_identity",
    "lgma",
    "lgma_multibase",
    "lgma_residual",
    "lgma_quad",
    "lgma_unconstrained",
    "lgma_value_diag",
    "lgma_multibase_value_diag",
)
GENERATOR_TYPES = ("full", "diagonal", "symmetric")
VALUE_TRANSFORMS = (
    "none",
    "diag",
    "lie",
    "lie_exp",
    "lie_residual",
    "lie_quadratic",
    "unconstrained",
)
ACTION_HEADS = ("mlp", "flow")
ACTION_HEAD_CONFIG_KEYS = {
    "action_head",
    "flow_hidden_mult",
    "flow_sampling_steps",
    "flow_noise_scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Minimal VLA attention benchmark")
    parser.add_argument("--train_metas_path", required=True)
    parser.add_argument("--val_metas_path", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/vla"))
    parser.add_argument(
        "--resume_checkpoint",
        type=Path,
        default=None,
        help="Resume from a checkpoint.pt file or checkpoint_step_* directory.",
    )
    parser.add_argument("--attention", choices=ATTENTION_CHOICES, default="mha")
    parser.add_argument("--num_base_heads", type=int, default=1)
    parser.add_argument("--num_generators", type=int, default=4)
    parser.add_argument("--generator_type", choices=GENERATOR_TYPES, default="full")
    parser.add_argument(
        "--generator_mixing",
        choices=("softmax", "none"),
        default="softmax",
        help="Map per-head generator coordinates with softmax or use them directly.",
    )
    parser.add_argument("--theta_init_scale", type=float, default=0.02)
    parser.add_argument("--generator_init_scale", type=float, default=0.02)
    parser.add_argument(
        "--stabilize_generators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make dense metric and Lie-value generator bases trace zero.",
    )
    parser.add_argument(
        "--normalize_generators",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--base_dim", type=int, default=None)
    parser.add_argument("--value_dim", type=int, default=None)
    parser.add_argument(
        "--value_beta",
        type=float,
        default=None,
        help="Optional separate gain for value Lie transforms. Defaults to metric_beta.",
    )
    parser.add_argument("--value_transform", choices=VALUE_TRANSFORMS, default="none")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--text_length", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--action_head", choices=ACTION_HEADS, default="mlp")
    parser.add_argument(
        "--flow_hidden_mult",
        type=int,
        default=2,
        help="Hidden width multiplier for --action_head flow.",
    )
    parser.add_argument(
        "--flow_sampling_steps",
        type=int,
        default=10,
        help="Euler integration steps used when sampling a flow action head.",
    )
    parser.add_argument(
        "--flow_noise_scale",
        type=float,
        default=1.0,
        help="Gaussian prior scale for flow matching.",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument(
        "--wandb_tags",
        default=None,
        help="Comma-separated W&B tags, for example `libero,vla,lgma_residual,b2`.",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb_dir", type=Path, default=None)
    parser.add_argument(
        "--no_ddp",
        action="store_true",
        help="Disable torchrun/DDP setup even when WORLD_SIZE > 1.",
    )
    parser.add_argument(
        "--reset_action_head_on_resume",
        action="store_true",
        help=(
            "Load compatible checkpoint weights except action_head.*, keep the checkpoint step, "
            "and start with a fresh optimizer. Use when changing the action head."
        ),
    )
    return parser.parse_args()


def distributed_env_enabled(args: argparse.Namespace) -> bool:
    return not args.no_ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    if not distributed_env_enabled(args):
        return False, 0, 0, 1, torch.device(args.device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size, device


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> VLATransformerPolicy:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def distributed_mean(value: torch.Tensor, enabled: bool) -> torch.Tensor:
    value = value.detach().float()
    if enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def make_config(args: argparse.Namespace) -> VLAPolicyConfig:
    num_base_heads = args.num_base_heads
    if args.attention == "lgma_multibase" and num_base_heads == 1:
        num_base_heads = 2
    return VLAPolicyConfig(
        image_size=args.image_size,
        num_views=args.num_views,
        vocab_size=args.vocab_size,
        text_length=args.text_length,
        action_horizon=args.action_horizon,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention=args.attention,
        num_generators=args.num_generators,
        generator_type=args.generator_type,
        generator_mixing=args.generator_mixing,
        theta_init_scale=args.theta_init_scale,
        generator_init_scale=args.generator_init_scale,
        stabilize_generators=args.stabilize_generators,
        normalize_generators=args.normalize_generators,
        head_generator_symmetric_cap=args.head_generator_symmetric_cap,
        num_base_heads=num_base_heads,
        base_dim=args.base_dim,
        value_dim=args.value_dim,
        value_beta=args.value_beta,
        value_transform=args.value_transform,
        action_head=args.action_head,
        flow_hidden_mult=args.flow_hidden_mult,
        flow_sampling_steps=args.flow_sampling_steps,
        flow_noise_scale=args.flow_noise_scale,
    )


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def make_datasets(args: argparse.Namespace) -> tuple[XVLAMetaDataset | torch.utils.data.Subset, object | None]:
    train_dataset = XVLAMetaDataset(
        args.train_metas_path,
        action_horizon=args.action_horizon,
        num_views=args.num_views,
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        text_length=args.text_length,
        stride=args.stride,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
    )
    if args.val_metas_path:
        val_dataset = XVLAMetaDataset(
            args.val_metas_path,
            action_horizon=args.action_horizon,
            num_views=args.num_views,
            image_size=args.image_size,
            vocab_size=args.vocab_size,
            text_length=args.text_length,
            stride=args.stride,
            max_episodes=args.max_episodes,
            max_samples=args.max_samples,
        )
        return train_dataset, val_dataset
    if args.val_fraction <= 0.0 or len(train_dataset) < 2:
        return train_dataset, None
    val_len = max(1, int(len(train_dataset) * args.val_fraction))
    train_len = len(train_dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    return random_split(train_dataset, [train_len, val_len], generator=generator)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_from_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = move_batch(batch, device)
    with autocast_context(device, precision):
        raw_model = unwrap_model(model)
        if raw_model.config.action_head == "flow":
            target = batch["action"]
            timestep = torch.rand(target.shape[0], device=device, dtype=target.dtype)
            noise = torch.randn_like(target) * raw_model.config.flow_noise_scale
            mix = timestep.view(-1, 1, 1)
            noisy_action = (1.0 - mix) * noise + mix * target
            target_velocity = target - noise
            pred = model(
                image_input=batch["image_input"],
                image_mask=batch["image_mask"],
                text_token_ids=batch["text_token_ids"],
                proprio=batch["proprio"],
                noisy_action=noisy_action,
                flow_timestep=timestep,
            )
            loss_dict = ee6d_continuous_loss(pred, target_velocity)
        else:
            pred = model(
                image_input=batch["image_input"],
                image_mask=batch["image_mask"],
                text_token_ids=batch["text_token_ids"],
                proprio=batch["proprio"],
            )
            loss_dict = ee6d_loss(pred, batch["action"])
        loss = sum(loss_dict.values())
    return loss, loss_dict


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int,
    distributed: bool = False,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=device)
    for batch in loader:
        loss, loss_dict = loss_from_batch(model, batch, device, precision)
        logs = {key: value.detach().float() for key, value in loss_dict.items()}
        logs["loss_total"] = loss.detach().float()
        for key, value in logs.items():
            totals[key] = totals.get(key, torch.zeros((), device=device)) + value
        count += 1.0
        if int(count.item()) >= max_batches:
            break
    if distributed:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        for value in totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    model.train()
    if float(count.item()) == 0.0:
        return {}
    return {f"val_{key}": float((value / count).detach().cpu()) for key, value in totals.items()}


def lgma_diagnostics(model: VLATransformerPolicy) -> dict[str, float]:
    logs: dict[str, float] = {}
    for layer_idx, attn in enumerate(model.attention_modules):
        if not hasattr(attn, "compute_metrics"):
            continue
        metrics = attn.compute_metrics().detach().float()
        similarity = metric_delta_cosine_similarity(metrics)
        prefix = f"attn{layer_idx}"
        logs[f"{prefix}_metric_delta_offdiag_cosine"] = float(mean_off_diagonal(similarity).cpu())
        logs[f"{prefix}_metric_distance_from_identity"] = float(
            metric_distance_from_identity(metrics).mean().cpu()
        )
        grouped = grouped_similarity_stats(
            similarity,
            num_base_heads=getattr(attn, "num_base_heads", 1),
            generated_heads_per_base=getattr(attn, "generated_heads_per_base", attn.num_heads),
        )
        for key, value in grouped.items():
            logs[f"{prefix}_{key}"] = float(value.cpu())
    return logs


def write_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: VLAPolicyConfig,
    output_dir: Path,
    step: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    ckpt_dir = output_dir / f"checkpoint_step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "config": config.to_dict(),
            "step": step,
        },
        ckpt_dir / "checkpoint.pt",
    )
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)


def resolve_checkpoint_path(path: Path) -> Path:
    checkpoint_path = path / "checkpoint.pt" if path.is_dir() else path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def _config_mismatch_message(
    checkpoint_config: VLAPolicyConfig,
    current_config: VLAPolicyConfig,
    allowed_keys: set[str] | None = None,
) -> str | None:
    checkpoint_dict = checkpoint_config.to_dict()
    current_dict = current_config.to_dict()
    diffs = []
    for key in sorted(set(checkpoint_dict) | set(current_dict)):
        if allowed_keys is not None and key in allowed_keys:
            continue
        if checkpoint_dict.get(key) != current_dict.get(key):
            diffs.append(f"{key}: checkpoint={checkpoint_dict.get(key)!r}, current={current_dict.get(key)!r}")
    if not diffs:
        return None
    preview = "; ".join(diffs[:8])
    suffix = "" if len(diffs) <= 8 else f"; ... {len(diffs) - 8} more"
    return f"resume checkpoint config does not match current args ({preview}{suffix})"


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    config: VLAPolicyConfig,
    device: torch.device,
    reset_action_head: bool = False,
) -> int:
    checkpoint_path = resolve_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = VLAPolicyConfig(**payload["config"])
    allowed_mismatch = ACTION_HEAD_CONFIG_KEYS if reset_action_head else None
    mismatch = _config_mismatch_message(checkpoint_config, config, allowed_mismatch)
    if mismatch is not None:
        raise ValueError(mismatch)
    raw_model = unwrap_model(model)
    if reset_action_head:
        current_state = raw_model.state_dict()
        loaded_state = {}
        skipped = []
        for key, value in payload["model"].items():
            if key.startswith("action_head."):
                skipped.append(key)
                continue
            if key not in current_state or current_state[key].shape != value.shape:
                skipped.append(key)
                continue
            loaded_state[key] = value
        raw_model.load_state_dict(loaded_state, strict=False)
        print(
            json.dumps(
                {
                    "event": "checkpoint_model_initialized",
                    "path": str(checkpoint_path),
                    "loaded_tensors": len(loaded_state),
                    "skipped_tensors": len(skipped),
                    "reset_action_head": True,
                    "optimizer_loaded": False,
                }
            )
        )
    else:
        raw_model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
    if not reset_action_head and scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    step = int(payload.get("step", 0))
    print(json.dumps({"event": "checkpoint_loaded", "step": step, "path": str(checkpoint_path)}))
    return step


def main() -> None:
    args = parse_args()
    if (
        args.head_generator_symmetric_cap is not None
        and args.head_generator_symmetric_cap <= 0
    ):
        raise SystemExit("--head_generator_symmetric_cap must be positive")
    distributed, rank, local_rank, world_size, device = setup_distributed(args)
    torch.manual_seed(args.seed + rank)
    np_seed = args.seed % (2**32)
    try:
        import numpy as np

        np.random.seed((np_seed + rank) % (2**32))
    except Exception:
        pass

    config = make_config(args)
    if is_main_process(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2)
    if distributed:
        dist.barrier()

    train_dataset, val_dataset = make_datasets(args)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed and val_dataset is not None
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        if val_dataset is not None
        else None
    )

    model = VLATransformerPolicy(config).to(device)
    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DistributedDataParallel(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.precision == "fp16")

    step = 0
    if args.resume_checkpoint is not None:
        step = load_checkpoint(
            args.resume_checkpoint,
            model,
            optimizer,
            scaler,
            config,
            device,
            reset_action_head=args.reset_action_head_on_resume,
        )
    if distributed:
        dist.barrier()

    raw_model = unwrap_model(model)
    wandb_run = None
    if is_main_process(rank):
        first_attn = raw_model.attention_modules[0]
        accounting = attention_accounting(
            first_attn,
            sequence_length=config.text_length + config.num_views + 1 + config.action_horizon,
            batch_size=args.batch_size,
        )
        print(f"model_parameters={count_parameters(raw_model):,}")
        print(f"first_attention_accounting={accounting}")
        print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset) if val_dataset is not None else 0}")
        print(f"distributed={distributed} world_size={world_size} device={device}")
        wandb_run = init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            output_dir=args.wandb_dir or args.output_dir,
            config={
                "args": vars(args),
                "model_config": config.to_dict(),
                "attention_accounting": accounting.__dict__,
                "parameters": count_parameters(raw_model),
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset) if val_dataset is not None else 0,
                "distributed": distributed,
                "world_size": world_size,
                "resume_step": step,
            },
        )

    log_path = args.output_dir / "metrics.jsonl"
    model.train()
    epoch = step // max(len(train_loader), 1)
    start = time.time()
    try:
        while step < args.steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch += 1
            for batch in train_loader:
                step += 1
                optimizer.zero_grad(set_to_none=True)
                loss, loss_dict = loss_from_batch(model, batch, device, args.precision)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if args.max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                if step % args.log_every == 0 or step == 1:
                    reduced_loss = distributed_mean(loss.detach(), distributed)
                    reduced_parts = {
                        key: distributed_mean(value.detach(), distributed)
                        for key, value in loss_dict.items()
                    }
                    if is_main_process(rank):
                        logs: dict[str, object] = {
                            "step": step,
                            "loss_total": float(reduced_loss.cpu()),
                            "elapsed_sec": time.time() - start,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        logs.update({key: float(value.cpu()) for key, value in reduced_parts.items()})
                        logs.update(lgma_diagnostics(raw_model))
                        print(json.dumps(logs, sort_keys=True))
                        write_jsonl(log_path, logs)
                        log_wandb(wandb_run, logs, step=step)

                if val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                    logs = {"step": step, **evaluate(model, val_loader, device, args.precision, args.eval_batches, distributed)}
                    if is_main_process(rank):
                        print(json.dumps(logs, sort_keys=True))
                        write_jsonl(log_path, logs)
                        log_wandb(wandb_run, logs, step=step)

                if is_main_process(rank) and args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(model, optimizer, config, args.output_dir, step, scaler)

                if step >= args.steps:
                    break

        if is_main_process(rank):
            save_checkpoint(model, optimizer, config, args.output_dir, step, scaler)
            log_wandb(wandb_run, {"final_step": step}, step=step)
    finally:
        if is_main_process(rank):
            finish_wandb(wandb_run)
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
