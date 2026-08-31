from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgma.vision import (
    VISION_ATTENTION_TYPES,
    DeiTConfig,
    DeiTClassifier,
    vision_parameter_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train controlled DeiT-B attention variants on ImageNet-1K"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-format", choices=("wds", "imagefolder"), default="wds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-type", choices=sorted(VISION_ATTENTION_TYPES), default="mha")
    parser.add_argument("--reduced-qk-dim", type=int, default=384)
    parser.add_argument("--collaborative-qk-dim", type=int, default=384)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-base-heads", type=int, default=4)
    parser.add_argument("--num-generators", type=int, default=8)
    parser.add_argument("--generator-mixing", choices=("softmax", "none"), default="softmax")
    parser.add_argument(
        "--theta-init",
        choices=("balanced_simplex", "random_sphere", "circle"),
        default="balanced_simplex",
    )
    parser.add_argument("--theta-init-scale", type=float, default=4.0)
    parser.add_argument("--generator-init-scale", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256, help="Per-device batch size")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--base-learning-rate", type=float, default=5e-4)
    parser.add_argument("--reference-batch-size", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        help="Optionally clip the global gradient norm; norms are logged regardless.",
    )
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--repeated-augmentation",
        type=int,
        default=0,
        help=(
            "RepeatAugment count for indexable ImageFolder datasets. Keep at 0 "
            "for iterable WebDataset shards, which timm cannot repeat-sample."
        ),
    )
    parser.add_argument("--model-ema-decay", type=float, default=0.99996)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--minimum-final-top1",
        type=float,
        help="Fail the job when the final validation top-1 is below this smoke-test threshold.",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--wandb-project", default="gt-mha-imagenet")
    parser.add_argument("--wandb-entity", default="i2rLAB")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def is_primary(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dependencies() -> tuple[Any, ...]:
    try:
        from timm.data import Mixup, create_dataset, create_loader
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        from timm.loss import SoftTargetCrossEntropy
        from timm.utils import ModelEmaV2
    except ImportError as error:
        raise SystemExit("Install with: pip install -e '.[vision,tracking]'") from error
    return (
        Mixup,
        create_dataset,
        create_loader,
        IMAGENET_DEFAULT_MEAN,
        IMAGENET_DEFAULT_STD,
        SoftTargetCrossEntropy,
        ModelEmaV2,
    )


def model_config(args: argparse.Namespace) -> DeiTConfig:
    return DeiTConfig(
        attention_type=args.attention_type,
        reduced_qk_dim=args.reduced_qk_dim,
        collaborative_qk_dim=args.collaborative_qk_dim,
        num_kv_heads=args.num_kv_heads,
        num_base_heads=args.num_base_heads,
        num_generators=args.num_generators,
        generator_mixing=args.generator_mixing,
        theta_init=args.theta_init,
        theta_init_scale=args.theta_init_scale,
        generator_init_scale=args.generator_init_scale,
        drop_path_rate=args.drop_path_rate,
    )


def optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    no_decay_names = set(getattr(model, "no_weight_decay", lambda: set())())
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim <= 1
            or name in no_decay_names
            or name.endswith((".theta", ".value_theta", ".mixing_vector"))
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _wds_split(data_dir: Path, split: str) -> tuple[str, int]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing WebDataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if split == "train":
        if manifest.get("train_order") != "deterministic_random_shuffle":
            raise ValueError(
                "training shards are not marked as deterministically shuffled"
            )
        shard_records = manifest["splits"]["train"].get("shards", [])
        if not shard_records or any("unique_classes" not in row for row in shard_records):
            raise ValueError("training shard class-diversity audit is missing")
        num_classes = int(manifest.get("num_classes", 0))
        minimum_classes = min(32, num_classes)
        poorly_mixed = [
            row["name"]
            for row in shard_records
            if int(row["samples"]) >= 128
            and int(row["unique_classes"]) < minimum_classes
        ]
        if poorly_mixed:
            raise ValueError(
                "training shards are insufficiently class-mixed: "
                + ", ".join(poorly_mixed[:5])
            )
    split_manifest = manifest["splits"][split]
    return f'{split_manifest["pattern"]}|{split_manifest["samples"]}', int(
        split_manifest["samples"]
    )


def create_data_loaders(
    args: argparse.Namespace,
    *,
    world_size: int,
    device: torch.device,
) -> tuple[Any, Any, int, int, Any]:
    if args.dataset_format == "wds" and args.repeated_augmentation:
        raise ValueError(
            "timm RepeatAugment requires an indexable dataset; use "
            "--repeated-augmentation 0 with WebDataset"
        )
    (
        Mixup,
        create_dataset,
        create_loader,
        mean,
        std,
        _,
        _,
    ) = dependencies()
    distributed = world_size > 1
    if args.dataset_format == "wds":
        train_split, train_samples = _wds_split(args.data_dir, "train")
        val_split, val_samples = _wds_split(args.data_dir, "val")
        train_dataset = create_dataset(
            "wds/",
            root=str(args.data_dir),
            split=train_split,
            is_training=True,
            batch_size=args.batch_size,
            seed=args.seed,
            num_samples=train_samples,
        )
        val_dataset = create_dataset(
            "wds/",
            root=str(args.data_dir),
            split=val_split,
            is_training=False,
            batch_size=args.batch_size,
            seed=args.seed,
            num_samples=val_samples,
        )
        configure_exact_wds_validation(val_dataset)
    else:
        train_dataset = create_dataset(
            "",
            root=str(args.data_dir),
            split="train",
            is_training=True,
            batch_size=args.batch_size,
            seed=args.seed,
            search_split=True,
        )
        val_dataset = create_dataset(
            "",
            root=str(args.data_dir),
            split="val",
            is_training=False,
            batch_size=args.batch_size,
            seed=args.seed,
            search_split=True,
        )
        train_samples, val_samples = len(train_dataset), len(val_dataset)

    common = {
        "input_size": (3, 224, 224),
        "use_prefetcher": False,
        "mean": mean,
        "std": std,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        is_training=True,
        no_aug=False,
        scale=(0.08, 1.0),
        ratio=(0.75, 4.0 / 3.0),
        hflip=0.5,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        num_aug_repeats=args.repeated_augmentation,
        distributed=distributed,
        **common,
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=args.batch_size,
        is_training=False,
        interpolation="bicubic",
        crop_pct=0.875,
        # WDS pads distributed validation to an equal number of samples per
        # worker. Keep validation exact and run it on rank zero instead.
        distributed=False,
        **common,
    )
    mixup = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.label_smoothing,
        num_classes=1000,
    )
    return train_loader, val_loader, train_samples, val_samples, mixup


def configure_exact_wds_validation(dataset: Any) -> None:
    """Make a rank-zero WDS validation loader read every shard exactly once.

    ``ReaderWds`` captures the initialized distributed world when constructed.
    Passing ``distributed=False`` to timm's loader does not undo that reader
    split. Since ``evaluate_exact`` intentionally evaluates only on rank zero,
    reset the still-lazy validation reader to one replica before workers start.
    """
    reader = getattr(dataset, "reader", None)
    if reader is None:
        raise TypeError("WebDataset validation dataset does not expose a reader")
    if getattr(reader, "ds", None) is not None:
        raise RuntimeError("WebDataset validation reader was initialized too early")
    for name, value in (
        ("dist_rank", 0),
        ("dist_num_replicas", 1),
        ("global_worker_id", 0),
        ("global_num_workers", 1),
    ):
        if not hasattr(reader, name):
            raise TypeError(f"WebDataset reader is missing {name}")
        setattr(reader, name, value)


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    warmup_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        fraction = step / max(warmup_steps, 1)
        learning_rate = warmup_lr + fraction * (base_lr - warmup_lr)
    else:
        fraction = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        fraction = min(max(fraction, 0.0), 1.0)
        learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * fraction)
        )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def gradient_l2_norm(model: nn.Module) -> float:
    squared: torch.Tensor | None = None
    for parameter in model.parameters():
        if parameter.grad is not None:
            term = parameter.grad.detach().float().square().sum()
            squared = term if squared is None else squared + term
    return 0.0 if squared is None else float(squared.sqrt())


def parameter_l2_norm(model: nn.Module) -> float:
    squared: torch.Tensor | None = None
    for parameter in model.parameters():
        term = parameter.detach().float().square().sum()
        squared = term if squared is None else squared + term
    return 0.0 if squared is None else float(squared.sqrt())


def reduce_statistics(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    precision: str,
    max_batches: int | None,
    *,
    reduce_distributed: bool = True,
) -> dict[str, float]:
    model.eval()
    loss_sum = correct1 = correct5 = samples = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    started = time.time()
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, precision):
            logits = model(images)
            loss_sum += float(criterion(logits, targets))
        predictions = logits.topk(5, dim=1).indices
        correct = predictions.eq(targets[:, None])
        correct1 += float(correct[:, :1].sum())
        correct5 += float(correct.sum())
        samples += targets.numel()
    if reduce_distributed:
        loss_sum, correct1, correct5, samples = reduce_statistics(
            [loss_sum, correct1, correct5, samples], device
        )
    elapsed = max(time.time() - started, 1e-6)
    return {
        "loss": loss_sum / max(samples, 1),
        "top1": 100.0 * correct1 / max(samples, 1),
        "top5": 100.0 * correct5 / max(samples, 1),
        "samples": samples,
        "images_per_second": samples / elapsed,
    }


def evaluate_exact(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    precision: str,
    max_batches: int | None,
    rank: int,
) -> dict[str, float]:
    """Evaluate each validation example once, then broadcast rank-zero metrics."""
    keys = ("loss", "top1", "top5", "samples", "images_per_second")
    if rank == 0:
        metrics = evaluate(
            model,
            loader,
            device,
            precision,
            max_batches,
            reduce_distributed=False,
        )
        values = [metrics[key] for key in keys]
    else:
        values = [0.0] * len(keys)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return dict(zip(keys, tensor.cpu().tolist()))


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_ema: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_top1: float,
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "model_ema": model_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_top1": best_top1,
            "args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.max_gradient_norm is not None and args.max_gradient_norm <= 0:
        raise SystemExit("--max-gradient-norm must be positive")
    if args.minimum_final_top1 is not None and args.minimum_final_top1 < 0:
        raise SystemExit("--minimum-final-top1 must be non-negative")
    rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed, rank)
    config = model_config(args)
    model = DeiTClassifier(config)
    counts = vision_parameter_counts(model)
    if is_primary(rank):
        print(json.dumps({"config": model.configuration(), "parameters": counts}, indent=2))
    if args.dry_run:
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    (
        _,
        _,
        _,
        _,
        _,
        SoftTargetCrossEntropy,
        ModelEmaV2,
    ) = dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, train_samples, _, mixup = create_data_loaders(
        args, world_size=world_size, device=device
    )
    effective_global_batch = args.batch_size * world_size
    base_lr = args.base_learning_rate * effective_global_batch / args.reference_batch_size
    steps_per_epoch = len(train_loader)
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    model.to(device)
    model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device=None)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.weight_decay),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    start_epoch = global_step = 0
    best_top1 = -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model_ema.module.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_top1 = float(checkpoint.get("best_top1", -math.inf))
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    criterion = SoftTargetCrossEntropy()
    wandb_run = None
    if is_primary(rank) and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config={
                    **vars(args),
                    **counts,
                    "effective_global_batch": effective_global_batch,
                    "effective_learning_rate": base_lr,
                    "train_samples": train_samples,
                },
                dir=str(args.output_dir),
                id=args.wandb_run_name,
                resume="allow",
            )
        except ImportError:
            print("wandb is not installed; continuing without experiment tracking")

    if args.eval_only:
        metrics = evaluate_exact(
            model_ema.module,
            val_loader,
            device,
            args.precision,
            args.max_val_batches,
            rank,
        )
        if is_primary(rank):
            print(json.dumps({"validation": metrics}, indent=2))
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    if is_primary(rank):
        (args.output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "model": model_config(args).__dict__,
                    "parameters": counts,
                    "effective_global_batch": effective_global_batch,
                    "effective_learning_rate": base_lr,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    last_validation_top1: float | None = None
    for epoch in range(start_epoch, args.epochs):
        # Epoch-scoped seeds keep main-process Mixup and stochastic-depth draws
        # stable when a 48-hour allocation resumes at an epoch boundary.
        seed_everything(args.seed + 10_000 * epoch, rank)
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.time()
        loss_sum = sample_count = 0.0
        for batch_index, (images, targets) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            learning_rate = set_learning_rate(
                optimizer,
                global_step,
                total_steps,
                warmup_steps,
                base_lr,
                args.warmup_learning_rate,
                args.min_learning_rate,
            )
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images, targets = mixup(images, targets)
            next_global_step = global_step + 1
            measure_diagnostics = (
                is_primary(rank) and next_global_step % args.log_interval == 0
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                logits = model(images)
                loss = criterion(logits, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.max_gradient_norm is not None or measure_diagnostics:
                    scaler.unscale_(optimizer)
                if args.max_gradient_norm is not None:
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_gradient_norm
                        )
                    )
                elif measure_diagnostics:
                    gradient_norm = gradient_l2_norm(model)
                else:
                    gradient_norm = None
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_gradient_norm is not None:
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_gradient_norm
                        )
                    )
                elif measure_diagnostics:
                    gradient_norm = gradient_l2_norm(model)
                else:
                    gradient_norm = None
                optimizer.step()
            model_ema.update(unwrap_model(model))
            batch_samples = images.shape[0]
            loss_sum += float(loss.detach()) * batch_samples
            sample_count += batch_samples
            global_step += 1
            if is_primary(rank) and global_step % args.log_interval == 0:
                payload = {
                    "epoch": epoch,
                    "step": global_step,
                    "train/loss": float(loss.detach()),
                    "train/epoch_loss_so_far": loss_sum / max(sample_count, 1),
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": gradient_norm,
                    "train/parameter_norm": parameter_l2_norm(unwrap_model(model)),
                }
                print(json.dumps(payload), flush=True)
                if wandb_run is not None:
                    wandb_run.log(payload, step=global_step)

        train_loss_sum, trained_samples = reduce_statistics(
            [loss_sum, sample_count], device
        )
        train_elapsed = max(time.time() - epoch_started, 1e-6)
        validation = evaluate_exact(
            model_ema.module,
            val_loader,
            device,
            args.precision,
            args.max_val_batches,
            rank,
        )
        last_validation_top1 = validation["top1"]
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        )
        summary = {
            "epoch": epoch,
            "step": global_step,
            "train/loss": train_loss_sum / max(trained_samples, 1),
            "train/images_per_second": trained_samples / train_elapsed,
            "train/peak_memory_gib": peak_memory,
            "validation/loss": validation["loss"],
            "validation/top1": validation["top1"],
            "validation/top5": validation["top5"],
            "validation/samples": validation["samples"],
            "validation/images_per_second": validation["images_per_second"],
        }
        if is_primary(rank):
            print(json.dumps(summary), flush=True)
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            improved = validation["top1"] > best_top1
            best_top1 = max(best_top1, validation["top1"])
            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_top1=best_top1,
                args=args,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "checkpoint_best.pt",
                    model=model,
                    model_ema=model_ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_top1=best_top1,
                    args=args,
                )
            if wandb_run is not None:
                wandb_run.log(summary, step=global_step)
        if dist.is_initialized():
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()
    if (
        args.minimum_final_top1 is not None
        and (
            last_validation_top1 is None
            or last_validation_top1 < args.minimum_final_top1
        )
    ):
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(
            f"final validation top-1 {last_validation_top1} is below required "
            f"{args.minimum_final_top1}"
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
