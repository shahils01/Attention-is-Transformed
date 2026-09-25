#!/usr/bin/env python3
"""Short, matched BERT continued-MLM pilot with transform-diversity regularization."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-num-bases", type=int, required=True)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--margin-multiplier", type=float, default=1.25)
    parser.add_argument("--minimum-margin", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91_337)
    parser.add_argument("--validation-mask-seed", type=int, default=17_029)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def fixed_validation_mlm_mask(
    batch: dict[str, list[list[int]]],
    indices: list[int],
    *,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    seed: int,
) -> dict[str, list[list[int]]]:
    masked_inputs: list[list[int]] = []
    labels: list[list[int]] = []
    special_masks = batch.get("special_tokens_mask")
    if special_masks is None:
        raise ValueError("fixed MLM masking requires special_tokens_mask")
    for input_ids, special_mask, example_index in zip(
        batch["input_ids"], special_masks, indices
    ):
        generator = torch.Generator().manual_seed(seed + int(example_index))
        inputs = torch.tensor(input_ids, dtype=torch.long)
        targets = inputs.clone()
        probability = torch.full(inputs.shape, mlm_probability, dtype=torch.float32)
        probability.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), 0.0)
        selected = torch.bernoulli(probability, generator=generator).bool()
        targets[~selected] = -100
        replaced = (
            torch.bernoulli(torch.full(inputs.shape, 0.8), generator=generator).bool()
            & selected
        )
        inputs[replaced] = mask_token_id
        random_replaced = (
            torch.bernoulli(torch.full(inputs.shape, 0.5), generator=generator).bool()
            & selected
            & ~replaced
        )
        random_tokens = torch.randint(
            vocab_size, inputs.shape, generator=generator, dtype=torch.long
        )
        inputs[random_replaced] = random_tokens[random_replaced]
        masked_inputs.append(inputs.tolist())
        labels.append(targets.tolist())
    return {"input_ids": masked_inputs, "labels": labels}


class TrainEvalMLMCollator:
    def __init__(self, train_collator: Any, fixed_collator: Any) -> None:
        self.train_collator = train_collator
        self.fixed_collator = fixed_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        fixed = ["labels" in feature for feature in features]
        if any(fixed) and not all(fixed):
            raise ValueError("cannot mix dynamically and statically masked examples")
        return (self.fixed_collator if all(fixed) else self.train_collator)(features)


def gt_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if all(
            hasattr(module, name)
            for name in (
                "compute_metrics",
                "compute_value_transforms",
                "num_base_heads",
                "generated_heads_per_base",
            )
        ):
            modules.append(module)
    if not modules:
        raise ValueError("no GT-MHA modules found")
    return modules


def within_base_distances(
    transforms: torch.Tensor,
    num_bases: int,
    heads_per_base: int,
) -> torch.Tensor:
    if transforms.shape[0] != num_bases * heads_per_base:
        raise ValueError("transform count does not match base assignment")
    grouped = transforms.float().reshape(num_bases, heads_per_base, -1)
    differences = grouped[:, :, None, :] - grouped[:, None, :, :]
    distances = differences.norm(dim=-1) / math.sqrt(transforms.shape[-1])
    mask = torch.triu(
        torch.ones(heads_per_base, heads_per_base, device=transforms.device, dtype=torch.bool),
        diagonal=1,
    )
    return distances[:, mask]


@torch.no_grad()
def transformation_statistics(model: torch.nn.Module) -> dict[str, float]:
    values: dict[str, list[torch.Tensor]] = {"qk": [], "value": []}
    for module in gt_modules(model):
        num_bases = int(module.num_base_heads)
        heads_per_base = int(module.generated_heads_per_base)
        values["qk"].append(
            within_base_distances(module.compute_metrics(), num_bases, heads_per_base).cpu()
        )
        values["value"].append(
            within_base_distances(
                module.compute_value_transforms(), num_bases, heads_per_base
            ).cpu()
        )
    report: dict[str, float] = {}
    for pathway, tensors in values.items():
        distances = torch.cat([value.reshape(-1) for value in tensors])
        report[f"{pathway}_within_distance_mean"] = float(distances.mean())
        report[f"{pathway}_within_distance_median"] = float(distances.median())
        report[f"{pathway}_within_distance_min"] = float(distances.min())
        report[f"{pathway}_within_distance_max"] = float(distances.max())
    return report


def make_margins(
    model: torch.nn.Module,
    multiplier: float,
    minimum: float,
) -> dict[int, dict[str, float]]:
    margins: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for module in gt_modules(model):
            num_bases = int(module.num_base_heads)
            heads_per_base = int(module.generated_heads_per_base)
            qk = within_base_distances(
                module.compute_metrics(), num_bases, heads_per_base
            )
            value = within_base_distances(
                module.compute_value_transforms(), num_bases, heads_per_base
            )
            margins[id(module)] = {
                "qk": max(minimum, multiplier * float(qk.median())),
                "value": max(minimum, multiplier * float(value.median())),
            }
    return margins


def diversity_margin_loss(
    model: torch.nn.Module,
    margins: dict[int, dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qk_penalties = []
    value_penalties = []
    for module in gt_modules(model):
        num_bases = int(module.num_base_heads)
        heads_per_base = int(module.generated_heads_per_base)
        qk_distances = within_base_distances(
            module.compute_metrics(), num_bases, heads_per_base
        )
        value_distances = within_base_distances(
            module.compute_value_transforms(), num_bases, heads_per_base
        )
        qk_margin = margins[id(module)]["qk"]
        value_margin = margins[id(module)]["value"]
        qk_penalties.append(
            (torch.relu(qk_margin - qk_distances) / qk_margin).square().mean()
        )
        value_penalties.append(
            (torch.relu(value_margin - value_distances) / value_margin).square().mean()
        )
    qk_loss = torch.stack(qk_penalties).mean()
    value_loss = torch.stack(value_penalties).mean()
    return 0.5 * (qk_loss + value_loss), qk_loss, value_loss


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1 and args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    if args.expected_num_bases < 1:
        raise SystemExit("--expected-num-bases must be positive")
    if args.diversity_weight < 0.0:
        raise SystemExit("--diversity-weight must be non-negative")
    if args.margin_multiplier <= 1.0:
        raise SystemExit("--margin-multiplier must exceed 1")

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )
    from transformers.trainer_pt_utils import get_parameter_names
    from lgma.bert import (
        bert_optimizer_parameter_groups,
        load_bert_masked_lm,
    )

    set_seed(args.seed)
    checkpoint_root = args.checkpoint_root.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    model, audit = load_bert_masked_lm(str(checkpoint_root))
    modules = gt_modules(model)
    observed_bases = {int(module.num_base_heads) for module in modules}
    if observed_bases != {args.expected_num_bases}:
        raise ValueError(
            f"expected C={args.expected_num_bases}, checkpoint contains {sorted(observed_bases)}"
        )
    model.gradient_checkpointing_enable()
    initial_transform_stats = transformation_statistics(model)
    margins = make_margins(
        model, multiplier=args.margin_multiplier, minimum=args.minimum_margin
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight-passed",
                    "checkpoint_root": str(checkpoint_root),
                    "expected_num_bases": args.expected_num_bases,
                    "layers": len(modules),
                    "checkpoint_state_sha256": sha256(
                        checkpoint_root / "gt_mha_state_dict.pt"
                    ),
                    "initial_transform_statistics": initial_transform_stats,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    args.output_dir.mkdir(parents=True)

    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], return_special_tokens_mask=True)

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing BERT MLM corpus",
    )

    def group(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        joined = {key: sum(values, []) for key, values in batch.items()}
        length = (len(joined["input_ids"]) // 128) * 128
        return {
            key: [values[index : index + 128] for index in range(0, length, 128)]
            for key, values in joined.items()
        }

    tokenized = tokenized.map(group, batched=True, desc="Packing fixed-length sequences")
    fixed_validation = tokenized["validation"].map(
        lambda batch, indices: fixed_validation_mlm_mask(
            batch,
            indices,
            mask_token_id=int(tokenizer.mask_token_id),
            vocab_size=len(tokenizer),
            mlm_probability=0.15,
            seed=args.validation_mask_seed,
        ),
        batched=True,
        with_indices=True,
        remove_columns=["special_tokens_mask"],
        desc="Creating fixed masked MLM validation set",
    )

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "logging_steps": 50,
        "save_strategy": "no",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.num_workers,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": [],
        "remove_unused_columns": False,
    }
    evaluation_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[evaluation_key] = "no"
    training_args = TrainingArguments(**training_kwargs)
    optimizer_groups, coordinate_names = bert_optimizer_parameter_groups(
        model, weight_decay=0.01, get_parameter_names=get_parameter_names
    )
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)

    class DiversityTrainer(Trainer):
        def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, Any]:
            del num_items_in_batch
            outputs = model(**inputs)
            task_loss = outputs.loss
            loss = task_loss
            if model.training and args.diversity_weight > 0.0:
                regularizer, _, _ = diversity_margin_loss(model, margins)
                loss = task_loss + args.diversity_weight * regularizer
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": fixed_validation,
        "data_collator": TrainEvalMLMCollator(
            DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            ),
            default_data_collator,
        ),
        "optimizers": (optimizer, None),
    }
    token_key = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[token_key] = tokenizer
    trainer = DiversityTrainer(**trainer_kwargs)

    initial_eval = trainer.evaluate(metric_key_prefix="initial")
    result = trainer.train()
    final_eval = trainer.evaluate(metric_key_prefix="final")
    final_transform_stats = transformation_statistics(model)
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        torch.save(model.state_dict(), args.output_dir / "gt_mha_state_dict.pt")

    protocol = {
        "source_root": str(source_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_state_sha256": sha256(checkpoint_root / "gt_mha_state_dict.pt"),
        "source_bert_sha256": sha256(source_root / "src" / "lgma" / "bert.py"),
        "source_attention_sha256": sha256(
            source_root / "src" / "lgma" / "attention.py"
        ),
        "expected_num_bases": args.expected_num_bases,
        "continuation_seed": args.seed,
        "max_steps": args.max_steps,
        "effective_batch_size": (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * world_size
        ),
        "world_size": world_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "diversity_weight": args.diversity_weight,
        "regularizer": "same-base normalized Frobenius margin on QK and value transforms",
        "margin_multiplier": args.margin_multiplier,
        "minimum_margin": args.minimum_margin,
        "head_coordinate_parameters": coordinate_names,
        "replacement_audit": audit,
        "initial_transform_statistics": initial_transform_stats,
        "final_transform_statistics": final_transform_stats,
        "initial_evaluation": initial_eval,
        "final_evaluation": final_eval,
        "training_metrics": result.metrics,
    }
    if trainer.is_world_process_zero():
        (args.output_dir / "pilot_results.json").write_text(
            json.dumps(protocol, indent=2) + "\n"
        )
        print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
