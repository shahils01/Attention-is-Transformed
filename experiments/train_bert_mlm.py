from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lgma.bert import GT_MHA_BERT_ATTENTION_TYPES, bert_parameter_counts, load_bert_masked_lm


HEAD_COORDINATE_PARAMETER_NAMES = {"theta", "value_theta"}


def fixed_validation_mlm_mask(
    batch: dict[str, list[list[int]]],
    indices: list[int],
    *,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    seed: int,
) -> dict[str, list[list[int]]]:
    """Apply deterministic BERT 80/10/10 masking keyed by example index."""
    masked_inputs: list[list[int]] = []
    labels: list[list[int]] = []
    special_masks = batch.get("special_tokens_mask")
    if special_masks is None:
        raise ValueError("fixed MLM masking requires special_tokens_mask")
    if len(batch["input_ids"]) != len(indices) or len(special_masks) != len(indices):
        raise ValueError("MLM batch and example indices must have matching lengths")

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
    """Use dynamic masking for training and precomputed masks for evaluation."""

    def __init__(self, train_collator: Any, fixed_collator: Any) -> None:
        self.train_collator = train_collator
        self.fixed_collator = fixed_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            return self.fixed_collator(features)
        fixed = ["labels" in feature for feature in features]
        if any(fixed) and not all(fixed):
            raise ValueError("cannot collate a mixture of dynamic and fixed MLM examples")
        collator = self.fixed_collator if all(fixed) else self.train_collator
        return collator(features)


def is_head_coordinate_parameter(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in HEAD_COORDINATE_PARAMETER_NAMES


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    get_parameter_names: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build Hugging Face-style groups with no decay on GT-MHA coordinates."""
    decay_names = set(get_parameter_names(model, [nn.LayerNorm]))
    decay_names = {name for name in decay_names if "bias" not in name}
    coordinate_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and is_head_coordinate_parameter(name)
    )
    decay_names.difference_update(coordinate_names)
    groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and name in decay_names
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and name not in decay_names
            ],
            "weight_decay": 0.0,
        },
    ]
    return groups, coordinate_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BERT MLM training with official MHA or paper GT-MHA")
    p.add_argument("--model-name-or-path", default="google-bert/bert-base-uncased")
    p.add_argument("--attention-type", choices=["mha", *sorted(GT_MHA_BERT_ATTENTION_TYPES)], default="mha")
    p.add_argument("--initialization", choices=["checkpoint", "random"], default="checkpoint")
    p.add_argument("--num-base-heads", type=int, default=4)
    p.add_argument("--num-generators", type=int, default=8)
    p.add_argument(
        "--generator-mixing", "--generator_mixing",
        dest="generator_mixing", choices=["softmax", "none"], default="softmax",
    )
    p.add_argument("--use-sdpa", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fuse-base-qkv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sdpa-gqa-mode", choices=["auto", "native", "expand"], default="auto")
    p.add_argument("--enforce-paper-gt-mha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset-name", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-sequence-length", type=int, default=128)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument(
        "--validation-mask-seed",
        type=int,
        default=17_029,
        help="Model-independent seed for the fixed masked validation set.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-device-train-batch-size", type=int, default=32)
    p.add_argument("--per-device-eval-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=1_000)
    p.add_argument("--save-steps", type=int, default=1_000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--run-name")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def dependencies() -> tuple[Any, ...]:
    try:
        from datasets import load_dataset
        from transformers import (
            AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator,
            set_seed, Trainer, TrainingArguments,
        )
        from transformers.trainer_pt_utils import get_parameter_names
    except ImportError as exc:
        raise SystemExit("Install with: pip install -e '.[bert,tracking]'") from exc
    return (
        load_dataset,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        default_data_collator,
        get_parameter_names,
        set_seed,
        Trainer,
        TrainingArguments,
    )


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("--bf16 and --fp16 are mutually exclusive")
    if not 0.0 < args.mlm_probability < 1.0:
        raise SystemExit("--mlm-probability must be in (0, 1)")
    if args.validation_mask_seed < 0:
        raise SystemExit("--validation-mask-seed must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        load_dataset,
        AutoTokenizer,
        Collator,
        default_data_collator,
        get_parameter_names,
        set_seed,
        Trainer,
        TrainingArguments,
    ) = dependencies()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model, audit = load_bert_masked_lm(
        args.model_name_or_path, attention_type=args.attention_type,
        initialization=args.initialization, num_base_heads=args.num_base_heads,
        num_generators=args.num_generators, generator_mixing=args.generator_mixing,
        use_sdpa=args.use_sdpa, fuse_base_qkv=args.fuse_base_qkv,
        sdpa_gqa_mode=args.sdpa_gqa_mode,
        enforce_paper_gt_mha=args.enforce_paper_gt_mha,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    parameter_counts = bert_parameter_counts(model)
    for name, value in parameter_counts.items():
        setattr(model.config, name, value)
    dataset_args = {"name": args.dataset_config} if args.dataset_config else {}
    raw = load_dataset(args.dataset_name, **dataset_args)
    missing = {s for s in (args.train_split, args.validation_split) if s not in raw}
    if missing:
        raise ValueError(f"dataset is missing splits: {sorted(missing)}")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch[args.text_column], return_special_tokens_mask=True)

    tokenized = raw.map(
        tokenize, batched=True, remove_columns=raw[args.train_split].column_names,
        desc="Tokenizing BERT MLM corpus",
    )

    def group(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        joined = {key: sum(values, []) for key, values in batch.items()}
        length = len(joined["input_ids"])
        length = (length // args.max_sequence_length) * args.max_sequence_length
        return {
            key: [values[i:i + args.max_sequence_length] for i in range(0, length, args.max_sequence_length)]
            for key, values in joined.items()
        }

    tokenized = tokenized.map(group, batched=True, desc="Packing fixed-length sequences")
    if tokenizer.mask_token_id is None:
        raise ValueError("the tokenizer must define a mask token for MLM training")
    fixed_validation = tokenized[args.validation_split].map(
        lambda batch, indices: fixed_validation_mlm_mask(
            batch,
            indices,
            mask_token_id=int(tokenizer.mask_token_id),
            vocab_size=len(tokenizer),
            mlm_probability=args.mlm_probability,
            seed=args.validation_mask_seed,
        ),
        batched=True,
        with_indices=True,
        remove_columns=["special_tokens_mask"],
        desc="Creating fixed masked MLM validation set",
    )
    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "max_steps": args.max_steps,
        "logging_steps": args.logging_steps, "eval_steps": args.eval_steps,
        "save_steps": args.save_steps, "save_total_limit": args.save_total_limit,
        "seed": args.seed, "data_seed": args.seed,
        "dataloader_num_workers": args.num_workers, "bf16": args.bf16,
        "fp16": args.fp16, "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "run_name": args.run_name, "remove_unused_columns": False,
    }
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs[key] = "steps"
    training_args = TrainingArguments(**kwargs)
    optimizer_groups, coordinate_names = optimizer_parameter_groups(
        model,
        weight_decay=args.weight_decay,
        get_parameter_names=get_parameter_names,
    )
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
    trainer_kwargs = {
        "model": model, "args": training_args,
        "train_dataset": tokenized[args.train_split],
        "eval_dataset": fixed_validation,
        "data_collator": TrainEvalMLMCollator(
            Collator(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability),
            default_data_collator,
        ),
        "optimizers": (optimizer, None),
    }
    token_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[token_key] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    manifest = {
        "model_name_or_path": args.model_name_or_path, "initialization": args.initialization,
        "attention_type": args.attention_type, "num_base_heads": args.num_base_heads,
        "num_generators": args.num_generators, "generator_mixing": args.generator_mixing,
        "use_sdpa": args.use_sdpa, "fuse_base_qkv": args.fuse_base_qkv,
        "sdpa_gqa_mode": args.sdpa_gqa_mode,
        "enforce_paper_gt_mha": args.enforce_paper_gt_mha,
        "validation_mask_seed": args.validation_mask_seed,
        "validation_masking": "fixed_per_example_bert_80_10_10",
        "head_coordinate_weight_decay": 0.0,
        "head_coordinate_parameters": coordinate_names,
        "teacher_student_distillation": False, "replacement_audit": audit,
        "parameter_counts": parameter_counts,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "bert_gt_mha_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    if audit:
        torch.save(model.state_dict(), args.output_dir / "gt_mha_state_dict.pt")
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
