from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lgma.bert import GT_MHA_BERT_ATTENTION_TYPES, bert_parameter_counts, load_bert_masked_lm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BERT MLM training with official MHA or paper GT-MHA")
    p.add_argument("--model-name-or-path", default="google-bert/bert-base-uncased")
    p.add_argument("--attention-type", choices=["mha", *sorted(GT_MHA_BERT_ATTENTION_TYPES)], default="mha")
    p.add_argument("--initialization", choices=["checkpoint", "random"], default="checkpoint")
    p.add_argument("--num-base-heads", type=int, default=4)
    p.add_argument("--num-generators", type=int, default=8)
    p.add_argument("--enforce-paper-gt-mha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset-name", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-sequence-length", type=int, default=128)
    p.add_argument("--mlm-probability", type=float, default=0.15)
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
            AutoTokenizer, DataCollatorForLanguageModeling, set_seed, Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install with: pip install -e '.[bert,tracking]'") from exc
    return load_dataset, AutoTokenizer, DataCollatorForLanguageModeling, set_seed, Trainer, TrainingArguments


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("--bf16 and --fp16 are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_dataset, AutoTokenizer, Collator, set_seed, Trainer, TrainingArguments = dependencies()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model, audit = load_bert_masked_lm(
        args.model_name_or_path, attention_type=args.attention_type,
        initialization=args.initialization, num_base_heads=args.num_base_heads,
        num_generators=args.num_generators,
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
    trainer_kwargs = {
        "model": model, "args": training_args,
        "train_dataset": tokenized[args.train_split],
        "eval_dataset": tokenized[args.validation_split],
        "data_collator": Collator(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability),
    }
    token_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[token_key] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    manifest = {
        "model_name_or_path": args.model_name_or_path, "initialization": args.initialization,
        "attention_type": args.attention_type, "num_base_heads": args.num_base_heads,
        "num_generators": args.num_generators, "enforce_paper_gt_mha": args.enforce_paper_gt_mha,
        "teacher_student_distillation": False, "replacement_audit": audit,
        "parameter_counts": parameter_counts,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "bert_gt_mha_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    if audit:
        import torch
        torch.save(model.state_dict(), args.output_dir / "gt_mha_state_dict.pt")
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
