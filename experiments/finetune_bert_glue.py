from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lgma.bert import BERT_ATTENTION_TYPES, load_bert_sequence_classifier

GLUE_COLUMNS = {
    "cola": ("sentence", None), "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"), "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None), "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Fine-tune BERT attention baselines or converted GT-MHA on GLUE"
    )
    p.add_argument("--model-name-or-path", default="google-bert/bert-base-uncased")
    p.add_argument("--task", choices=sorted(GLUE_COLUMNS), required=True)
    p.add_argument("--attention-type", choices=sorted(BERT_ATTENTION_TYPES), default="mha")
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-base-heads", type=int, default=4)
    p.add_argument("--num-generators", type=int, default=8)
    p.add_argument("--gt-qk-base-dim", type=int)
    p.add_argument("--gt-value-head-dim", type=int)
    p.add_argument(
        "--generator-mixing", "--generator_mixing",
        dest="generator_mixing", choices=["softmax", "none"], default="softmax",
    )
    p.add_argument("--use-sdpa", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fuse-base-qkv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sdpa-gqa-mode", choices=["auto", "native", "expand"], default="auto")
    p.add_argument("--enforce-paper-gt-mha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-sequence-length", type=int, default=128)
    p.add_argument("--per-device-train-batch-size", type=int, default=32)
    p.add_argument("--per-device-eval-batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--run-name")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def dependencies() -> tuple[Any, ...]:
    try:
        import evaluate
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install with: pip install -e '.[bert,tracking]'") from exc
    return evaluate, load_dataset, AutoTokenizer, DataCollatorWithPadding, set_seed, Trainer, TrainingArguments


def main() -> None:
    args = parse_args()
    effective_num_kv_heads = 1 if args.attention_type == "mqa" else args.num_kv_heads
    if args.bf16 and args.fp16:
        raise SystemExit("--bf16 and --fp16 are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluate, load_dataset, AutoTokenizer, Collator, set_seed, Trainer, TrainingArguments = dependencies()
    set_seed(args.seed)
    raw = load_dataset("glue", args.task)
    regression = args.task == "stsb"
    num_labels = 1 if regression else raw["train"].features["label"].num_classes
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model, audit = load_bert_sequence_classifier(
        args.model_name_or_path, num_labels=num_labels,
        attention_type=args.attention_type, num_kv_heads=effective_num_kv_heads,
        num_base_heads=args.num_base_heads,
        num_generators=args.num_generators, qk_base_dim=args.gt_qk_base_dim,
        value_head_dim=args.gt_value_head_dim, generator_mixing=args.generator_mixing,
        use_sdpa=args.use_sdpa, fuse_base_qkv=args.fuse_base_qkv,
        sdpa_gqa_mode=args.sdpa_gqa_mode,
        enforce_paper_gt_mha=args.enforce_paper_gt_mha,
        trust_remote_code=args.trust_remote_code,
    )
    first, second = GLUE_COLUMNS[args.task]

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[first], batch[second] if second else None, truncation=True,
            max_length=args.max_sequence_length,
        )

    tokenized = raw.map(tokenize, batched=True, desc=f"Tokenizing GLUE/{args.task}")
    metric = evaluate.load("glue", args.task)

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, labels = prediction
        predictions = np.squeeze(logits) if regression else np.argmax(logits, axis=-1)
        values = metric.compute(predictions=predictions, references=labels)
        if len(values) > 1:
            values["combined_score"] = float(np.mean(list(values.values())))
        return values

    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "num_train_epochs": args.num_train_epochs,
        "seed": args.seed, "data_seed": args.seed,
        "dataloader_num_workers": args.num_workers, "bf16": args.bf16, "fp16": args.fp16,
        "logging_strategy": "steps", "logging_steps": 50, "save_strategy": "epoch",
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "run_name": args.run_name, "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs[key] = "epoch"
    training_args = TrainingArguments(**kwargs)
    validation = "validation_matched" if args.task == "mnli" else "validation"
    trainer_kwargs = {
        "model": model, "args": training_args, "train_dataset": tokenized["train"],
        "eval_dataset": tokenized[validation], "data_collator": Collator(tokenizer),
        "compute_metrics": compute_metrics,
    }
    token_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[token_key] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    manifest = {
        "model_name_or_path": args.model_name_or_path, "task": args.task,
        "attention_type": args.attention_type, "num_kv_heads": effective_num_kv_heads,
        "num_base_heads": args.num_base_heads,
        "num_generators": args.num_generators,
        "qk_base_dim": args.gt_qk_base_dim,
        "value_head_dim": args.gt_value_head_dim,
        "generator_mixing": args.generator_mixing,
        "teacher_student_distillation": False,
        "replacement_audit": audit,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "bert_glue_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", evaluation)


if __name__ == "__main__":
    main()
