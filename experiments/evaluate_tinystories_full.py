from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tinystories_runtime import evaluate_sequential_loss, load_tinystories_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate a TinyStories checkpoint over a complete split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "file"],
        default="val",
        help="Use checkpoint train/validation text, or --eval_data_path for a held-out file.",
    )
    parser.add_argument("--eval_data_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for a quick evaluation; default covers the split.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def encode_external_file(path: Path, tokenizer) -> torch.Tensor:
    text = path.read_text(encoding="utf-8")
    missing = sorted(set(text).difference(tokenizer.stoi))
    if missing:
        preview = ", ".join(repr(char) for char in missing[:10])
        raise SystemExit(f"evaluation file contains characters outside the checkpoint vocabulary: {preview}")
    return tokenizer.encode(text)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max_tokens must be positive")
    if args.split == "file" and args.eval_data_path is None:
        raise SystemExit("--eval_data_path is required with --split file")
    if args.split != "file" and args.eval_data_path is not None:
        raise SystemExit("--eval_data_path is only valid with --split file")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")

    device = torch.device(args.device)
    model, tokenizer, train_encoded, val_encoded, config, step = load_tinystories_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
    )
    if args.split == "train":
        encoded = train_encoded
        source = args.data_path or "checkpoint:train"
    elif args.split == "val":
        encoded = val_encoded
        source = args.val_data_path or "checkpoint:validation"
    else:
        encoded = encode_external_file(args.eval_data_path, tokenizer)
        source = args.eval_data_path

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics = evaluate_sequential_loss(
        model,
        encoded,
        args.batch_size,
        int(config["context_length"]),
        device,
        precision=args.precision,
        max_tokens=args.max_tokens,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "split": args.split,
        "source": str(source),
        "deterministic": True,
        "coverage": "full" if args.max_tokens is None else "prefix",
        "batch_size": args.batch_size,
        "context_length": int(config["context_length"]),
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "precision": args.precision,
        "elapsed_seconds": elapsed,
        "tokens_per_second": metrics["evaluated_tokens"] / max(elapsed, 1e-12),
        **metrics,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
