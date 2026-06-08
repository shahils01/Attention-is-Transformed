from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tinystories_runtime import evaluate_loss, load_tinystories_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TinyStories checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Training text path. Defaults to the path saved in the checkpoint.",
    )
    parser.add_argument(
        "--val_data_path",
        type=Path,
        default=None,
        help="Validation text path. Defaults to the path saved in the checkpoint.",
    )
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_batches <= 0:
        raise SystemExit("--eval_batches must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, tokenizer, train_encoded, val_encoded, config, step = load_tinystories_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
    )
    encoded = val_encoded if args.split == "val" else train_encoded
    loss, perplexity = evaluate_loss(
        model=model,
        encoded=encoded,
        batch_size=args.batch_size,
        seq_len=int(config["context_length"]),
        device=device,
        eval_batches=args.eval_batches,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "split": args.split,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "context_length": int(config["context_length"]),
        "vocab_size": tokenizer.vocab_size,
        "loss": loss,
        "perplexity": perplexity,
        "bits_per_character": loss / math.log(2.0),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
