from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinystories_runtime import generate_text, load_tinystories_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively sample from a TinyStories checkpoint.")
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_once(args: argparse.Namespace, model, tokenizer, device: torch.device, prompt: str) -> str:
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max_new_tokens must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, tokenizer, _, _, _, step = load_tinystories_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
    )

    if args.prompt is not None:
        print(sample_once(args, model, tokenizer, device, args.prompt))
        return

    print(f"Loaded checkpoint step {step}. Enter a prompt, or /quit to exit.")
    while True:
        prompt = input("\n> ")
        if prompt.strip() in {"/q", "/quit", "/exit"}:
            break
        if not prompt:
            continue
        print(sample_once(args, model, tokenizer, device, prompt))


if __name__ == "__main__":
    main()
