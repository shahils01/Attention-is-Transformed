"""Copy only the model fields required by the full-validation evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit(f"destination already exists: {args.destination}")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    payload = {
        "step": checkpoint["step"],
        "model_config": checkpoint["model_config"],
        "model_state": checkpoint["model_state"],
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(".pt.part")
    torch.save(payload, temporary)
    temporary.replace(args.destination)
    print(f"{args.destination} {args.destination.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
