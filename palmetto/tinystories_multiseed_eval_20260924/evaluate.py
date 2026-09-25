"""Evaluate one TinyStories checkpoint with the paper's story-isolated protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from lgma.transformer import TinyTransformerLM


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["MHA", "GT-MHA"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    train_path = args.data_dir / "TinyStoriesV2-GPT4-train.txt"
    val_path = args.data_dir / "TinyStoriesV2-GPT4-valid.txt"
    chars: set[str] = set()
    with train_path.open(encoding="utf-8") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            chars.update(chunk)
    validation = val_path.read_text(encoding="utf-8")
    chars.update(validation)
    vocabulary = sorted(chars)
    stoi = {char: index for index, char in enumerate(vocabulary)}
    stories = validation.split("<|endoftext|>")
    story_ids = [index for index, story in enumerate(stories) if len(story) > 1]

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["model_config"]
    assert config["context_length"] == 512
    if args.method == "GT-MHA":
        assert config["num_base_heads"] == 4
        assert config["num_generators"] == 8
    model = TinyTransformerLM(vocab_size=len(vocabulary), **config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    step = int(checkpoint["step"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del checkpoint
    model = model.to("cuda").eval()

    rows: list[dict[str, int | float]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for position, story_id in enumerate(story_ids, start=1):
            story = stories[story_id]
            encoded = torch.tensor([stoi[char] for char in story], dtype=torch.long, device="cuda")
            story_loss = 0.0
            count = 0
            for start in range(0, len(story) - 1, 512):
                length = min(512, len(story) - 1 - start)
                precision_context = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
                with precision_context:
                    logits = model(encoded[start : start + length].unsqueeze(0))
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, len(vocabulary)),
                        encoded[start + 1 : start + length + 1],
                        reduction="sum",
                    )
                story_loss += float(loss.item())
                count += length
            assert math.isfinite(story_loss) and count == len(story) - 1
            rows.append({"story_id": story_id, "nll_sum": story_loss, "tokens": count})
            if position % 1000 == 0:
                print(json.dumps({"method": args.method, "seed": args.seed, "stories_done": position}), flush=True)
                (args.output_dir / "partial.json").write_text(json.dumps(rows), encoding="utf-8")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    total_loss = sum(row["nll_sum"] for row in rows)
    total_tokens = sum(row["tokens"] for row in rows)
    nll = total_loss / total_tokens
    summary = {
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "validation_path": str(val_path),
        "validation_sha256": hashlib.sha256(validation.encode("utf-8")).hexdigest(),
        "protocol": f"story-isolated, nonoverlapping 512-character contexts; separators excluded; {args.precision.upper()}",
        "stories": len(rows),
        "evaluated_tokens": total_tokens,
        "parameters": parameters,
        "nll": nll,
        "perplexity": math.exp(nll),
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "per_story.json").write_text(json.dumps(rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
