from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.accounting import attention_accounting, count_parameters
from lgma.synthetic import CharTokenizer, make_lm_batch
from lgma.transformer import TinyTransformerLM, load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny character-level LM training script.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "tiny_lgma_full.json"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    text = Path(args.data_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)
    config = load_model_config(args.config)
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    seq_len = int(config["context_length"])

    final_loss = None
    for step in range(args.steps):
        batch = make_lm_batch(encoded, args.batch_size, seq_len, device=device)
        _, loss = model(batch.input_ids, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) == args.steps:
            print(json.dumps({"step": step + 1, "loss": final_loss}))

    first_attn = model.first_attention
    print(
        json.dumps(
            {
                "config": args.config,
                "vocab_size": tokenizer.vocab_size,
                "parameters": count_parameters(model),
                "attention_accounting": attention_accounting(
                    first_attn, sequence_length=seq_len, batch_size=args.batch_size
                ).__dict__,
                "final_loss": final_loss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
