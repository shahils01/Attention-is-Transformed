from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapter_trainer.distill_tinyllama import (
    TokenBatcher,
    dtype_from_precision,
    freeze,
    get_decoder_layers,
    import_transformers,
    parse_layers,
    replace_model_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a TinyLlama model with LGMA adapters")
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--calibration_text_file", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    AutoModelForCausalLM, AutoTokenizer = import_transformers()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_precision(args.precision),
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    ).to(device)
    layers = parse_layers(args.layers, len(get_decoder_layers(model)))
    replace_model_attention(model, args, layers, device=device)
    freeze(model)
    batcher = TokenBatcher(
        tokenizer,
        args.calibration_text_file,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    losses = []
    for _ in range(args.eval_steps):
        batch = batcher.next_batch()
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch["labels"][:, :-1].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses.append(float(loss.detach().cpu()))
    mean_loss = sum(losses) / max(len(losses), 1)
    print(f"eval_loss={mean_loss:.6f}")
    print(f"perplexity={math.exp(mean_loss):.6f}")


if __name__ == "__main__":
    main()
