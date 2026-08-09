from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.synthetic import CharTokenizer, make_lm_batch
from lgma.transformer import TinyTransformerLM


DEFAULT_STOP_SEQUENCE = "<|endoftext|>"


def read_texts(data_path: Path, val_data_path: Path | None) -> tuple[str, str | None]:
    train_text = data_path.read_text(encoding="utf-8")
    val_text = val_data_path.read_text(encoding="utf-8") if val_data_path is not None else None
    return train_text, val_text


def build_tokenizer(train_text: str, val_text: str | None) -> CharTokenizer:
    if val_text is None:
        return CharTokenizer(train_text)
    return CharTokenizer(train_text + val_text)


def paths_from_checkpoint(
    checkpoint: dict[str, object],
    data_path: Path | None,
    val_data_path: Path | None,
) -> tuple[Path, Path | None]:
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    resolved_data_path = data_path or saved_args.get("data_path")
    if resolved_data_path is None:
        raise SystemExit("--data_path is required because the checkpoint does not contain one")
    resolved_val_path = val_data_path
    if resolved_val_path is None and saved_args.get("val_data_path") is not None:
        resolved_val_path = Path(str(saved_args["val_data_path"]))
    return Path(str(resolved_data_path)), resolved_val_path


def load_tinystories_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    data_path: Path | None = None,
    val_data_path: Path | None = None,
) -> tuple[TinyTransformerLM, CharTokenizer, torch.Tensor, torch.Tensor, dict[str, object], int]:
    # These are full checkpoints produced by train_tinystories.py, including
    # configuration metadata that is not accepted by weights-only loading.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_path, val_path = paths_from_checkpoint(checkpoint, data_path, val_data_path)
    train_text, val_text = read_texts(train_path, val_path)
    tokenizer = build_tokenizer(train_text, val_text)
    train_encoded = tokenizer.encode(train_text)
    val_encoded = tokenizer.encode(val_text) if val_text is not None else train_encoded

    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise SystemExit("checkpoint is missing model_config")
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config).to(device)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise SystemExit("checkpoint is missing model_state")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, train_encoded, val_encoded, config, int(checkpoint.get("step", 0))


@torch.no_grad()
def evaluate_loss(
    model: TinyTransformerLM,
    encoded: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_batches: int,
) -> tuple[float, float]:
    losses = []
    for _ in range(eval_batches):
        batch = make_lm_batch(encoded, batch_size, seq_len, device=device)
        _, loss = model(batch.input_ids, batch.targets)
        losses.append(loss.detach())
    loss = float(torch.stack(losses).mean().cpu())
    return loss, math.exp(min(loss, 20.0))


def encode_prompt(tokenizer: CharTokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    missing = sorted({char for char in prompt if char not in tokenizer.stoi})
    if missing:
        printable = ", ".join(repr(char) for char in missing[:10])
        raise SystemExit(f"prompt contains characters outside the training vocabulary: {printable}")
    return tokenizer.encode(prompt).unsqueeze(0).to(device)


@torch.no_grad()
def generate_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
    stop_sequence: str | None = DEFAULT_STOP_SEQUENCE,
) -> str:
    if not prompt:
        raise SystemExit("prompt must not be empty")
    if temperature <= 0:
        raise SystemExit("--temperature must be positive")
    ids = encode_prompt(tokenizer, prompt, device)
    prompt_len = ids.size(1)
    stop_ids = None
    if stop_sequence:
        missing = [char for char in stop_sequence if char not in tokenizer.stoi]
        if not missing:
            stop_ids = tokenizer.encode(stop_sequence).to(device)
    for _ in range(max_new_tokens):
        context = ids[:, -model.context_length :]
        logits = model(context)[:, -1, :] / temperature
        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if stop_ids is not None and ids.size(1) - prompt_len >= stop_ids.numel():
            if torch.equal(ids[0, -stop_ids.numel() :], stop_ids):
                break
    generated = tokenizer.decode(ids[0, prompt_len:].cpu())
    if stop_sequence:
        generated = generated.split(stop_sequence, 1)[0]
    return generated
