from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SyntheticTask = Literal["copy", "reverse", "modular", "previous", "cumsum_mod"]


@dataclass(frozen=True)
class SyntheticBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor


def make_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    return SyntheticBatch(input_ids=tokens, targets=tokens.clone())


def make_reverse_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    return SyntheticBatch(input_ids=tokens, targets=torch.flip(tokens, dims=[1]))


def make_modular_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    shifted = torch.roll(tokens, shifts=-1, dims=1)
    targets = (tokens + shifted) % vocab_size
    return SyntheticBatch(input_ids=tokens, targets=targets)


def make_previous_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.zeros_like(tokens)
    targets[:, 1:] = tokens[:, :-1]
    return SyntheticBatch(input_ids=tokens, targets=targets)


def make_cumsum_mod_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.cumsum(tokens, dim=1) % vocab_size
    return SyntheticBatch(input_ids=tokens, targets=targets)


def make_synthetic_batch(
    task: SyntheticTask,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    if task == "copy":
        return make_copy_batch(batch_size, seq_len, vocab_size, device=device)
    if task == "reverse":
        return make_reverse_batch(batch_size, seq_len, vocab_size, device=device)
    if task == "modular":
        return make_modular_batch(batch_size, seq_len, vocab_size, device=device)
    if task == "previous":
        return make_previous_batch(batch_size, seq_len, vocab_size, device=device)
    if task == "cumsum_mod":
        return make_cumsum_mod_batch(batch_size, seq_len, vocab_size, device=device)
    raise ValueError(f"unsupported synthetic task: {task}")


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        if not chars:
            raise ValueError("text must contain at least one character")
        self.itos = chars
        self.stoi = {char: idx for idx, char in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[char] for char in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)


def make_lm_batch(
    encoded: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    if encoded.numel() <= seq_len + 1:
        raise ValueError("encoded text is too short for the requested sequence length")
    starts = torch.randint(0, encoded.numel() - seq_len - 1, (batch_size,))
    inputs = torch.stack([encoded[start : start + seq_len] for start in starts]).to(device)
    targets = torch.stack([encoded[start + 1 : start + seq_len + 1] for start in starts]).to(device)
    return SyntheticBatch(input_ids=inputs, targets=targets)
