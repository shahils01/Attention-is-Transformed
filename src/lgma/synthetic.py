from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SyntheticTask = Literal[
    "copy",
    "reverse",
    "modular",
    "previous",
    "cumsum_mod",
    "multi_relation",
]


@dataclass(frozen=True)
class SyntheticBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    relation_ids: torch.Tensor | None = None


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


def make_multi_relation_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
    relation_ids: torch.Tensor | None = None,
) -> SyntheticBatch:
    """Controlled task where a relation token selects a sequence transformation.

    Position 0 is a control token in {0, 1, 2, 3}. Positions 1..T contain data
    tokens. The target applies the selected relation to the data positions:

    0: copy, 1: reverse, 2: previous-token lookup, 3: next-token lookup.
    """
    num_relations = 4
    if seq_len < 3:
        raise ValueError("multi_relation requires seq_len >= 3")
    if vocab_size <= num_relations:
        raise ValueError("multi_relation requires vocab_size > 4")

    if relation_ids is None:
        relation_ids = torch.randint(0, num_relations, (batch_size,), device=device)
    else:
        relation_ids = relation_ids.to(device=device, dtype=torch.long)
        if relation_ids.shape != (batch_size,):
            raise ValueError("relation_ids must have shape (batch_size,)")
        if (relation_ids < 0).any() or (relation_ids >= num_relations).any():
            raise ValueError("relation_ids must be in [0, 3]")

    data = torch.randint(num_relations, vocab_size, (batch_size, seq_len - 1), device=device)
    copy_target = data
    reverse_target = torch.flip(data, dims=[1])
    previous_target = torch.cat([data[:, :1], data[:, :-1]], dim=1)
    next_target = torch.cat([data[:, 1:], data[:, -1:]], dim=1)
    candidates = torch.stack(
        [copy_target, reverse_target, previous_target, next_target],
        dim=1,
    )
    batch_indices = torch.arange(batch_size, device=device)
    target_body = candidates[batch_indices, relation_ids]

    controls = relation_ids[:, None]
    return SyntheticBatch(
        input_ids=torch.cat([controls, data], dim=1),
        targets=torch.cat([controls, target_body], dim=1),
        relation_ids=relation_ids,
    )


def make_synthetic_batch(
    task: SyntheticTask,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
    relation_ids: torch.Tensor | None = None,
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
    if task == "multi_relation":
        return make_multi_relation_batch(
            batch_size,
            seq_len,
            vocab_size,
            device=device,
            relation_ids=relation_ids,
        )
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
    *,
    generator: torch.Generator | None = None,
) -> SyntheticBatch:
    if encoded.numel() <= seq_len + 1:
        raise ValueError("encoded text is too short for the requested sequence length")
    starts = torch.randint(
        0,
        encoded.numel() - seq_len - 1,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([encoded[start : start + seq_len] for start in starts]).to(device)
    targets = torch.stack([encoded[start + 1 : start + seq_len + 1] for start in starts]).to(device)
    return SyntheticBatch(input_ids=inputs, targets=targets)
