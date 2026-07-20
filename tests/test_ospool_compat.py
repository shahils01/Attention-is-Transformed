from __future__ import annotations

import torch

from lgma.synthetic import CharTokenizer
from ospool.train_tinystories_compat import (
    compact_build_tokenizer,
    compact_encode,
    compact_make_lm_batch,
)


def test_compact_encoding_matches_existing_character_ids() -> None:
    text = "tiny stories\nare useful"
    tokenizer = CharTokenizer(text)
    expected = torch.tensor([tokenizer.stoi[character] for character in text])

    encoded = compact_encode(tokenizer, text)

    assert encoded.dtype == torch.uint8
    assert torch.equal(encoded.long(), expected)


def test_compact_tokenizer_uses_train_and_validation_vocabularies() -> None:
    tokenizer = compact_build_tokenizer("abc", "cd!")

    assert set(tokenizer.itos) == {"a", "b", "c", "d", "!"}


def test_compact_batches_are_long_and_shift_targets() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    tokenizer = CharTokenizer(text)
    encoded = compact_encode(tokenizer, text)

    batch = compact_make_lm_batch(encoded, batch_size=3, seq_len=8)

    assert batch.input_ids.dtype == torch.long
    assert batch.targets.dtype == torch.long
    assert batch.input_ids.shape == (3, 8)
    assert torch.equal(batch.input_ids[:, 1:], batch.targets[:, :-1])
