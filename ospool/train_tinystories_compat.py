"""OSPool-only compatibility launcher for TinyStories training.

This module deliberately leaves the Clemson/Palmetto training entry point
unchanged.  It installs compact, process-local data-loading replacements and
then delegates argument parsing and training to experiments/train_tinystories.py.
"""

from __future__ import annotations

import json
import sys
from array import array
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compact_build_tokenizer(train_text: str, val_text: str | None):
    """Build the normal character tokenizer without concatenating large texts."""
    from lgma.synthetic import CharTokenizer

    characters = set(train_text)
    if val_text is not None:
        characters.update(val_text)
    return CharTokenizer("".join(characters))


def compact_encode(tokenizer: Any, text: str) -> torch.Tensor:
    """Encode text without constructing a Python list of one int per character."""
    vocab_size = tokenizer.vocab_size
    if vocab_size <= 2**8:
        typecode, dtype = "B", torch.uint8
    elif vocab_size <= 2**16:
        typecode, dtype = "H", torch.uint16
    elif vocab_size <= 2**32:
        typecode, dtype = "I", torch.uint32
    else:
        raise ValueError("OSPool compact encoding supports at most 2**32 characters")

    buffer = array(typecode, (tokenizer.stoi[character] for character in text))
    return torch.frombuffer(buffer, dtype=dtype)


def compact_make_lm_batch(
    encoded: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
):
    """Create int64 model inputs from a compact CPU token tensor."""
    from lgma.synthetic import SyntheticBatch

    if encoded.numel() <= seq_len + 1:
        raise ValueError("encoded text is too short for the requested sequence length")
    starts = torch.randint(0, encoded.numel() - seq_len - 1, (batch_size,))
    offsets = torch.arange(seq_len)
    positions = starts[:, None] + offsets[None, :]
    inputs = encoded[positions].to(device=device, dtype=torch.long)
    targets = encoded[positions + 1].to(device=device, dtype=torch.long)
    return SyntheticBatch(input_ids=inputs, targets=targets)


def install_ospool_patches():
    """Install loader replacements only in this OSPool launcher process."""
    from experiments import train_tinystories as training
    from lgma.synthetic import CharTokenizer

    CharTokenizer.encode = compact_encode
    training.build_tokenizer = compact_build_tokenizer
    training.make_lm_batch = compact_make_lm_batch
    return training


def main() -> None:
    training = install_ospool_patches()
    print(
        json.dumps(
            {
                "event": "ospool_compatibility_enabled",
                "compact_token_encoding": True,
                "existing_training_pipeline_modified": False,
            }
        ),
        flush=True,
    )
    training.main()


if __name__ == "__main__":
    main()
