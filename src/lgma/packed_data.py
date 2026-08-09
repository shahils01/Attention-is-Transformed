from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lgma.synthetic import SyntheticBatch


@dataclass(frozen=True)
class PackedTokenizerInfo:
    vocab_size: int
    special_token_ids: dict[str, int]


class PackedTokenSplit:
    """Read-only collection of memory-mapped, contiguous token shards."""

    def __init__(self, root: Path, split: str, expected_tokens: int) -> None:
        self.root = root
        self.split = split
        self.paths = sorted((root / "shards" / split).glob("*.bin"))
        if not self.paths:
            raise ValueError(f"no packed token shards found for split {split!r} in {root}")
        self._arrays = [np.memmap(path, mode="r", dtype=np.dtype("<u2")) for path in self.paths]
        self.shard_tokens = [int(array.size) for array in self._arrays]
        self.total_tokens = sum(self.shard_tokens)
        if self.total_tokens != expected_tokens:
            raise ValueError(
                f"{split} token count mismatch: metadata={expected_tokens}, "
                f"files={self.total_tokens}"
            )

    def numel(self) -> int:
        return self.total_tokens

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | str = "cpu",
        *,
        generator: torch.Generator | None = None,
    ) -> SyntheticBatch:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        valid_starts = [max(tokens - seq_len, 0) for tokens in self.shard_tokens]
        cumulative = []
        running = 0
        for count in valid_starts:
            running += count
            cumulative.append(running)
        if running <= 0:
            raise ValueError(f"split {self.split!r} is too short for sequence length {seq_len}")

        sampled = torch.randint(0, running, (batch_size,), generator=generator).tolist()
        inputs = np.empty((batch_size, seq_len), dtype=np.int64)
        targets = np.empty((batch_size, seq_len), dtype=np.int64)
        for row, global_start in enumerate(sampled):
            shard_index = bisect.bisect_right(cumulative, global_start)
            previous = cumulative[shard_index - 1] if shard_index > 0 else 0
            local_start = global_start - previous
            sequence = np.asarray(
                self._arrays[shard_index][local_start : local_start + seq_len + 1],
                dtype=np.int64,
            )
            inputs[row] = sequence[:-1]
            targets[row] = sequence[1:]
        return SyntheticBatch(
            input_ids=torch.from_numpy(inputs).to(device),
            targets=torch.from_numpy(targets).to(device),
        )


@dataclass(frozen=True)
class PackedTokenCorpus:
    root: Path
    tokenizer: PackedTokenizerInfo
    train: PackedTokenSplit
    validation: PackedTokenSplit
    test: PackedTokenSplit
    metadata: dict[str, object]


def load_packed_token_corpus(root: str | Path) -> PackedTokenCorpus:
    root = Path(root)
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"packed dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported packed dataset schema version")
    if metadata.get("dtype") != "uint16-le":
        raise ValueError(f"unsupported packed dtype: {metadata.get('dtype')}")
    vocab_size = int(metadata["vocab_size"])
    if vocab_size > 65_536:
        raise ValueError("uint16 packed data cannot represent this vocabulary")
    special_token_ids = {
        str(key): int(value)
        for key, value in dict(metadata["special_token_ids"]).items()
    }
    splits = dict(metadata["splits"])
    corpus = PackedTokenCorpus(
        root=root,
        tokenizer=PackedTokenizerInfo(vocab_size, special_token_ids),
        train=PackedTokenSplit(root, "train", int(splits["train"]["tokens"])),
        validation=PackedTokenSplit(
            root, "validation", int(splits["validation"]["tokens"])
        ),
        test=PackedTokenSplit(root, "test", int(splits["test"]["tokens"])),
        metadata=metadata,
    )
    return corpus
