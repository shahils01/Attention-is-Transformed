"""WMT14 EN→DE data loading and an encoder–decoder Transformer.

By default, an attention variant replaces MHA in encoder self-attention,
decoder self-attention, and decoder cross-attention. This makes translation
runs end-to-end tests of the proposed replacement rather than partial ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from lgma.transformer import AttentionType, build_attention


class IndexedTokenSequences:
    """Read uint32 token streams paired with uint64 sequence offsets."""

    def __init__(self, prefix: str | Path) -> None:
        prefix = Path(prefix)
        self.tokens = np.memmap(Path(f'{prefix}.bin'), dtype=np.uint32, mode='r')
        self.offsets = np.load(Path(f'{prefix}.idx.npy'), mmap_mode='r')
        if self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise ValueError(f'invalid sequence index: {prefix}')

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> np.ndarray:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class WMT14Dataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        root = Path(root)
        self.source = IndexedTokenSequences(root / f'{split}.en')
        self.target = IndexedTokenSequences(root / f'{split}.de')
        if len(self.source) != len(self.target):
            raise ValueError(f'{split} EN/DE files have different lengths')

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.source[index], self.target[index]

    @property
    def source_lengths(self) -> np.ndarray:
        return np.diff(self.source.offsets).astype(np.int64, copy=False)

    @property
    def target_lengths(self) -> np.ndarray:
        return np.diff(self.target.offsets).astype(np.int64, copy=False)


class TokenBucketBatcher:
    """Randomized length buckets with independent source/target padded-token caps.

    Each emitted list of indices satisfies ``len(batch) * max(src_len)`` and
    ``len(batch) * max(tgt_len)`` <= ``tokens_per_batch`` after truncation.
    This is the same useful interpretation as a translation ``max_tokens``
    setting: it bounds padded work/memory, not merely the unpadded average.
    """

    def __init__(self, dataset: WMT14Dataset, *, tokens_per_batch: int, max_source_length: int,
                 max_target_length: int, seed: int, rank: int = 0, world_size: int = 1,
                 bucket_size: int = 4096) -> None:
        if tokens_per_batch <= 0 or bucket_size <= 0:
            raise ValueError('tokens_per_batch and bucket_size must be positive')
        self.dataset, self.tokens_per_batch = dataset, tokens_per_batch
        self.source_lengths = np.minimum(dataset.source_lengths, max_source_length)
        self.target_lengths = np.minimum(dataset.target_lengths, max_target_length)
        if int(np.maximum(self.source_lengths, self.target_lengths).max()) > tokens_per_batch:
            raise ValueError('tokens_per_batch is smaller than an individual truncated sequence')
        self.rank, self.world_size, self.bucket_size = rank, world_size, bucket_size
        self.generator = torch.Generator().manual_seed(seed)
        self.batches: list[list[int]] = []
        self.position = 0
        self._new_epoch()

    def _new_epoch(self) -> None:
        # Sharding before bucket formation gives each DDP rank distinct pairs.
        order = torch.randperm(len(self.dataset), generator=self.generator).numpy()[self.rank::self.world_size]
        batches: list[list[int]] = []
        for start in range(0, len(order), self.bucket_size):
            bucket = order[start:start + self.bucket_size]
            bucket = bucket[np.argsort(np.maximum(self.source_lengths[bucket], self.target_lengths[bucket]), kind='stable')]
            current: list[int] = []
            max_source = max_target = 0
            for index in bucket.tolist():
                source = max(max_source, int(self.source_lengths[index]))
                target = max(max_target, int(self.target_lengths[index]))
                if current and (source * (len(current) + 1) > self.tokens_per_batch or target * (len(current) + 1) > self.tokens_per_batch):
                    batches.append(current)
                    current, max_source, max_target = [], 0, 0
                    source, target = int(self.source_lengths[index]), int(self.target_lengths[index])
                current.append(index); max_source, max_target = source, target
            if current:
                batches.append(current)
        permutation = torch.randperm(len(batches), generator=self.generator).tolist()
        self.batches = [batches[index] for index in permutation]
        self.position = 0

    def next_batch(self) -> list[int]:
        if self.position >= len(self.batches):
            self._new_epoch()
        batch = self.batches[self.position]
        self.position += 1
        return batch


@dataclass
class TranslationBatch:
    source_ids: torch.Tensor
    source_padding_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    target_ids: torch.Tensor


def collate_translation(
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    *, pad_id: int, bos_id: int, eos_id: int | None = None,
    max_source_length: int, max_target_length: int,
) -> TranslationBatch:
    if not samples:
        raise ValueError('empty translation batch')
    src = [torch.from_numpy(x[:max_source_length].copy()) for x, _ in samples]
    tgt = [torch.from_numpy(y[:max_target_length].copy()) for _, y in samples]
    # Encoded examples already end in EOS, but truncation used to silently
    # remove it.  That leaves the decoder with no stop target on every long
    # sentence and hurts generation disproportionately to teacher-forced loss.
    if eos_id is not None:
        for truncated, (source, _) in zip(src, samples):
            if len(source) > max_source_length:
                truncated[-1] = eos_id
        for truncated, (_, target) in zip(tgt, samples):
            if len(target) > max_target_length:
                truncated[-1] = eos_id
    max_src, max_tgt = max(map(len, src)), max(map(len, tgt))
    source_ids = torch.full((len(samples), max_src), pad_id, dtype=torch.long)
    targets = torch.full((len(samples), max_tgt), pad_id, dtype=torch.long)
    decoder_inputs = torch.full((len(samples), max_tgt), pad_id, dtype=torch.long)
    for row, (source, target) in enumerate(zip(src, tgt)):
        source_ids[row, : len(source)] = source
        targets[row, : len(target)] = target
        decoder_inputs[row, 0] = bos_id
        if len(target) > 1:
            decoder_inputs[row, 1 : len(target)] = target[:-1]
    return TranslationBatch(source_ids, source_ids.eq(pad_id), decoder_inputs, targets)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, self_attention: nn.Module, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attention, self.norm1, self.norm2 = self_attention, nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm1(x), key_padding_mask=padding)
        return x + self.ffn(self.norm2(x))


class DecoderLayer(nn.Module):
    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, target_padding: torch.Tensor, source_padding: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm1(x), key_padding_mask=target_padding)
        x = x + self.cross_attention(self.norm2(x), context=memory, key_padding_mask=source_padding)
        return x + self.ffn(self.norm3(x))


class WMT14Transformer(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int = 512, ffn_dim: int = 2048, num_encoder_layers: int = 6, num_decoder_layers: int = 6, num_heads: int = 8, head_dim: int = 64, attention_type: AttentionType = 'mha', cross_attention_type: AttentionType | None = None, dropout: float = 0.1, max_source_length: int = 256, max_target_length: int = 256, pad_id: int = 3, share_all_embeddings: bool = False, **attention_kwargs) -> None:
        super().__init__()
        if d_model != num_heads * head_dim:
            raise ValueError('d_model must equal num_heads * head_dim')
        self.pad_id, self.d_model = pad_id, d_model
        self.source_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # WMT14 assets use one joint EN–DE SentencePiece vocabulary, so full
        # embedding sharing is valid and matches the usual Transformer-base
        # parameter budget.
        self.target_embedding = self.source_embedding if share_all_embeddings else nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.source_positions = nn.Embedding(max_source_length, d_model)
        self.target_positions = nn.Embedding(max_target_length, d_model)
        self.dropout = nn.Dropout(dropout)
        cross_attention_type = attention_type if cross_attention_type is None else cross_attention_type
        def attention(kind: AttentionType, causal: bool) -> nn.Module:
            return build_attention(kind, d_model, num_heads, head_dim, dropout=dropout, causal=causal, **attention_kwargs)
        self.encoder = nn.ModuleList([EncoderLayer(attention(attention_type, False), d_model, ffn_dim, dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([
            DecoderLayer(
                attention(attention_type, True),
                attention(cross_attention_type, False),
                d_model,
                ffn_dim,
                dropout,
            )
            for _ in range(num_decoder_layers)
        ])
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        if share_all_embeddings:
            self.output_projection.weight = self.source_embedding.weight
        self._reset_embedding_parameters(share_all_embeddings)

    def _reset_embedding_parameters(self, share_all_embeddings: bool) -> None:
        """Use Transformer-scale embeddings, especially for tied softmax.

        ``nn.Embedding`` defaults to unit-variance weights.  Tying those raw
        weights to the vocabulary projection produces initial logits with a
        standard deviation on the order of ``sqrt(d_model)``.  For a 32k
        vocabulary this can yield losses in the tens and very slow recovery.
        """
        std = self.d_model ** -0.5
        nn.init.normal_(self.source_embedding.weight, mean=0.0, std=std)
        if not share_all_embeddings:
            nn.init.normal_(self.target_embedding.weight, mean=0.0, std=std)
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)
        nn.init.normal_(self.source_positions.weight, mean=0.0, std=std)
        nn.init.normal_(self.target_positions.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.source_embedding.weight[self.pad_id].zero_()
            if not share_all_embeddings:
                self.target_embedding.weight[self.pad_id].zero_()

    @staticmethod
    def _embed(ids: torch.Tensor, token: nn.Embedding, position: nn.Embedding, dropout: nn.Module) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        return dropout(token(ids) + position(positions)[None])

    def encode(self, source_ids: torch.Tensor, source_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        padding = source_ids.eq(self.pad_id) if source_padding_mask is None else source_padding_mask
        x = self._embed(source_ids, self.source_embedding, self.source_positions, self.dropout)
        for layer in self.encoder:
            x = layer(x, padding)
        return self.encoder_norm(x), padding

    def forward(self, source_ids: torch.Tensor, decoder_input_ids: torch.Tensor, source_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory, source_padding = self.encode(source_ids, source_padding_mask)
        target_padding = decoder_input_ids.eq(self.pad_id)
        x = self._embed(decoder_input_ids, self.target_embedding, self.target_positions, self.dropout)
        for layer in self.decoder:
            x = layer(x, memory, target_padding, source_padding)
        return self.output_projection(self.decoder_norm(x))

    def loss(self, batch: TranslationBatch) -> torch.Tensor:
        logits = self(batch.source_ids, batch.decoder_input_ids, batch.source_padding_mask)
        return F.cross_entropy(logits.flatten(0, 1), batch.target_ids.flatten(), ignore_index=self.pad_id)

    @torch.no_grad()
    def greedy_decode(self, source_ids: torch.Tensor, *, bos_id: int, eos_id: int, max_length: int) -> torch.Tensor:
        """Batched greedy decoding used for smoke tests and live monitoring.

        Final WMT14 reporting will use beam search; keeping greedy decoding
        separately labelled prevents accidental comparison of the two metrics.
        """
        memory, source_padding = self.encode(source_ids)
        generated = torch.full((source_ids.shape[0], 1), bos_id, dtype=torch.long, device=source_ids.device)
        finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
        for _ in range(max_length - 1):
            target_padding = generated.eq(self.pad_id)
            x = self._embed(generated, self.target_embedding, self.target_positions, self.dropout)
            for layer in self.decoder:
                x = layer(x, memory, target_padding, source_padding)
            next_id = self.output_projection(self.decoder_norm(x[:, -1])).argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, self.pad_id), next_id)
            generated = torch.cat((generated, next_id[:, None]), dim=1)
            finished |= next_id.eq(eos_id)
            if bool(finished.all()):
                break
        return generated
