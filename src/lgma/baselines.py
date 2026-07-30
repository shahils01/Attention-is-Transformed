from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from lgma.attention import _negative_large


def _apply_masks(
    scores: torch.Tensor,
    causal: bool,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    neg_large = _negative_large(scores.dtype)
    batch, _, target_len, source_len = scores.shape
    if causal:
        mask = torch.ones(target_len, source_len, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask[None, None, :, :], neg_large)
    if attn_mask is not None:
        attn_mask = attn_mask.to(scores.device)
        if attn_mask.dtype == torch.bool:
            if attn_mask.ndim == 2:
                scores = scores.masked_fill(attn_mask[None, None, :, :], neg_large)
            elif attn_mask.ndim == 3:
                scores = scores.masked_fill(attn_mask[:, None, :, :], neg_large)
            elif attn_mask.ndim == 4:
                scores = scores.masked_fill(attn_mask, neg_large)
            else:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
        else:
            if attn_mask.ndim == 2:
                scores = scores + attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                scores = scores + attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                scores = scores + attn_mask
            else:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
    if key_padding_mask is not None:
        if key_padding_mask.shape != (batch, source_len):
            raise ValueError("key_padding_mask must have shape (batch, source_len)")
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :].to(scores.device, dtype=torch.bool), neg_large
        )
    return scores


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
) -> torch.Tensor | None:
    """Use memory-efficient SDPA when available, otherwise request the explicit path."""
    if not hasattr(F, "scaled_dot_product_attention"):
        return None
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
    )


def _grouped_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
) -> torch.Tensor | None:
    """Use native GQA SDPA when supported, otherwise expand KV only for SDPA."""
    if not hasattr(F, "scaled_dot_product_attention"):
        return None

    if q.shape[1] == k.shape[1]:
        return _sdpa_attention(q, k, v, dropout_p=dropout_p, causal=causal)

    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            enable_gqa=True,
        )
    except TypeError:
        # PyTorch releases before native GQA support still avoid materializing
        # the quadratic attention-score tensor through this fallback.
        repeat = q.shape[1] // k.shape[1]
        return _sdpa_attention(
            q,
            k.repeat_interleave(repeat, dim=1),
            v.repeat_interleave(repeat, dim=1),
            dropout_p=dropout_p,
            causal=causal,
        )


class StandardMultiheadAttention(nn.Module):
    """Small explicit MHA baseline with independent Q/K/V head projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("d_model, num_heads, and head_dim must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.dropout = dropout
        self.causal = causal

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out_heads = None
        if not need_weights and attn_mask is None and key_padding_mask is None:
            out_heads = _sdpa_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
            )
        if out_heads is None:
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.head_dim)
            scores = _apply_masks(scores, self.causal, attn_mask, key_padding_mask)
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out_heads = torch.einsum("bhts,bhsd->bhtd", attn, v)
        out = out_heads.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        out = self.out_proj(out)
        if need_weights:
            return out, attn
        return out


class SharedIdentityAttention(nn.Module):
    """Shared Q/K/V baseline equivalent to LGMA with all metrics fixed to identity."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal
        self.q_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.einsum("btd,bsd->bts", q, k) / math.sqrt(self.head_dim)
        scores = scores[:, None, :, :].expand(batch, self.num_heads, seq_len, seq_len)
        scores = _apply_masks(scores, self.causal, attn_mask, key_padding_mask)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out_heads = torch.einsum("bhts,bsd->bhtd", attn, v)
        out = out_heads.transpose(1, 2).contiguous().view(
            batch, seq_len, self.num_heads * self.head_dim
        )
        out = self.out_proj(out)
        if need_weights:
            return out, attn
        return out


class GroupedQueryAttention(nn.Module):
    """Minimal grouped-query attention baseline, with MQA as num_kv_heads=1."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.inner_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.dropout = dropout
        self.causal = causal

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        out_heads = None
        if not need_weights and attn_mask is None and key_padding_mask is None:
            out_heads = _grouped_sdpa_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
            )
        if out_heads is None:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.head_dim)
            scores = _apply_masks(scores, self.causal, attn_mask, key_padding_mask)
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out_heads = torch.einsum("bhts,bhsd->bhtd", attn, v)
        out = out_heads.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        out = self.out_proj(out)
        if need_weights:
            return out, attn
        return out


class MultiQueryAttention(GroupedQueryAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=1,
            dropout=dropout,
            bias=bias,
            causal=causal,
        )
