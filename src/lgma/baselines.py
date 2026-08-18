from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from lgma.attention import _negative_large


KVCache = tuple[torch.Tensor, torch.Tensor]


def _append_to_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    past_key_value: KVCache | None,
) -> KVCache:
    """Append sequence-first cache entries stored as [batch, heads, time, dim]."""
    if past_key_value is None:
        return key, value
    past_key, past_value = past_key_value
    if past_key.ndim != 4 or past_value.ndim != 4:
        raise ValueError("cached keys and values must have shape [batch, heads, time, dim]")
    if past_key.shape[:2] + past_key.shape[3:] != key.shape[:2] + key.shape[3:]:
        raise ValueError("cached key shape is incompatible with the current key projection")
    if past_value.shape[:2] + past_value.shape[3:] != value.shape[:2] + value.shape[3:]:
        raise ValueError("cached value shape is incompatible with the current value projection")
    return torch.cat((past_key, key), dim=2), torch.cat((past_value, value), dim=2)


def _format_attention_output(
    output: torch.Tensor,
    attention: torch.Tensor | None,
    present_key_value: KVCache,
    *,
    need_weights: bool,
    use_cache: bool,
):
    if need_weights and use_cache:
        return output, attention, present_key_value
    if need_weights:
        return output, attention
    if use_cache:
        return output, present_key_value
    return output


def _causal_sdpa_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor | None, bool]:
    """Return an offset causal mask for cached/chunked decoding."""
    if not causal:
        return None, False
    target_len = q.shape[-2]
    source_len = k.shape[-2]
    if target_len == source_len:
        return None, True
    if target_len > source_len:
        raise ValueError("causal attention cannot have more queries than available keys")
    blocked = torch.ones(
        target_len,
        source_len,
        device=q.device,
        dtype=torch.bool,
    ).triu(source_len - target_len + 1)
    mask = torch.zeros(target_len, source_len, device=q.device, dtype=q.dtype)
    return mask.masked_fill(blocked, _negative_large(q.dtype)), False


def _apply_masks(
    scores: torch.Tensor,
    causal: bool,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    neg_large = _negative_large(scores.dtype)
    batch, _, target_len, source_len = scores.shape
    if causal:
        mask = torch.ones(
            target_len,
            source_len,
            device=scores.device,
            dtype=torch.bool,
        ).triu(source_len - target_len + 1)
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
    attn_mask, is_causal = _causal_sdpa_mask(q, k, causal)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
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

    attn_mask, is_causal = _causal_sdpa_mask(q, k, causal)
    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
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
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        context: torch.Tensor | None = None,
    ):
        batch, seq_len, _ = x.shape
        key_value = x if context is None else context
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        source_len = key_value.shape[1]
        k_new = self.k_proj(key_value).view(batch, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(key_value).view(batch, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = _append_to_cache(k_new, v_new, past_key_value)
        present_key_value = (k, v)

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
        return _format_attention_output(
            out,
            attn if need_weights else None,
            present_key_value,
            need_weights=need_weights,
            use_cache=use_cache,
        )


class CollaborativeAttention(nn.Module):
    """Collaborative multi-head attention with shared Q/K projections.

    This is the direct mixing-vector baseline from Cordonnier et al.,
    "Multi-Head Attention: Collaborate Instead of Concatenate". Each head
    learns an unrestricted vector ``m_h`` and scores shared queries and keys as
    ``q.T @ diag(m_h) @ k``. Values remain independently projected per head.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
        base_dim: int | None = None,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("d_model, num_heads, and head_dim must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_base_heads = 1
        self.generated_heads_per_base = num_heads
        self.head_dim = head_dim
        self.base_dim = head_dim if base_dim is None else base_dim
        self.value_dim = head_dim if value_dim is None else value_dim
        if self.base_dim <= 0 or self.value_dim <= 0:
            raise ValueError("base_dim and value_dim must be positive")
        self.inner_dim = num_heads * self.value_dim
        self.dropout = dropout
        self.causal = causal

        self.q_proj = nn.Linear(d_model, self.base_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.base_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias)
        self.mixing_vector = nn.Parameter(torch.ones(num_heads, self.base_dim))
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.ones_(self.mixing_vector)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def compute_metrics(self) -> torch.Tensor:
        """Return the exact per-head diagonal mixing matrices."""
        return torch.diag_embed(self.mixing_vector)

    def compute_head_generators(self) -> torch.Tensor:
        """Return the residual diagonal used by shared metric diagnostics."""
        return torch.diag_embed(self.mixing_vector - 1.0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        context: torch.Tensor | None = None,
    ):
        batch, seq_len, _ = x.shape
        key_value = x if context is None else context
        q_shared = self.q_proj(x)
        k_new = self.k_proj(key_value)[:, None, :, :]
        v_new = (
            self.v_proj(key_value)
            .view(batch, key_value.shape[1], self.num_heads, self.value_dim)
            .transpose(1, 2)
        )
        k_cached, v = _append_to_cache(k_new, v_new, past_key_value)
        present_key_value = (k_cached, v)

        q = q_shared[:, None, :, :] * self.mixing_vector[None, :, None, :]
        k = k_cached.expand(-1, self.num_heads, -1, -1)

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
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.base_dim)
            scores = _apply_masks(scores, self.causal, attn_mask, key_padding_mask)
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out_heads = torch.einsum("bhts,bhsv->bhtv", attn, v)

        out = out_heads.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        out = self.out_proj(out)
        return _format_attention_output(
            out,
            attn if need_weights else None,
            present_key_value,
            need_weights=need_weights,
            use_cache=use_cache,
        )


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
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        context: torch.Tensor | None = None,
    ):
        batch, seq_len, _ = x.shape
        key_value = x if context is None else context
        q = self.q_proj(x)
        k_new = self.k_proj(key_value)[:, None, :, :]
        v_new = self.v_proj(key_value)[:, None, :, :]
        k_cached, v_cached = _append_to_cache(k_new, v_new, past_key_value)
        present_key_value = (k_cached, v_cached)
        k = k_cached[:, 0]
        v = v_cached[:, 0]
        source_len = k.shape[1]
        scores = torch.einsum("btd,bsd->bts", q, k) / math.sqrt(self.head_dim)
        scores = scores[:, None, :, :].expand(batch, self.num_heads, seq_len, source_len)
        scores = _apply_masks(scores, self.causal, attn_mask, key_padding_mask)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out_heads = torch.einsum("bhts,bsd->bhtd", attn, v)
        out = out_heads.transpose(1, 2).contiguous().view(
            batch, seq_len, self.num_heads * self.head_dim
        )
        out = self.out_proj(out)
        return _format_attention_output(
            out,
            attn if need_weights else None,
            present_key_value,
            need_weights=need_weights,
            use_cache=use_cache,
        )


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
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        context: torch.Tensor | None = None,
    ):
        batch, seq_len, _ = x.shape
        key_value = x if context is None else context
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        source_len = key_value.shape[1]
        k_new = self.k_proj(key_value).view(batch, source_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(key_value).view(batch, source_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k, v = _append_to_cache(k_new, v_new, past_key_value)
        present_key_value = (k, v)
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
        return _format_attention_output(
            out,
            attn if need_weights else None,
            present_key_value,
            need_weights=need_weights,
            use_cache=use_cache,
        )


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
