"""Opt-in, in-memory QKV packing for GQA inference."""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
import torch.nn.functional as F

from lgma.baselines import (
    GroupedQueryAttention,
    KVCache,
    _append_to_cache,
    _apply_masks,
    _format_attention_output,
    _grouped_sdpa_attention,
)
from lgma.kv_cache import StaticKVCache
from lgma.transformer import TinyTransformerLM


class PackedGroupedQueryAttention(nn.Module):
    """An eval-only GQA layer with one QKV projection for self-attention."""

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, source: GroupedQueryAttention) -> None:
        super().__init__()
        if type(source) is not GroupedQueryAttention:
            raise TypeError("QKV packing requires an ordinary GroupedQueryAttention layer")
        if source.training or source.attn_dropout.training:
            raise ValueError("QKV packing requires a GQA layer in eval mode")

        self.num_heads = source.num_heads
        self.num_kv_heads = source.num_kv_heads
        self.head_dim = source.head_dim
        self.inner_dim = source.inner_dim
        self.kv_dim = source.kv_dim
        self.causal = source.causal
        projections = (source.q_proj, source.k_proj, source.v_proj)
        biases = tuple(proj.bias for proj in projections)
        has_bias = source.q_proj.bias is not None
        if any((bias is not None) != has_bias for bias in biases):
            raise ValueError("QKV packing requires projections with matching bias settings")
        self.qkv_proj = nn.Linear(
            source.d_model, self.inner_dim + 2 * self.kv_dim, bias=has_bias, device="meta"
        )
        self.qkv_proj.weight = nn.Parameter(torch.cat([proj.weight for proj in projections]))
        if has_bias:
            self.qkv_proj.bias = nn.Parameter(torch.cat([bias for bias in biases if bias is not None]))
        self.out_proj = deepcopy(source.out_proj)
        self.requires_grad_(False)
        self.eval()

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
        if self.training:
            raise RuntimeError("Packed GQA is inference-only and requires eval mode")

        batch, seq_len, _ = x.shape
        if context is None:
            q_flat, k_flat, v_flat = self.qkv_proj(x).split(
                (self.inner_dim, self.kv_dim, self.kv_dim), dim=-1
            )
        else:
            # Cross-attention projects queries and context with contiguous row slices.
            weight, bias = self.qkv_proj.weight, self.qkv_proj.bias
            q_flat = F.linear(
                x, weight[: self.inner_dim], None if bias is None else bias[: self.inner_dim]
            )
            k_flat, v_flat = F.linear(
                context, weight[self.inner_dim :], None if bias is None else bias[self.inner_dim :]
            ).split(self.kv_dim, dim=-1)

        source_len = k_flat.shape[1]
        q = q_flat.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_flat.view(batch, source_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(batch, source_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k, v = _append_to_cache(k, v, past_key_value)
        if isinstance(past_key_value, StaticKVCache):
            present_key_value = past_key_value
        elif use_cache and past_key_value is None:
            # The first tuple cache must not retain the packed query activations.
            present_key_value = k.clone(), v.clone()
        else:
            present_key_value = k, v

        attention = None
        out_heads = None
        if not need_weights and attn_mask is None and key_padding_mask is None:
            # The single query is the final position and can attend to all available keys.
            causal = self.causal and not (seq_len == 1 and k.shape[-2] > 0)
            out_heads = _grouped_sdpa_attention(q, k, v, dropout_p=0.0, causal=causal)
        if out_heads is None:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.head_dim)
            scores = _apply_masks(scores, self.causal, attn_mask, key_padding_mask)
            attention = torch.softmax(scores, dim=-1)
            out_heads = torch.einsum("bhts,bhsd->bhtd", attention, v)

        out = out_heads.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        return _format_attention_output(
            self.out_proj(out),
            attention if need_weights else None,
            present_key_value,
            need_weights=need_weights,
            use_cache=use_cache,
        )


@torch.inference_mode(False)
@torch.no_grad()
def pack_gqa_model(model: TinyTransformerLM) -> TinyTransformerLM:
    """Copy an eval GQA model with packed projections and frozen inference weights.

    Convert after loading a normal GQA checkpoint. The packed state dictionary
    uses different projection names from the original model.
    """
    if not isinstance(model, TinyTransformerLM):
        raise TypeError("QKV packing requires a TinyTransformerLM")
    if any(module.training for module in model.modules()):
        raise ValueError("QKV packing requires a model in eval mode")

    packed_layers = {
        id(block.attn): PackedGroupedQueryAttention(block.attn) for block in model.blocks
    }
    return deepcopy(model, memo=packed_layers).requires_grad_(False)
