from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _negative_large(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


class LieGeneratedMetricAttention(nn.Module):
    """Multi-head attention with shared Q/K/V and Lie-generated head metrics."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        num_generators: int,
        dropout: float = 0.0,
        bias: bool = False,
        generator_type: str = "full",
        use_sdpa: bool = True,
        causal: bool = False,
        stabilize_generators: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or head_dim <= 0 or num_generators <= 0:
            raise ValueError("d_model, num_heads, head_dim, and num_generators must be positive")
        if generator_type not in {"full", "diagonal", "symmetric"}:
            raise ValueError(f"unsupported generator_type: {generator_type}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_generators = num_generators
        self.dropout = dropout
        self.generator_type = generator_type
        self.use_sdpa = use_sdpa
        self.causal = causal
        self.stabilize_generators = stabilize_generators

        self.q_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

        if generator_type == "diagonal":
            self.generators = nn.Parameter(torch.empty(num_generators, head_dim))
        else:
            self.generators = nn.Parameter(torch.empty(num_generators, head_dim, head_dim))
        self.theta = nn.Parameter(torch.empty(num_heads, num_generators))

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

        generator_std = 0.02 / math.sqrt(self.head_dim)
        nn.init.normal_(self.generators, mean=0.0, std=generator_std)
        nn.init.normal_(self.theta, mean=0.0, std=0.02)

    def _dense_generators(self) -> torch.Tensor:
        if self.generator_type == "diagonal":
            return torch.diag_embed(self.generators)

        generators = self.generators
        if self.generator_type == "symmetric":
            generators = 0.5 * (generators + generators.transpose(-1, -2))
        if self.stabilize_generators:
            generators = self._make_trace_zero(generators)
        return generators

    @staticmethod
    def _make_trace_zero(generators: torch.Tensor) -> torch.Tensor:
        dim = generators.shape[-1]
        trace = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        eye = torch.eye(dim, device=generators.device, dtype=generators.dtype)
        return generators - (trace[..., None, None] / dim) * eye

    def compute_metrics(self) -> torch.Tensor:
        if self.generator_type == "diagonal":
            diagonal = torch.einsum("hm,md->hd", self.theta, self.generators)
            scale = torch.exp(diagonal)
            return torch.diag_embed(scale)

        generators = self._dense_generators()
        head_generators = torch.einsum("hm,mde->hde", self.theta, generators)
        if self.generator_type == "symmetric":
            head_generators = 0.5 * (head_generators + head_generators.transpose(-1, -2))
        return torch.stack([torch.linalg.matrix_exp(generator) for generator in head_generators], dim=0)

    def effective_generators(self) -> torch.Tensor:
        """Return dense stabilized generator basis for diagnostics."""
        return self._dense_generators()

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _apply_metric_to_queries(self, q: torch.Tensor) -> torch.Tensor:
        if self.generator_type == "diagonal":
            diagonal = torch.einsum("hm,md->hd", self.theta, self.generators)
            scale = torch.exp(diagonal)
            return q[:, None, :, :] * scale[None, :, None, :]
        metrics = self.compute_metrics()
        return torch.einsum("btd,hde->bhte", q, metrics)

    def _prepare_additive_mask(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        neg_large = _negative_large(scores.dtype)
        batch, _, target_len, source_len = scores.shape

        if self.causal:
            causal_mask = torch.ones(
                target_len, source_len, device=scores.device, dtype=torch.bool
            ).triu(1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], neg_large)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=scores.device)
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
                scores = scores + self._broadcast_attn_mask(attn_mask, batch)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, source_len):
                raise ValueError(
                    "key_padding_mask must have shape (batch, source_len), got "
                    f"{tuple(key_padding_mask.shape)}"
                )
            mask = key_padding_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask[:, None, None, :], neg_large)

        return scores

    @staticmethod
    def _broadcast_attn_mask(attn_mask: torch.Tensor, batch: int) -> torch.Tensor:
        if attn_mask.ndim == 2:
            return attn_mask[None, None, :, :]
        if attn_mask.ndim == 3:
            if attn_mask.shape[0] == batch:
                return attn_mask[:, None, :, :]
            return attn_mask[:, :, :, None].squeeze(-1)
        if attn_mask.ndim == 4:
            return attn_mask
        raise ValueError("attn_mask must have 2, 3, or 4 dimensions")

    def _explicit_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_metric = self._apply_metric_to_queries(q)
        scores = torch.einsum("bhtd,bsd->bhts", q_metric, k)
        scores = scores / math.sqrt(self.head_dim)
        scores = self._prepare_additive_mask(scores, attn_mask, key_padding_mask)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out_heads = torch.einsum("bhts,bsd->bhtd", attn, v)
        return out_heads, attn

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = q.shape
        q_metric = self._apply_metric_to_queries(q)
        k_expanded = k[:, None, :, :].expand(batch, self.num_heads, seq_len, self.head_dim)
        v_expanded = v[:, None, :, :].expand(batch, self.num_heads, seq_len, self.head_dim)

        sdpa_mask = self._prepare_sdpa_mask(attn_mask, q.dtype, q.device)
        if key_padding_mask is not None:
            neg_large = _negative_large(q.dtype)
            pad_mask = key_padding_mask[:, None, None, :].to(device=q.device, dtype=torch.bool)
            pad_additive = torch.zeros(
                batch, 1, 1, seq_len, device=q.device, dtype=q.dtype
            ).masked_fill(pad_mask, neg_large)
            if sdpa_mask is None:
                sdpa_mask = pad_additive
            else:
                sdpa_mask = sdpa_mask + pad_additive

        if not hasattr(F, "scaled_dot_product_attention"):
            out_heads, _ = self._explicit_attention(q, k, v, attn_mask, key_padding_mask)
            return out_heads

        dropout_p = self.dropout if self.training else 0.0
        return F.scaled_dot_product_attention(
            q_metric,
            k_expanded,
            v_expanded,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=self.causal and sdpa_mask is None,
        )

    @staticmethod
    def _prepare_sdpa_mask(
        attn_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attn_mask is None:
            return None
        mask = attn_mask.to(device=device)
        if mask.dtype == torch.bool:
            neg_large = _negative_large(dtype)
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            elif mask.ndim != 4:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
            return torch.zeros_like(mask, dtype=dtype).masked_fill(mask, neg_large)
        mask = mask.to(dtype=dtype)
        if mask.ndim == 2:
            return mask[None, None, :, :]
        if mask.ndim == 3:
            return mask[:, None, :, :]
        if mask.ndim == 4:
            return mask
        raise ValueError("attn_mask must have 2, 3, or 4 dimensions")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, seq_len, d_model), got {tuple(x.shape)}")

        q, k, v = self._project(x)
        if self.use_sdpa and not need_weights:
            out_heads = self._sdpa_attention(q, k, v, attn_mask, key_padding_mask)
            attn = None
        else:
            out_heads, attn = self._explicit_attention(q, k, v, attn_mask, key_padding_mask)

        batch, _, seq_len, _ = out_heads.shape
        out_heads = out_heads.transpose(1, 2).contiguous()
        out_heads = out_heads.view(batch, seq_len, self.num_heads * self.head_dim)
        out = self.out_proj(out_heads)
        if need_weights:
            if attn is None:
                raise RuntimeError("attention weights unavailable on SDPA path")
            return out, attn
        return out
