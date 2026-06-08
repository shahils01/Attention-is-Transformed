from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("expected at least two dimensions")
    return x.reshape(x.shape[0], -1)


def metric_cosine_similarity(metrics: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise cosine similarity for metrics shaped (heads, dim, dim)."""
    flat = _flatten_heads(metrics)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(eps)
    return flat @ flat.transpose(0, 1)


def attention_cosine_similarity(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise cosine similarity for attention maps shaped (batch, heads, T, S)."""
    if attn.ndim != 4:
        raise ValueError("attn must have shape (batch, heads, target_len, source_len)")
    heads = attn.transpose(0, 1).reshape(attn.shape[1], -1)
    heads = heads / heads.norm(dim=-1, keepdim=True).clamp_min(eps)
    return heads @ heads.transpose(0, 1)


def attention_kl_divergence(
    attn_a: torch.Tensor,
    attn_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(attn_a || attn_b), reduced over source positions only."""
    if attn_a.shape != attn_b.shape:
        raise ValueError("attention tensors must have the same shape")
    a = attn_a.clamp_min(eps)
    b = attn_b.clamp_min(eps)
    return (a * (a.log() - b.log())).sum(dim=-1)


def attention_entropy(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy over source positions for attention shaped (batch, heads, T, S)."""
    probs = attn.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def metric_singular_values(metrics: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(metrics)


def metric_condition_number(metrics: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    singular_values = metric_singular_values(metrics)
    return singular_values[..., 0] / singular_values[..., -1].clamp_min(eps)


def generator_norms(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "effective_generators"):
        generators = module.effective_generators()
    elif hasattr(module, "generators"):
        generators = module.generators
    else:
        raise ValueError("module does not expose generators")
    return generators.reshape(generators.shape[0], -1).norm(dim=-1)


def mean_off_diagonal(similarity: torch.Tensor) -> torch.Tensor:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    size = similarity.shape[0]
    mask = ~torch.eye(size, device=similarity.device, dtype=torch.bool)
    return similarity[mask].mean()


def pairwise_attention_kl(attn: torch.Tensor) -> torch.Tensor:
    if attn.ndim != 4:
        raise ValueError("attn must have shape (batch, heads, target_len, source_len)")
    heads = attn.transpose(0, 1)
    rows = []
    for i in range(heads.shape[0]):
        cols = []
        for j in range(heads.shape[0]):
            cols.append(attention_kl_divergence(heads[i], heads[j]).mean())
        rows.append(torch.stack(cols))
    return torch.stack(rows)


def normalized_entropy(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    entropy = attention_entropy(attn, eps=eps)
    max_entropy = torch.tensor(attn.shape[-1], device=attn.device, dtype=attn.dtype).log()
    return entropy / max_entropy.clamp_min(eps)
