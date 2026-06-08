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


def metric_delta_cosine_similarity(metrics: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise cosine similarity after removing the shared identity metric."""
    if metrics.ndim != 3 or metrics.shape[-1] != metrics.shape[-2]:
        raise ValueError("metrics must have shape (heads, dim, dim)")
    eye = torch.eye(metrics.shape[-1], device=metrics.device, dtype=metrics.dtype)
    return metric_cosine_similarity(metrics - eye[None, :, :], eps=eps)


def attention_cosine_similarity(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise cosine similarity for attention maps shaped (batch, heads, T, S)."""
    if attn.ndim != 4:
        raise ValueError("attn must have shape (batch, heads, target_len, source_len)")
    heads = attn.transpose(0, 1).reshape(attn.shape[1], -1)
    heads = heads / heads.norm(dim=-1, keepdim=True).clamp_min(eps)
    return heads @ heads.transpose(0, 1)


def centered_attention_cosine_similarity(
    attn: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise attention-map cosine after removing each head's mean probability."""
    if attn.ndim != 4:
        raise ValueError("attn must have shape (batch, heads, target_len, source_len)")
    centered = attn - attn.mean(dim=(-1, -2), keepdim=True)
    return attention_cosine_similarity(centered, eps=eps)


def score_cosine_similarity(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise cosine similarity for score/logit maps shaped (batch, heads, T, S)."""
    if scores.ndim != 4:
        raise ValueError("scores must have shape (batch, heads, target_len, source_len)")
    heads = scores.transpose(0, 1).reshape(scores.shape[1], -1)
    heads = heads - heads.mean(dim=-1, keepdim=True)
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


def metric_distance_from_identity(metrics: torch.Tensor) -> torch.Tensor:
    if metrics.ndim != 3 or metrics.shape[-1] != metrics.shape[-2]:
        raise ValueError("metrics must have shape (heads, dim, dim)")
    eye = torch.eye(metrics.shape[-1], device=metrics.device, dtype=metrics.dtype)
    return (metrics - eye[None, :, :]).reshape(metrics.shape[0], -1).norm(dim=-1)


def induced_bilinear_forms(module: nn.Module, metrics: torch.Tensor | None = None) -> torch.Tensor:
    """Compute B_h = W_Q^T M_h W_K for an LGMA-style module."""
    if not hasattr(module, "q_proj") or not hasattr(module, "k_proj"):
        raise ValueError("module must expose q_proj and k_proj")
    if metrics is None:
        if not hasattr(module, "compute_metrics"):
            raise ValueError("module does not expose compute_metrics")
        metrics = module.compute_metrics()
    wq = module.q_proj.weight
    wk = module.k_proj.weight
    return torch.einsum("id,hde,ej->hij", wq.transpose(0, 1), metrics, wk)


def induced_metric_cosine_similarity(
    module: nn.Module,
    metrics: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return metric_cosine_similarity(induced_bilinear_forms(module, metrics), eps=eps)


def generator_norms(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "effective_generators"):
        generators = module.effective_generators()
    elif hasattr(module, "generators"):
        generators = module.generators
    else:
        raise ValueError("module does not expose generators")
    return generators.reshape(generators.shape[0], -1).norm(dim=-1)


def grouped_gradient_norms(module: nn.Module) -> dict[str, float]:
    groups = {
        "theta": ("theta",),
        "generators": ("generators", "raw_metrics"),
        "q_proj": ("q_proj",),
        "k_proj": ("k_proj",),
        "v_proj": ("v_proj",),
    }
    output: dict[str, float] = {}
    for group, names in groups.items():
        squared = 0.0
        found = False
        for name, param in module.named_parameters():
            if param.grad is None or not any(part in name for part in names):
                continue
            found = True
            squared += float(param.grad.detach().float().pow(2).sum().cpu())
        output[f"grad_norm_{group}"] = squared**0.5 if found else 0.0
    return output


def mean_off_diagonal(similarity: torch.Tensor) -> torch.Tensor:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    size = similarity.shape[0]
    mask = ~torch.eye(size, device=similarity.device, dtype=torch.bool)
    return similarity[mask].mean()


def off_diagonal_squared_mean(similarity: torch.Tensor) -> torch.Tensor:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    size = similarity.shape[0]
    mask = ~torch.eye(size, device=similarity.device, dtype=torch.bool)
    return similarity[mask].pow(2).mean()


def metric_diversity_loss(
    metrics: torch.Tensor,
    squared: bool = False,
    use_delta: bool = True,
) -> torch.Tensor:
    similarity = (
        metric_delta_cosine_similarity(metrics)
        if use_delta
        else metric_cosine_similarity(metrics)
    )
    if squared:
        return off_diagonal_squared_mean(similarity)
    return mean_off_diagonal(similarity)


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
