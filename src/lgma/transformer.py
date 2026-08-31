from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import (
    CollaborativeAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    SharedIdentityAttention,
    StandardMultiheadAttention,
)

AttentionType = Literal[
    "mha",
    "reduced_mha",
    "mqa",
    "gqa",
    "collaborative",
    "shared_identity",
    "lgma",
    "lgma_v2",
    "lgma_residual",
    "lgma_quad",
    "lgma_unconstrained",
    "lgma_value_diag",
    "lgma_multibase",
    "lgma_multibase_value_diag",
    "gt_mha_exact",
    "gt_mha_residual",
    "gt_mha_quadratic",
]
LGMA_ATTENTION_TYPES = {
    "lgma",
    "lgma_v2",
    "lgma_residual",
    "lgma_quad",
    "lgma_unconstrained",
    "lgma_value_diag",
    "lgma_multibase",
    "lgma_multibase_value_diag",
    "gt_mha_exact",
    "gt_mha_residual",
    "gt_mha_quadratic",
}


def validate_paper_gt_mha_module(module: nn.Module) -> None:
    """Fail fast unless ``module`` matches a manuscript GT-MHA variant."""
    paper_modes = {"exp", "residual", "quadratic"}
    expected = {
        "num_base_heads": 4,
        "num_generators": 8,
        "generator_type": "full",
        "generator_mixing": "softmax",
        "stabilize_generators": False,
        "normalize_generators": False,
        "head_generator_symmetric_cap": None,
        "logit_scale_mode": "sqrt_dim",
        "learn_head_temperature": False,
    }
    if not isinstance(module, LieGeneratedMetricAttention):
        raise ValueError("paper GT-MHA requires LieGeneratedMetricAttention")
    mismatches = []
    if module.metric_mode not in paper_modes:
        mismatches.append(
            f"metric_mode={module.metric_mode!r} (expected one of {sorted(paper_modes)!r})"
        )
    if module.value_transform_mode != module.metric_mode:
        mismatches.append(
            f"value_transform_mode={module.value_transform_mode!r} "
            f"(expected {module.metric_mode!r} to match the score transform)"
        )
    mismatches.extend(
        f"{key}={getattr(module, key)!r} (expected {value!r})"
        for key, value in expected.items()
        if getattr(module, key) != value
    )
    if module.base_dim != module.head_dim:
        mismatches.append(f"base_dim={module.base_dim!r} (expected head_dim={module.head_dim})")
    if module.value_dim != module.head_dim:
        mismatches.append(f"value_dim={module.value_dim!r} (expected head_dim={module.head_dim})")
    if hasattr(module, "head_logit_scale"):
        mismatches.append("unexpected learned head_logit_scale parameter")
    if mismatches:
        raise ValueError("paper GT-MHA module mismatch: " + "; ".join(mismatches))

KVCache = tuple[torch.Tensor, torch.Tensor]
ModelKVCache = tuple[KVCache, ...]


def build_attention(
    attention_type: AttentionType,
    d_model: int,
    num_heads: int,
    head_dim: int,
    num_generators: int = 0,
    generator_type: str = "full",
    generator_mixing: str = "softmax",
    dropout: float = 0.0,
    causal: bool = True,
    num_kv_heads: int | None = None,
    theta_init_scale: float = 4.0,
    generator_init_scale: float = 0.02,
    stabilize_generators: bool = True,
    normalize_generators: bool = False,
    head_generator_symmetric_cap: float | None = None,
    base_dim: int | None = None,
    value_dim: int | None = None,
    metric_mode: str = "exp",
    metric_beta: float = 1.0,
    metric_clip_min: float | None = None,
    metric_clip_max: float | None = None,
    value_beta: float | None = None,
    theta_init: str = "balanced_simplex",
    logit_scale_mode: str = "sqrt_dim",
    learn_head_temperature: bool = False,
    value_transform: str = "none",
    num_base_heads: int = 1,
    fuse_base_qkv: bool = False,
    fold_value_transform_into_output: bool = False,
    sdpa_gqa_mode: str = "auto",
) -> nn.Module:
    if attention_type in {"mha", "reduced_mha"}:
        return StandardMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal,
        )
    if attention_type == "mqa":
        return MultiQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal,
        )
    if attention_type == "gqa":
        if num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 4)
        return GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            causal=causal,
        )
    if attention_type == "collaborative":
        return CollaborativeAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal,
            base_dim=base_dim,
            value_dim=value_dim,
        )
    if attention_type == "shared_identity":
        return SharedIdentityAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal,
        )
    if attention_type in LGMA_ATTENTION_TYPES:
        if num_generators <= 0:
            raise ValueError("num_generators must be positive for LGMA")
        if attention_type == "gt_mha_exact":
            # Canonical manuscript equation: M_h = exp(A_h), followed by
            # Q_h K_h^T / sqrt(d_h), with no extra learned temperature.
            metric_mode = "exp"
            logit_scale_mode = "sqrt_dim"
            learn_head_temperature = False
        elif attention_type == "gt_mha_residual":
            # First-order manuscript approximation: I + A_h.
            metric_mode = "residual"
            logit_scale_mode = "sqrt_dim"
            learn_head_temperature = False
        elif attention_type == "gt_mha_quadratic":
            # Second-order manuscript approximation: I + A_h + A_h^2 / 2.
            metric_mode = "quadratic"
            logit_scale_mode = "sqrt_dim"
            learn_head_temperature = False
        elif attention_type == "lgma_v2":
            metric_mode = "exp"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
        elif attention_type == "lgma_residual":
            metric_mode = "residual"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
        elif attention_type == "lgma_quad":
            metric_mode = "quadratic"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
        elif attention_type == "lgma_unconstrained":
            metric_mode = "unconstrained"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
        elif attention_type == "lgma_value_diag":
            metric_mode = "exp"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
            value_transform = "diag"
        elif attention_type == "lgma_multibase":
            metric_mode = "exp"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
            if num_base_heads == 1:
                num_base_heads = 2
        elif attention_type == "lgma_multibase_value_diag":
            metric_mode = "exp"
            logit_scale_mode = "rms_metric"
            learn_head_temperature = True
            value_transform = "diag"
            if num_base_heads == 1:
                num_base_heads = 2
        return LieGeneratedMetricAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            num_generators=num_generators,
            dropout=dropout,
            generator_type=generator_type,
            generator_mixing=generator_mixing,
            causal=causal,
            stabilize_generators=stabilize_generators,
            theta_init_scale=theta_init_scale,
            generator_init_scale=generator_init_scale,
            normalize_generators=normalize_generators,
            head_generator_symmetric_cap=head_generator_symmetric_cap,
            base_dim=base_dim,
            value_dim=value_dim,
            metric_mode=metric_mode,
            metric_beta=metric_beta,
            metric_clip_min=metric_clip_min,
            metric_clip_max=metric_clip_max,
            value_beta=value_beta,
            theta_init=theta_init,
            logit_scale_mode=logit_scale_mode,
            learn_head_temperature=learn_head_temperature,
            value_transform=value_transform,
            num_base_heads=num_base_heads,
            fuse_base_qkv=fuse_base_qkv,
            fold_value_transform_into_output=fold_value_transform_into_output,
            sdpa_gqa_mode=sdpa_gqa_mode,
        )
    raise ValueError(f"unsupported attention_type: {attention_type}")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        attention_type: AttentionType,
        num_generators: int = 0,
        generator_type: str = "full",
        generator_mixing: str = "softmax",
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        causal: bool = True,
        num_kv_heads: int | None = None,
        theta_init_scale: float = 4.0,
        generator_init_scale: float = 0.02,
        stabilize_generators: bool = True,
        normalize_generators: bool = False,
        head_generator_symmetric_cap: float | None = None,
        base_dim: int | None = None,
        value_dim: int | None = None,
        metric_mode: str = "exp",
        metric_beta: float = 1.0,
        metric_clip_min: float | None = None,
        metric_clip_max: float | None = None,
        value_beta: float | None = None,
        theta_init: str = "balanced_simplex",
        logit_scale_mode: str = "sqrt_dim",
        learn_head_temperature: bool = False,
        value_transform: str = "none",
        num_base_heads: int = 1,
        fuse_base_qkv: bool = False,
        fold_value_transform_into_output: bool = False,
        sdpa_gqa_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = build_attention(
            attention_type=attention_type,
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            num_generators=num_generators,
            generator_type=generator_type,
            generator_mixing=generator_mixing,
            dropout=dropout,
            causal=causal,
            num_kv_heads=num_kv_heads,
            theta_init_scale=theta_init_scale,
            generator_init_scale=generator_init_scale,
            stabilize_generators=stabilize_generators,
            normalize_generators=normalize_generators,
            head_generator_symmetric_cap=head_generator_symmetric_cap,
            base_dim=base_dim,
            value_dim=value_dim,
            metric_mode=metric_mode,
            metric_beta=metric_beta,
            metric_clip_min=metric_clip_min,
            metric_clip_max=metric_clip_max,
            value_beta=value_beta,
            theta_init=theta_init,
            logit_scale_mode=logit_scale_mode,
            learn_head_temperature=learn_head_temperature,
            value_transform=value_transform,
            num_base_heads=num_base_heads,
            fuse_base_qkv=fuse_base_qkv,
            fold_value_transform_into_output=fold_value_transform_into_output,
            sdpa_gqa_mode=sdpa_gqa_mode,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = mlp_ratio * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ):
        attention_result = self.attn(
            self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_result
        else:
            attention_output = attention_result
            present_key_value = None
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        if use_cache:
            return x, present_key_value
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        attention_type: AttentionType,
        num_generators: int = 0,
        generator_type: str = "full",
        generator_mixing: str = "softmax",
        context_length: int = 256,
        dropout: float = 0.0,
        num_kv_heads: int | None = None,
        causal: bool = True,
        theta_init_scale: float = 4.0,
        generator_init_scale: float = 0.02,
        stabilize_generators: bool = True,
        normalize_generators: bool = False,
        head_generator_symmetric_cap: float | None = None,
        base_dim: int | None = None,
        value_dim: int | None = None,
        metric_mode: str = "exp",
        metric_beta: float = 1.0,
        metric_clip_min: float | None = None,
        metric_clip_max: float | None = None,
        value_beta: float | None = None,
        theta_init: str = "balanced_simplex",
        logit_scale_mode: str = "sqrt_dim",
        learn_head_temperature: bool = False,
        value_transform: str = "none",
        num_base_heads: int = 1,
        fuse_base_qkv: bool = False,
        fold_value_transform_into_output: bool = False,
        sdpa_gqa_mode: str = "auto",
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        self.causal = causal
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    attention_type=attention_type,
                    num_generators=num_generators,
                    generator_type=generator_type,
                    generator_mixing=generator_mixing,
                    dropout=dropout,
                    causal=causal,
                    num_kv_heads=num_kv_heads,
                    theta_init_scale=theta_init_scale,
                    generator_init_scale=generator_init_scale,
                    stabilize_generators=stabilize_generators,
                    normalize_generators=normalize_generators,
                    head_generator_symmetric_cap=head_generator_symmetric_cap,
                    base_dim=base_dim,
                    value_dim=value_dim,
                    metric_mode=metric_mode,
                    metric_beta=metric_beta,
                    metric_clip_min=metric_clip_min,
                    metric_clip_max=metric_clip_max,
                    value_beta=value_beta,
                    theta_init=theta_init,
                    logit_scale_mode=logit_scale_mode,
                    learn_head_temperature=learn_head_temperature,
                    value_transform=value_transform,
                    num_base_heads=num_base_heads,
                    fuse_base_qkv=fuse_base_qkv,
                    fold_value_transform_into_output=fold_value_transform_into_output,
                    sdpa_gqa_mode=sdpa_gqa_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def first_attention(self) -> nn.Module:
        return self.blocks[0].attn

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: ModelKVCache | None = None,
        use_cache: bool = False,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if targets is not None and use_cache:
            raise ValueError("KV caching is only supported when targets is None")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per Transformer layer")
        batch, seq_len = input_ids.shape
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[-2]
            for layer_cache in past_key_values:
                if layer_cache[0].shape[-2] != past_length or layer_cache[1].shape[-2] != past_length:
                    raise ValueError("all layer caches must have the same sequence length")
        if past_length + seq_len > self.context_length:
            raise ValueError(
                f"cached length {past_length} plus sequence length {seq_len} exceeds "
                f"context_length {self.context_length}"
            )
        positions = torch.arange(past_length, past_length + seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        present_key_values = []
        for layer_index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            if use_cache:
                x, layer_present = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(layer_present)
            else:
                x = block(x)
        logits = self.lm_head(self.norm(x))
        if use_cache:
            return logits, tuple(present_key_values)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(batch * seq_len, self.vocab_size), targets.reshape(-1))
        return logits, loss


def load_model_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tiny_lm_from_config(path: str | Path, vocab_size: int) -> TinyTransformerLM:
    config = load_model_config(path)
    return TinyTransformerLM(vocab_size=vocab_size, **config)
