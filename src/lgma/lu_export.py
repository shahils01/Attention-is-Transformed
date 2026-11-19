from __future__ import annotations

import math
from copy import deepcopy

import torch

from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import GroupedQueryAttention
from lgma.transformer import TinyTransformerLM


@torch.no_grad()
def fold_lu_attention(module: LieGeneratedMetricAttention) -> GroupedQueryAttention:
    """Fold a fixed-base LU layer into an independent, eval-mode GQA layer.

    The result uses the standard GQA mask API. Use the score dtype for
    additive masks and 4-D shapes for per-head masks. Each masked row should
    retain a visible key for parity with the LU SDPA path.
    """
    if not isinstance(module, LieGeneratedMetricAttention) or module.metric_mode != "lu":
        raise ValueError("folding requires fixed-base LU attention")
    if module.base_dim != module.head_dim or module.value_dim != module.head_dim:
        raise ValueError("standard GQA requires base_dim == value_dim == head_dim")
    if module.num_base_heads != module.num_value_base_heads:
        raise ValueError("standard GQA requires matching key and value groups")
    if module.value_transform_mode != "none":
        raise ValueError("folding requires value_transform='none'")

    query = module.q_proj
    projections = (query, module.k_proj, module.v_proj, module.out_proj)
    bias = query.bias is not None
    if any((projection.bias is not None) != bias for projection in projections):
        raise ValueError("GQA projections must either all have biases or all omit them")
    device, dtype = query.weight.device, query.weight.dtype
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(f"unsupported LU weight dtype {dtype}")
    for name, parameter in module.named_parameters():
        if parameter.device != device or parameter.dtype != dtype:
            raise ValueError(f"{name} must use the query projection device and dtype")
        if not torch.isfinite(parameter).all():
            raise ValueError(f"LU attention parameter {name} contains nonfinite values")

    # Use the source attention's runtime denominator, including its RMS and
    # temperature clamps, then convert it to the ordinary GQA score scale.
    fold_dtype = torch.float64 if device.type in {"cpu", "cuda"} else torch.float32
    runtime_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    with torch.autocast(device_type=device.type, enabled=False):
        log_pivots = getattr(module, "lu_log_diag", None)
        if log_pivots is not None:
            pivots = log_pivots.to(runtime_dtype).exp()
            if not (torch.isfinite(pivots) & (pivots > 0)).all():
                raise ValueError("LU pivots must be finite and positive at the source dtype")
        runtime_metrics = module.compute_metrics(dtype=runtime_dtype)
        if not torch.isfinite(runtime_metrics.to(dtype)).all():
            raise ValueError("LU metrics contain nonfinite values at the source dtype")
        denominator = module._score_scale(metrics=runtime_metrics)
        if isinstance(denominator, torch.Tensor):
            if not (torch.isfinite(denominator) & (denominator > 0)).all():
                raise ValueError("LU score denominators must be finite and positive")
            scale = math.sqrt(module.head_dim) / denominator.to(fold_dtype)
        else:
            scale = math.sqrt(module.head_dim) / denominator

        metrics = (
            runtime_metrics
            if runtime_dtype == fold_dtype
            else module.compute_metrics(dtype=fold_dtype)
        )
        # F.linear stores each base's weights as [output, input]. Since an LU
        # head computes the row-vector product q M, its effective weights are
        # M^T W_Q, in base-major order with the identity anchor first.
        groups = module.num_base_heads
        dim = module.head_dim
        if isinstance(scale, torch.Tensor):
            metrics = metrics * scale[:, None, None]
        else:
            metrics = metrics * scale
        transforms = metrics.reshape(groups, module.generated_heads_per_base, dim, dim)
        transforms = transforms.transpose(-1, -2)
        base_weight = query.weight.to(fold_dtype).reshape(groups, 1, dim, module.d_model)
        weight = (transforms @ base_weight).reshape(-1, module.d_model).to(dtype)
        if not torch.isfinite(weight).all():
            raise ValueError("folded query weights are nonfinite at the export dtype")
        bias_value = None
        if query.bias is not None:
            base_bias = query.bias.to(fold_dtype).reshape(groups, 1, dim, 1)
            bias_value = (transforms @ base_bias).reshape(-1).to(dtype)
            if not torch.isfinite(bias_value).all():
                raise ValueError("folded query biases are nonfinite at the export dtype")

    # The destination's initial random weights are all replaced.
    with torch.random.fork_rng(devices=[]):
        folded = GroupedQueryAttention(
            d_model=module.d_model,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            num_kv_heads=groups,
            dropout=module.dropout,
            bias=bias,
            causal=module.causal,
        ).to(device=device, dtype=dtype)
    folded.q_proj.weight.copy_(weight)
    if folded.q_proj.bias is not None:
        folded.q_proj.bias.copy_(bias_value)
    folded.k_proj.load_state_dict(module.k_proj.state_dict())
    folded.v_proj.load_state_dict(module.v_proj.state_dict())
    folded.out_proj.load_state_dict(module.out_proj.state_dict())
    return folded.eval()


def export_lu_model(model: TinyTransformerLM) -> TinyTransformerLM:
    """Copy a TinyTransformerLM and fold every attention layer for inference."""
    if type(model) is not TinyTransformerLM or not model.blocks:
        raise ValueError("export requires a TinyTransformerLM with at least one layer")

    replacements = {}
    layout = None
    for index, block in enumerate(model.blocks):
        try:
            folded = fold_lu_attention(block.attn)
        except ValueError as exc:
            raise ValueError(f"attention layer {index}: {exc}") from exc
        if folded.q_proj.bias is not None:
            raise ValueError(
                f"attention layer {index}: TinyTransformerLM requires bias-free attention"
            )
        current_layout = (folded.num_heads, folded.head_dim, folded.num_kv_heads)
        if layout is None:
            layout = current_layout
        if current_layout != layout or folded.d_model != model.d_model:
            raise ValueError(f"attention layer {index}: inconsistent GQA projection layout")
        if folded.causal != model.causal or folded.dropout != model.drop.p:
            raise ValueError(f"attention layer {index}: inconsistent causal or dropout settings")
        replacements[id(block.attn)] = folded

    # The memo substitutes attention layers without copying their LU tensors.
    return deepcopy(model, memo=replacements).eval()
