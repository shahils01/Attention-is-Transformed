from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from adapter_trainer.lgma_llama_attention import LlamaLgmaAttention


@dataclass(frozen=True)
class AttentionReplacementSummary:
    layer_idx: int | None
    teacher_hidden_size: int
    teacher_num_heads: int
    teacher_num_kv_heads: int
    teacher_head_dim: int
    teacher_q_shape: tuple[int, ...]
    teacher_k_shape: tuple[int, ...]
    teacher_v_shape: tuple[int, ...]
    teacher_o_shape: tuple[int, ...]
    teacher_parameters: int
    teacher_kv_cache_bytes_per_token: int
    student_variant: str
    student_num_heads: int
    student_qk_base_heads: int
    student_value_base_heads: int
    student_base_dim: int
    student_value_dim: int
    student_qk_generated_heads_per_base: int
    student_value_generated_heads_per_base: int
    student_q_width: int
    student_k_width: int
    student_v_width: int
    student_parameters: int
    student_kv_cache_bytes_per_token: int


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def _teacher_num_heads(attention: nn.Module) -> int:
    config = getattr(attention, "config", None)
    return int(getattr(attention, "num_heads", getattr(config, "num_attention_heads", 0)))


def _teacher_num_kv_heads(attention: nn.Module, num_heads: int) -> int:
    config = getattr(attention, "config", None)
    return int(getattr(attention, "num_key_value_heads", getattr(config, "num_key_value_heads", num_heads)))


def _teacher_hidden_size(attention: nn.Module) -> int:
    config = getattr(attention, "config", None)
    hidden_size = int(getattr(attention, "hidden_size", getattr(config, "hidden_size", 0)))
    return hidden_size if hidden_size > 0 else int(attention.q_proj.in_features)


def _shape(layer: nn.Module) -> tuple[int, ...]:
    return tuple(layer.weight.shape)


def teacher_kv_cache_bytes_per_token(
    attention: nn.Module,
    dtype: torch.dtype = torch.float16,
) -> int:
    hidden_size = _teacher_hidden_size(attention)
    num_heads = _teacher_num_heads(attention)
    num_kv_heads = _teacher_num_kv_heads(attention, num_heads)
    head_dim = hidden_size // num_heads
    return 2 * num_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()


def student_kv_cache_bytes_per_token(
    attention: LlamaLgmaAttention,
    dtype: torch.dtype = torch.float16,
) -> int:
    return (
        attention.qk_num_base_heads * attention.base_dim
        + attention.value_num_base_heads * attention.value_dim
    ) * torch.empty((), dtype=dtype).element_size()


def build_replacement_summary(
    teacher_attention: nn.Module,
    student_attention: LlamaLgmaAttention,
    *,
    layer_idx: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> AttentionReplacementSummary:
    teacher_hidden_size = _teacher_hidden_size(teacher_attention)
    teacher_num_heads = _teacher_num_heads(teacher_attention)
    teacher_num_kv_heads = _teacher_num_kv_heads(teacher_attention, teacher_num_heads)
    teacher_head_dim = teacher_hidden_size // teacher_num_heads
    return AttentionReplacementSummary(
        layer_idx=layer_idx,
        teacher_hidden_size=teacher_hidden_size,
        teacher_num_heads=teacher_num_heads,
        teacher_num_kv_heads=teacher_num_kv_heads,
        teacher_head_dim=teacher_head_dim,
        teacher_q_shape=_shape(teacher_attention.q_proj),
        teacher_k_shape=_shape(teacher_attention.k_proj),
        teacher_v_shape=_shape(teacher_attention.v_proj),
        teacher_o_shape=_shape(teacher_attention.o_proj),
        teacher_parameters=count_parameters(teacher_attention),
        teacher_kv_cache_bytes_per_token=teacher_kv_cache_bytes_per_token(
            teacher_attention,
            dtype=dtype,
        ),
        student_variant=student_attention.attention_variant,
        student_num_heads=student_attention.num_heads,
        student_qk_base_heads=student_attention.qk_num_base_heads,
        student_value_base_heads=student_attention.value_num_base_heads,
        student_base_dim=student_attention.base_dim,
        student_value_dim=student_attention.value_dim,
        student_qk_generated_heads_per_base=student_attention.qk_generated_heads_per_base,
        student_value_generated_heads_per_base=student_attention.value_generated_heads_per_base,
        student_q_width=student_attention.q_proj.out_features,
        student_k_width=student_attention.k_proj.out_features,
        student_v_width=student_attention.v_proj.out_features,
        student_parameters=count_parameters(student_attention),
        student_kv_cache_bytes_per_token=student_kv_cache_bytes_per_token(
            student_attention,
            dtype=dtype,
        ),
    )


def format_replacement_summary(summary: AttentionReplacementSummary) -> str:
    layer = "?" if summary.layer_idx is None else str(summary.layer_idx)
    param_delta = summary.teacher_parameters - summary.student_parameters
    cache_delta = (
        summary.teacher_kv_cache_bytes_per_token - summary.student_kv_cache_bytes_per_token
    )
    return "\n".join(
        [
            f"[LGMA replacement] layer={layer}",
            (
                "  teacher: "
                f"hidden={summary.teacher_hidden_size}, "
                f"heads={summary.teacher_num_heads}, "
                f"kv_heads={summary.teacher_num_kv_heads}, "
                f"head_dim={summary.teacher_head_dim}, "
                f"q={summary.teacher_q_shape}, "
                f"k={summary.teacher_k_shape}, "
                f"v={summary.teacher_v_shape}, "
                f"o={summary.teacher_o_shape}"
            ),
            (
                "  student: "
                f"variant={summary.student_variant}, "
                f"heads={summary.student_num_heads}, "
                f"qk_base_heads={summary.student_qk_base_heads}, "
                f"value_base_heads={summary.student_value_base_heads}, "
                f"base_dim={summary.student_base_dim}, "
                f"value_dim={summary.student_value_dim}, "
                f"qk_generated_per_base={summary.student_qk_generated_heads_per_base}, "
                f"value_generated_per_base={summary.student_value_generated_heads_per_base}, "
                f"q_width={summary.student_q_width}, "
                f"k_width={summary.student_k_width}, "
                f"v_width={summary.student_v_width}"
            ),
            (
                "  accounting: "
                f"teacher_params={summary.teacher_parameters:,}, "
                f"student_params={summary.student_parameters:,}, "
                f"param_delta={param_delta:,}, "
                f"teacher_kv_cache_bytes/token={summary.teacher_kv_cache_bytes_per_token}, "
                f"student_kv_cache_bytes/token={summary.student_kv_cache_bytes_per_token}, "
                f"kv_cache_delta={cache_delta}"
            ),
        ]
    )


def summary_as_dict(summary: AttentionReplacementSummary) -> dict[str, Any]:
    return summary.__dict__.copy()
