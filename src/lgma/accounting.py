from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class AttentionAccounting:
    total_parameters: int
    qkv_parameters: int
    q_parameters: int
    k_parameters: int
    v_parameters: int
    generator_parameters: int
    kv_cache_bytes_per_token_per_layer: int
    attention_score_flops: int
    attention_maps: int
    base_heads: int
    generated_heads_per_base: int


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    parameters = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def count_named_parameters(
    module: nn.Module,
    substrings: tuple[str, ...],
    trainable_only: bool = True,
) -> int:
    total = 0
    for name, param in module.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        if any(part in name for part in substrings):
            total += param.numel()
    return total


def qkv_parameter_count(module: nn.Module) -> int:
    return count_named_parameters(module, ("q_proj", "k_proj", "v_proj"))


def q_parameter_count(module: nn.Module) -> int:
    return count_named_parameters(module, ("q_proj",))


def k_parameter_count(module: nn.Module) -> int:
    return count_named_parameters(module, ("k_proj",))


def v_parameter_count(module: nn.Module) -> int:
    return count_named_parameters(module, ("v_proj",))


def generator_parameter_count(module: nn.Module) -> int:
    return count_named_parameters(
        module,
        (
            "generators",
            "theta",
            "raw_metrics",
            "raw_value_transforms",
            "mixing_vector",
        ),
    )


def dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def kv_cache_bytes_per_token_per_layer(
    attention_module: nn.Module,
    dtype: torch.dtype = torch.float16,
) -> int:
    size = dtype_size(dtype)
    if hasattr(attention_module, "num_kv_heads"):
        return 2 * attention_module.num_kv_heads * attention_module.head_dim * size
    if attention_module.__class__.__name__ == "CollaborativeAttention":
        # Keys are shared, while values remain independently projected per head.
        return (
            attention_module.base_dim
            + attention_module.num_heads * attention_module.value_dim
        ) * size
    if attention_module.__class__.__name__ == "LieGeneratedMetricAttention":
        base_dim = getattr(attention_module, "base_dim", attention_module.head_dim)
        value_dim = getattr(attention_module, "value_dim", attention_module.head_dim)
        num_base_heads = getattr(attention_module, "num_base_heads", 1)
        return num_base_heads * (base_dim + value_dim) * size
    if attention_module.__class__.__name__ == "SharedIdentityAttention":
        return 2 * attention_module.head_dim * size
    if hasattr(attention_module, "num_heads") and hasattr(attention_module, "head_dim"):
        return 2 * attention_module.num_heads * attention_module.head_dim * size
    raise ValueError("attention module does not expose head/cache dimensions")


def attention_score_flops(
    attention_module: nn.Module,
    sequence_length: int,
    batch_size: int = 1,
) -> int:
    if not hasattr(attention_module, "num_heads") or not hasattr(attention_module, "head_dim"):
        raise ValueError("attention module does not expose num_heads/head_dim")
    score_dim = getattr(attention_module, "base_dim", attention_module.head_dim)
    # Multiply-add counted as two FLOPs.
    return (
        2
        * batch_size
        * attention_module.num_heads
        * sequence_length
        * sequence_length
        * score_dim
    )


def attention_accounting(
    attention_module: nn.Module,
    sequence_length: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> AttentionAccounting:
    return AttentionAccounting(
        total_parameters=count_parameters(attention_module),
        qkv_parameters=qkv_parameter_count(attention_module),
        q_parameters=q_parameter_count(attention_module),
        k_parameters=k_parameter_count(attention_module),
        v_parameters=v_parameter_count(attention_module),
        generator_parameters=generator_parameter_count(attention_module),
        kv_cache_bytes_per_token_per_layer=kv_cache_bytes_per_token_per_layer(
            attention_module, dtype=dtype
        ),
        attention_score_flops=attention_score_flops(
            attention_module, sequence_length=sequence_length, batch_size=batch_size
        ),
        attention_maps=attention_module.num_heads,
        base_heads=getattr(attention_module, "num_base_heads", attention_module.num_heads),
        generated_heads_per_base=getattr(attention_module, "generated_heads_per_base", 1),
    )


def measure_tokens_per_second(
    step_fn: Callable[[], object],
    tokens_per_step: int,
    warmup_steps: int = 2,
    measured_steps: int = 10,
) -> float:
    for _ in range(warmup_steps):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measured_steps):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return tokens_per_step * measured_steps / max(elapsed, 1e-12)


def reset_peak_memory(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device | str | None = None) -> int:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device)
    return 0
