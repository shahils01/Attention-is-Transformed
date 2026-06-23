from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _negative_large(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


@dataclass(frozen=True)
class LlamaLgmaConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    qk_num_base_heads: int = 2
    value_num_base_heads: int = 2
    num_generators: int = 4
    attention_variant: str = "lgma_residual"
    generator_type: str = "full"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    layer_idx: int | None = None
    return_tuple_len: int = 3


def variant_settings(attention_variant: str) -> dict[str, Any]:
    if attention_variant == "lgma":
        return {
            "metric_mode": "exp",
            "logit_scale_mode": "sqrt_dim",
            "learn_head_temperature": False,
            "theta_init": "random_sphere",
            "value_transform": "none",
        }
    if attention_variant == "lgma_v2":
        return {
            "metric_mode": "exp",
            "logit_scale_mode": "rms_metric",
            "learn_head_temperature": True,
            "theta_init": "circle",
            "value_transform": "none",
        }
    if attention_variant == "lgma_residual":
        return {
            "metric_mode": "residual",
            "logit_scale_mode": "rms_metric",
            "learn_head_temperature": True,
            "theta_init": "circle",
            "value_transform": "none",
        }
    if attention_variant == "lgma_multibase":
        return {
            "metric_mode": "exp",
            "logit_scale_mode": "rms_metric",
            "learn_head_temperature": True,
            "theta_init": "circle",
            "value_transform": "none",
        }
    if attention_variant == "lgma_quad":
        return {
            "metric_mode": "quadratic",
            "logit_scale_mode": "rms_metric",
            "learn_head_temperature": True,
            "theta_init": "circle",
            "value_transform": "none",
        }
    if attention_variant == "lgma_unconstrained":
        return {
            "metric_mode": "unconstrained",
            "logit_scale_mode": "rms_metric",
            "learn_head_temperature": True,
            "theta_init": "random_sphere",
            "value_transform": "none",
        }
    if attention_variant in {"lgma_value_diag", "lgma_multibase_value_diag"}:
        return {
            "metric_mode": "exp",
            "logit_scale_mode": "rms_metric",
            "learn_head_temperature": True,
            "theta_init": "circle",
            "value_transform": "diag",
        }
    raise ValueError(f"unsupported attention_variant: {attention_variant}")


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, seq_len, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        inv_freq = self.inv_freq.to(device=x.device)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)[:, None, :, :]
        sin = emb.sin().to(dtype=x.dtype)[:, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaLgmaAttention(nn.Module):
    """LGMA attention with a Hugging Face Llama-compatible forward signature.

    The module intentionally lives outside ``src/lgma`` so TinyLlama replacement
    experiments do not affect the existing from-scratch TinyStories/VLA code.
    """

    def __init__(self, config: LlamaLgmaConfig) -> None:
        super().__init__()
        if config.hidden_size <= 0 or config.num_attention_heads <= 0:
            raise ValueError("hidden_size and num_attention_heads must be positive")
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.qk_num_base_heads <= 0 or config.value_num_base_heads <= 0:
            raise ValueError("base-head counts must be positive")
        if config.num_attention_heads % config.qk_num_base_heads != 0:
            raise ValueError("num_attention_heads must be divisible by qk_num_base_heads")
        if config.num_attention_heads % config.value_num_base_heads != 0:
            raise ValueError("num_attention_heads must be divisible by value_num_base_heads")
        if config.num_generators <= 0:
            raise ValueError("num_generators must be positive")
        if config.generator_type not in {"full", "diagonal", "symmetric"}:
            raise ValueError(f"unsupported generator_type: {config.generator_type}")

        settings = variant_settings(config.attention_variant)
        if settings["metric_mode"] == "unconstrained" and config.generator_type == "diagonal":
            raise ValueError("lgma_unconstrained requires full or symmetric generators")

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base_dim = self.head_dim
        self.value_dim = self.head_dim
        self.qk_num_base_heads = config.qk_num_base_heads
        self.value_num_base_heads = config.value_num_base_heads
        self.qk_generated_heads_per_base = self.num_heads // self.qk_num_base_heads
        self.value_generated_heads_per_base = self.num_heads // self.value_num_base_heads
        self.num_generators = config.num_generators
        self.attention_variant = config.attention_variant
        self.generator_type = config.generator_type
        self.dropout = config.attention_dropout
        self.layer_idx = config.layer_idx
        self.metric_mode = settings["metric_mode"]
        self.logit_scale_mode = settings["logit_scale_mode"]
        self.learn_head_temperature = settings["learn_head_temperature"]
        self.theta_init = settings["theta_init"]
        self.value_transform = settings["value_transform"]
        if config.return_tuple_len not in {2, 3}:
            raise ValueError("return_tuple_len must be 2 or 3")
        self.return_tuple_len = config.return_tuple_len

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.qk_num_base_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.qk_num_base_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.value_num_base_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        if self.metric_mode == "unconstrained":
            self.raw_metrics = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        elif self.generator_type == "diagonal":
            self.generators = nn.Parameter(torch.empty(self.num_generators, self.head_dim))
        else:
            self.generators = nn.Parameter(
                torch.empty(self.num_generators, self.head_dim, self.head_dim)
            )
        if self.metric_mode != "unconstrained":
            self.theta = nn.Parameter(torch.empty(self.num_heads, self.num_generators))
        if self.learn_head_temperature:
            self.head_logit_scale = nn.Parameter(torch.ones(self.num_heads))
        if self.value_transform == "diag":
            self.value_scale = nn.Parameter(torch.ones(self.num_heads, self.head_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.metric_mode == "unconstrained":
            nn.init.zeros_(self.raw_metrics)
        else:
            std = 0.02 / math.sqrt(self.head_dim)
            nn.init.normal_(self.generators, mean=0.0, std=std)
            self._init_head_coordinates(self.theta)
        if hasattr(self, "head_logit_scale"):
            nn.init.ones_(self.head_logit_scale)
        if hasattr(self, "value_scale"):
            nn.init.ones_(self.value_scale)

    def _init_head_coordinates(self, coordinates: torch.Tensor) -> None:
        with torch.no_grad():
            if self.theta_init == "circle" and self.num_generators >= 2:
                head_ids = torch.arange(
                    self.num_heads,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                )
                generated_ids = head_ids.remainder(self.qk_generated_heads_per_base)
                angles = 2 * math.pi * generated_ids / self.qk_generated_heads_per_base
                directions = torch.zeros_like(coordinates)
                directions[:, 0] = torch.cos(angles)
                directions[:, 1] = torch.sin(angles)
                if self.num_generators > 2:
                    directions[:, 2:] = 0.01 * torch.randn_like(directions[:, 2:])
            else:
                directions = torch.randn_like(coordinates)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            coordinates.copy_(directions * 0.02)

    @classmethod
    def from_teacher_attention(
        cls,
        teacher_attention: nn.Module,
        *,
        qk_num_base_heads: int,
        value_num_base_heads: int,
        num_generators: int = 4,
        attention_variant: str = "lgma_residual",
        generator_type: str = "full",
        layer_idx: int | None = None,
    ) -> "LlamaLgmaAttention":
        teacher_config = getattr(teacher_attention, "config", None)
        hidden_size = int(
            getattr(teacher_attention, "hidden_size", getattr(teacher_config, "hidden_size", 0))
        )
        num_heads = int(
            getattr(
                teacher_attention,
                "num_heads",
                getattr(teacher_config, "num_attention_heads", 0),
            )
        )
        num_key_value_heads = int(
            getattr(
                teacher_attention,
                "num_key_value_heads",
                getattr(teacher_config, "num_key_value_heads", num_heads),
            )
        )
        if hidden_size <= 0:
            hidden_size = int(teacher_attention.q_proj.in_features)
        if num_heads <= 0:
            head_dim = int(getattr(teacher_attention, "head_dim", 0))
            if head_dim <= 0:
                raise ValueError("could not infer teacher num_heads/head_dim")
            num_heads = hidden_size // head_dim
        config = LlamaLgmaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            qk_num_base_heads=qk_num_base_heads,
            value_num_base_heads=value_num_base_heads,
            num_generators=num_generators,
            attention_variant=attention_variant,
            generator_type=generator_type,
            attention_dropout=float(
                getattr(teacher_attention, "attention_dropout", getattr(teacher_config, "attention_dropout", 0.0))
            ),
            attention_bias=bool(
                getattr(teacher_config, "attention_bias", teacher_attention.q_proj.bias is not None)
            ),
            rope_theta=float(getattr(teacher_config, "rope_theta", 10000.0)),
            max_position_embeddings=int(getattr(teacher_config, "max_position_embeddings", 2048)),
            layer_idx=layer_idx,
            return_tuple_len=infer_attention_return_tuple_len(teacher_attention),
        )
        student = cls(config)
        initialize_from_teacher(student, teacher_attention)
        return student

    def _dense_generators(self) -> torch.Tensor:
        if self.generator_type == "diagonal":
            return torch.diag_embed(self.generators)
        generators = self.generators
        if self.generator_type == "symmetric":
            generators = 0.5 * (generators + generators.transpose(-1, -2))
        trace = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        eye = torch.eye(self.head_dim, device=generators.device, dtype=generators.dtype)
        return generators - (trace[..., None, None] / self.head_dim) * eye

    def compute_metrics(self) -> torch.Tensor:
        if self.metric_mode == "unconstrained":
            eye = torch.eye(self.head_dim, device=self.raw_metrics.device, dtype=self.raw_metrics.dtype)
            return eye[None, :, :] + self.raw_metrics
        if self.generator_type == "diagonal":
            diagonal = torch.einsum("hm,md->hd", self.theta, self.generators)
            if self.metric_mode == "residual":
                return torch.diag_embed(1.0 + diagonal)
            if self.metric_mode == "quadratic":
                return torch.diag_embed(1.0 + diagonal + 0.5 * diagonal.square())
            return torch.diag_embed(torch.exp(diagonal))
        head_generators = torch.einsum("hm,mde->hde", self.theta, self._dense_generators())
        if self.metric_mode == "residual":
            eye = torch.eye(self.head_dim, device=head_generators.device, dtype=head_generators.dtype)
            return eye[None, :, :] + head_generators
        if self.metric_mode == "quadratic":
            eye = torch.eye(self.head_dim, device=head_generators.device, dtype=head_generators.dtype)
            return eye[None, :, :] + head_generators + 0.5 * torch.matmul(
                head_generators,
                head_generators,
            )
        return torch.linalg.matrix_exp(head_generators.float()).to(dtype=head_generators.dtype)

    def _reshape_qk_base(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.qk_num_base_heads, self.head_dim).transpose(1, 2)

    def _reshape_value_base(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.value_num_base_heads, self.head_dim).transpose(1, 2)

    def _apply_metric_to_queries(
        self,
        q_base: torch.Tensor,
        metrics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            metrics is None
            and self.generator_type == "diagonal"
            and self.metric_mode != "unconstrained"
        ):
            diagonal = torch.einsum("hm,md->hd", self.theta, self.generators)
            if self.metric_mode == "residual":
                scale = 1.0 + diagonal
            elif self.metric_mode == "quadratic":
                scale = 1.0 + diagonal + 0.5 * diagonal.square()
            else:
                scale = torch.exp(diagonal)
            scale = scale.view(self.qk_num_base_heads, self.qk_generated_heads_per_base, self.head_dim)
            q_metric = q_base[:, :, None, :, :] * scale[None, :, :, None, :]
        else:
            if metrics is None:
                metrics = self.compute_metrics()
            metrics_by_base = metrics.view(
                self.qk_num_base_heads,
                self.qk_generated_heads_per_base,
                self.head_dim,
                self.head_dim,
            )
            q_metric = torch.einsum("nbtd,bkde->nbkte", q_base, metrics_by_base)
        batch, _, _, seq_len, _ = q_metric.shape
        return q_metric.reshape(batch, self.num_heads, seq_len, self.head_dim)

    def _expand_keys(self, k_base: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = k_base.shape
        return (
            k_base[:, :, None, :, :]
            .expand(
                batch,
                self.qk_num_base_heads,
                self.qk_generated_heads_per_base,
                seq_len,
                self.head_dim,
            )
            .reshape(batch, self.num_heads, seq_len, self.head_dim)
        )

    def _expand_values(self, v_base: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = v_base.shape
        values = (
            v_base[:, :, None, :, :]
            .expand(
                batch,
                self.value_num_base_heads,
                self.value_generated_heads_per_base,
                seq_len,
                self.head_dim,
            )
            .reshape(batch, self.num_heads, seq_len, self.head_dim)
        )
        if self.value_transform == "diag":
            values = values * self.value_scale[None, :, None, :]
        return values

    def _score_scale(
        self,
        dtype: torch.dtype,
        device: torch.device,
        metrics: torch.Tensor | None = None,
    ) -> torch.Tensor | float:
        if self.logit_scale_mode == "sqrt_dim" and not hasattr(self, "head_logit_scale"):
            return math.sqrt(self.head_dim)
        denom = torch.full((self.num_heads,), math.sqrt(self.head_dim), dtype=dtype, device=device)
        if self.logit_scale_mode == "rms_metric":
            if metrics is None:
                metrics = self.compute_metrics()
            rms = metrics.float().pow(2).sum(dim=(-1, -2)).div(self.head_dim).sqrt()
            denom = denom * rms.to(device=device, dtype=dtype).clamp_min(1e-6)
        if hasattr(self, "head_logit_scale"):
            denom = denom / self.head_logit_scale.to(device=device, dtype=dtype).clamp_min(1e-6)
        return denom

    @staticmethod
    def _prepare_attention_mask(
        scores: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        if is_causal and attention_mask is None:
            _, _, target_len, source_len = scores.shape
            causal_mask = torch.ones(
                target_len,
                source_len,
                device=scores.device,
                dtype=torch.bool,
            ).triu(source_len - target_len + 1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], _negative_large(scores.dtype))
        if attention_mask is None:
            return scores
        mask = attention_mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            scores = scores.masked_fill(mask, _negative_large(scores.dtype))
        else:
            scores = scores + mask.to(dtype=scores.dtype)
        return scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
        if past_key_value is not None or use_cache:
            raise NotImplementedError("LGMA TinyLlama adapter training currently disables KV cache")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, seq_len, hidden_size]")

        q_base = self._reshape_qk_base(self.q_proj(hidden_states))
        k_base = self._reshape_qk_base(self.k_proj(hidden_states))
        v_base = self._reshape_value_base(self.v_proj(hidden_states))

        if position_embeddings is None:
            cos, sin = self.rotary_emb(q_base, position_ids)
        else:
            cos, sin = position_embeddings
            cos = cos[:, None, :, :] if cos.ndim == 3 else cos
            sin = sin[:, None, :, :] if sin.ndim == 3 else sin
        q_base, k_base = apply_rotary_pos_emb(q_base, k_base, cos, sin)

        metrics = self.compute_metrics() if self.logit_scale_mode == "rms_metric" else None
        q_metric = self._apply_metric_to_queries(q_base, metrics=metrics)
        k_heads = self._expand_keys(k_base)
        v_heads = self._expand_values(v_base)

        scores = torch.einsum("bhtd,bhsd->bhts", q_metric, k_heads)
        scale = self._score_scale(scores.dtype, scores.device, metrics=metrics)
        if isinstance(scale, torch.Tensor):
            scores = scores / scale[None, :, None, None]
        else:
            scores = scores / scale
        scores = self._prepare_attention_mask(scores, attention_mask, is_causal=True)
        attn_weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.einsum("bhts,bhsd->bhtd", attn_weights, v_heads)
        batch, _, seq_len, _ = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch,
            seq_len,
            self.hidden_size,
        )
        attn_output = self.o_proj(attn_output)
        if self.return_tuple_len == 2:
            return attn_output, attn_weights if output_attentions else None
        return attn_output, attn_weights if output_attentions else None, None


def _average_heads(weight: torch.Tensor, num_source_heads: int, num_target_heads: int) -> torch.Tensor:
    head_dim = weight.shape[0] // num_source_heads
    heads = weight.view(num_source_heads, head_dim, weight.shape[1])
    if num_source_heads == num_target_heads:
        return heads.reshape(num_target_heads * head_dim, weight.shape[1])
    if num_source_heads % num_target_heads == 0:
        group = num_source_heads // num_target_heads
        return heads.view(num_target_heads, group, head_dim, weight.shape[1]).mean(dim=1).reshape(
            num_target_heads * head_dim,
            weight.shape[1],
        )
    indices = torch.linspace(0, num_source_heads - 1, num_target_heads, device=weight.device).round().long()
    return heads.index_select(0, indices).reshape(num_target_heads * head_dim, weight.shape[1])


def infer_attention_return_tuple_len(teacher_attention: nn.Module) -> int:
    try:
        source = inspect.getsource(teacher_attention.forward)
    except (OSError, TypeError):
        return 3
    if "attn_output, attn_weights, past_key_value" in source:
        return 3
    if "attn_output, attn_weights" in source:
        return 2
    return 3


def initialize_from_teacher(student: LlamaLgmaAttention, teacher_attention: nn.Module) -> None:
    """Warm-start base projections by averaging teacher heads when possible."""
    with torch.no_grad():
        student.q_proj.weight.copy_(
            _average_heads(
                teacher_attention.q_proj.weight.detach(),
                student.num_heads,
                student.qk_num_base_heads,
            )
        )
        teacher_k_heads = teacher_attention.k_proj.weight.shape[0] // student.head_dim
        teacher_v_heads = teacher_attention.v_proj.weight.shape[0] // student.head_dim
        student.k_proj.weight.copy_(
            _average_heads(
                teacher_attention.k_proj.weight.detach(),
                teacher_k_heads,
                student.qk_num_base_heads,
            )
        )
        student.v_proj.weight.copy_(
            _average_heads(
                teacher_attention.v_proj.weight.detach(),
                teacher_v_heads,
                student.value_num_base_heads,
            )
        )
        student.o_proj.weight.copy_(teacher_attention.o_proj.weight.detach())
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            student_layer = getattr(student, name)
            teacher_layer = getattr(teacher_attention, name)
            if student_layer.bias is not None and teacher_layer.bias is not None:
                if name in {"q_proj", "k_proj"}:
                    source_heads = student.num_heads if name == "q_proj" else teacher_k_heads
                    student_layer.bias.copy_(
                        _average_heads(
                            teacher_layer.bias.detach()[:, None],
                            source_heads,
                            student.qk_num_base_heads,
                        ).squeeze(-1)
                    )
                elif name == "v_proj":
                    student_layer.bias.copy_(
                        _average_heads(
                            teacher_layer.bias.detach()[:, None],
                            teacher_v_heads,
                            student.value_num_base_heads,
                        ).squeeze(-1)
                    )
                else:
                    student_layer.bias.copy_(teacher_layer.bias.detach())
