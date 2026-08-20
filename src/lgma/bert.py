from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from lgma.attention import LieGeneratedMetricAttention, _negative_large
from lgma.transformer import validate_paper_gt_mha_module

BertAttentionType = Literal["mha", "gt_mha_exact", "gt_mha_residual", "gt_mha_quadratic"]
GT_MHA_BERT_ATTENTION_TYPES = {"gt_mha_exact", "gt_mha_residual", "gt_mha_quadratic"}


@dataclass(frozen=True)
class BertGtMhaConfig:
    hidden_size: int
    num_attention_heads: int
    attention_probs_dropout_prob: float = 0.1
    attention_type: str = "gt_mha_exact"
    num_base_heads: int = 4
    num_generators: int = 8
    generator_type: str = "full"
    generator_mixing: str = "softmax"
    use_sdpa: bool = False
    fuse_base_qkv: bool = False
    sdpa_gqa_mode: str = "auto"
    attention_bias: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_attention_heads <= 0:
            raise ValueError("hidden_size and num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_base_heads:
            raise ValueError("num_attention_heads must be divisible by num_base_heads")
        if self.attention_type not in GT_MHA_BERT_ATTENTION_TYPES:
            raise ValueError(f"unsupported BERT GT-MHA attention type: {self.attention_type}")


class BertGtMhaSelfAttention(nn.Module):
    """Drop-in ``BertSelfAttention`` using canonical paper GT-MHA."""

    def __init__(self, config: BertGtMhaConfig) -> None:
        super().__init__()
        self.gt_mha_config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        mode = {
            "gt_mha_exact": "exp",
            "gt_mha_residual": "residual",
            "gt_mha_quadratic": "quadratic",
        }[config.attention_type]
        self.gt_attention = LieGeneratedMetricAttention(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=self.attention_head_size,
            num_generators=config.num_generators,
            dropout=config.attention_probs_dropout_prob,
            bias=config.attention_bias,
            generator_type=config.generator_type,
            generator_mixing=config.generator_mixing,
            use_sdpa=config.use_sdpa,
            causal=False,
            stabilize_generators=False,
            normalize_generators=False,
            base_dim=self.attention_head_size,
            value_dim=self.attention_head_size,
            metric_mode=mode,
            theta_init="random_sphere",
            logit_scale_mode="sqrt_dim",
            learn_head_temperature=False,
            value_transform="lie",
            num_base_heads=config.num_base_heads,
            fuse_base_qkv=config.fuse_base_qkv,
            sdpa_gqa_mode=config.sdpa_gqa_mode,
        )
        # BERT's surrounding BertSelfOutput retains the official output projection.
        self.gt_attention.out_proj = nn.Identity()

    @property
    def query(self) -> nn.Linear:
        return self.gt_attention.q_proj

    @property
    def key(self) -> nn.Linear:
        return self.gt_attention.k_proj

    @property
    def value(self) -> nn.Linear:
        return self.gt_attention.v_proj

    def _mask(self, mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.to(hidden.device)
        if mask.ndim == 2:
            if mask.dtype == torch.bool or not mask.is_floating_point():
                additive = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill(
                    ~mask.bool(), _negative_large(hidden.dtype)
                )
                return additive[:, None, None, :]
            if mask.numel() == 0 or (mask.min() >= 0 and mask.max() <= 1):
                return ((1 - mask.to(hidden.dtype)) * _negative_large(hidden.dtype))[
                    :, None, None, :
                ]
        return mask.to(dtype=hidden.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, ...]:
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise ValueError("BertGtMhaSelfAttention supports encoder self-attention only")
        if past_key_value is not None:
            raise ValueError("BertGtMhaSelfAttention does not support decoder KV caching")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        additive_mask = self._mask(attention_mask, hidden_states)
        if head_mask is None:
            result = self.gt_attention(
                hidden_states,
                attn_mask=additive_mask,
                need_weights=output_attentions,
            )
            if output_attentions:
                context, probabilities = result
                return context, probabilities
            return (result,)

        # BERT head masks act on attention probabilities, so retain the
        # explicit path when one is supplied. The normal MLM path above can
        # use fused QKV projection and native SDPA/GQA.
        q, k, v = self.gt_attention._project(hidden_states)
        scores = self.gt_attention.compute_scores(q, k, apply_scale=True)
        scores = self.gt_attention._prepare_additive_mask(
            scores, additive_mask, None
        )
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = self.gt_attention.attn_dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask.to(device=probs.device, dtype=probs.dtype)
        context = torch.einsum("bhts,bhsd->bhtd", probs, self.gt_attention._expand_values(v))
        # Apply GT-MHA's per-head Value Lie transformations before returning to
        # BERT's surrounding self-output projection.  ``out_proj`` is Identity
        # in this adapter because BertSelfOutput already owns that projection.
        context = self.gt_attention._output_projection(context)
        return (context, probs) if output_attentions else (context,)


def _average_attention_heads(
    tensor: torch.Tensor, *, source_heads: int, target_heads: int
) -> torch.Tensor:
    if source_heads % target_heads or tensor.shape[0] % source_heads:
        raise ValueError("projection shape is incompatible with target base heads")
    head_dim = tensor.shape[0] // source_heads
    return tensor.reshape(
        target_heads, source_heads // target_heads, head_dim, *tensor.shape[1:]
    ).mean(dim=1).reshape(target_heads * head_dim, *tensor.shape[1:])


def initialize_gt_mha_from_bert_attention(
    student: BertGtMhaSelfAttention, teacher: nn.Module
) -> None:
    """Warm-start reduced GT-MHA projections by averaging teacher heads."""
    for name in ("query", "key", "value"):
        if not isinstance(getattr(teacher, name, None), nn.Linear):
            raise ValueError(f"teacher attention must expose a linear {name} projection")
    with torch.no_grad():
        for name in ("query", "key", "value"):
            source, target = getattr(teacher, name), getattr(student, name)
            kwargs = {
                "source_heads": student.num_attention_heads,
                "target_heads": student.gt_mha_config.num_base_heads,
            }
            target.weight.copy_(_average_attention_heads(source.weight.detach(), **kwargs))
            if target.bias is not None:
                target.bias.zero_() if source.bias is None else target.bias.copy_(
                    _average_attention_heads(source.bias.detach(), **kwargs)
                )


def _bert_encoder_layers(model: nn.Module) -> nn.ModuleList:
    for backbone in (getattr(model, "bert", None), getattr(model, "base_model", None), model):
        layers = getattr(getattr(backbone, "encoder", None), "layer", None)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise ValueError("expected a Hugging Face BERT model exposing bert.encoder.layer")


def bert_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return paper-facing BERT parameter counts without double-counting tensors."""
    layers = _bert_encoder_layers(model)
    self_attention = sum(
        parameter.numel()
        for layer in layers
        for parameter in layer.attention.self.parameters()
    )
    attention_output = sum(
        parameter.numel()
        for layer in layers
        for parameter in layer.attention.output.parameters()
    )
    return {
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "self_attention_parameter_count": self_attention,
        "attention_output_parameter_count": attention_output,
        "attention_block_parameter_count": self_attention + attention_output,
    }


def replace_bert_self_attention(
    model: nn.Module,
    *,
    attention_type: BertAttentionType,
    num_base_heads: int = 4,
    num_generators: int = 8,
    generator_mixing: str = "softmax",
    use_sdpa: bool = False,
    fuse_base_qkv: bool = False,
    sdpa_gqa_mode: str = "auto",
    initialize_from_mha: bool = True,
    enforce_paper_gt_mha: bool = True,
) -> list[dict[str, Any]]:
    if attention_type == "mha":
        return []
    if attention_type not in GT_MHA_BERT_ATTENTION_TYPES:
        raise ValueError(f"unsupported BERT attention type: {attention_type}")
    if enforce_paper_gt_mha and (num_base_heads != 4 or num_generators != 8):
        raise ValueError("paper GT-MHA requires num_base_heads=4 and num_generators=8")
    config = getattr(model, "config", None)
    if config is None or bool(getattr(config, "is_decoder", False)):
        raise ValueError("GT-MHA requires an encoder-only BERT config")
    audit = []
    for index, layer in enumerate(_bert_encoder_layers(model)):
        teacher = layer.attention.self
        teacher_params = sum(p.numel() for p in teacher.parameters())
        replacement = BertGtMhaSelfAttention(
            BertGtMhaConfig(
                hidden_size=int(config.hidden_size),
                num_attention_heads=int(config.num_attention_heads),
                attention_probs_dropout_prob=float(config.attention_probs_dropout_prob),
                attention_type=attention_type,
                num_base_heads=num_base_heads,
                num_generators=num_generators,
                generator_mixing=generator_mixing,
                use_sdpa=use_sdpa,
                fuse_base_qkv=fuse_base_qkv,
                sdpa_gqa_mode=sdpa_gqa_mode,
            )
        )
        if enforce_paper_gt_mha:
            validate_paper_gt_mha_module(replacement.gt_attention)
        parameter = next(teacher.parameters())
        replacement.to(device=parameter.device, dtype=parameter.dtype)
        if initialize_from_mha:
            initialize_gt_mha_from_bert_attention(replacement, teacher)
        replacement.train(teacher.training)
        layer.attention.self = replacement
        student_params = sum(p.numel() for p in replacement.parameters())
        audit.append(
            {
                "layer": index,
                "attention_type": attention_type,
                "initialization": "mean_teacher_heads" if initialize_from_mha else "random",
                "teacher_attention_parameters": teacher_params,
                "gt_mha_attention_parameters": student_params,
                "attention_parameter_reduction": teacher_params - student_params,
                "gt_mha_config": asdict(replacement.gt_mha_config),
            }
        )
    return audit


def load_bert_masked_lm(
    model_name_or_path: str,
    *,
    attention_type: BertAttentionType | None = None,
    initialization: Literal["checkpoint", "random"] = "checkpoint",
    num_base_heads: int = 4,
    num_generators: int = 8,
    generator_mixing: str = "softmax",
    use_sdpa: bool = False,
    fuse_base_qkv: bool = False,
    sdpa_gqa_mode: str = "auto",
    enforce_paper_gt_mha: bool = True,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    try:
        from transformers import AutoConfig, AutoModelForMaskedLM
    except ImportError as exc:
        raise ImportError("install BERT dependencies with `pip install -e '.[bert]'`") from exc
    path = Path(model_name_or_path)
    manifest_path, state_path = path / "bert_gt_mha_manifest.json", path / "gt_mha_state_dict.pt"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest:
        saved_type = manifest["attention_type"]
        if attention_type is not None and attention_type != saved_type:
            raise ValueError(f"requested {attention_type!r}, saved model uses {saved_type!r}")
        attention_type = saved_type
        num_base_heads, num_generators = manifest["num_base_heads"], manifest["num_generators"]
        generator_mixing = manifest.get("generator_mixing", "softmax")
        use_sdpa = bool(manifest.get("use_sdpa", use_sdpa))
        fuse_base_qkv = bool(manifest.get("fuse_base_qkv", fuse_base_qkv))
        sdpa_gqa_mode = manifest.get("sdpa_gqa_mode", sdpa_gqa_mode)
    attention_type = attention_type or "mha"
    if manifest and attention_type in GT_MHA_BERT_ATTENTION_TYPES:
        if not state_path.is_file():
            raise FileNotFoundError(f"GT-MHA state file is missing: {state_path}")
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=trust_remote_code)
        audit = replace_bert_self_attention(
            model, attention_type=attention_type, num_base_heads=int(num_base_heads),
            num_generators=int(num_generators), generator_mixing=generator_mixing,
            use_sdpa=use_sdpa, fuse_base_qkv=fuse_base_qkv,
            sdpa_gqa_mode=sdpa_gqa_mode,
            initialize_from_mha=False,
            enforce_paper_gt_mha=enforce_paper_gt_mha,
        )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        return model, audit
    if initialization == "checkpoint":
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    elif initialization == "random":
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=trust_remote_code)
    else:
        raise ValueError("initialization must be 'checkpoint' or 'random'")
    return model, replace_bert_self_attention(
        model, attention_type=attention_type, num_base_heads=num_base_heads,
        num_generators=num_generators, generator_mixing=generator_mixing,
        use_sdpa=use_sdpa, fuse_base_qkv=fuse_base_qkv,
        sdpa_gqa_mode=sdpa_gqa_mode,
        initialize_from_mha=initialization == "checkpoint",
        enforce_paper_gt_mha=enforce_paper_gt_mha,
    )


def load_bert_sequence_classifier(
    model_name_or_path: str,
    *,
    num_labels: int,
    attention_type: BertAttentionType = "mha",
    num_base_heads: int = 4,
    num_generators: int = 8,
    generator_mixing: str = "softmax",
    use_sdpa: bool = False,
    fuse_base_qkv: bool = False,
    sdpa_gqa_mode: str = "auto",
    enforce_paper_gt_mha: bool = True,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError("install BERT dependencies with `pip install -e '.[bert]'`") from exc
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels, trust_remote_code=trust_remote_code
    )
    return model, replace_bert_self_attention(
        model, attention_type=attention_type, num_base_heads=num_base_heads,
        num_generators=num_generators, generator_mixing=generator_mixing,
        use_sdpa=use_sdpa, fuse_base_qkv=fuse_base_qkv,
        sdpa_gqa_mode=sdpa_gqa_mode,
        initialize_from_mha=True,
        enforce_paper_gt_mha=enforce_paper_gt_mha,
    )
