from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import (
    CollaborativeAttention,
    GroupedQueryAttention,
    ReducedDimMultiheadAttention,
    SharedIdentityAttention,
    StandardMultiheadAttention,
)
from lgma.transformer import validate_paper_gt_mha_module


VisionAttentionType = Literal[
    "mha",
    "reduced_mha",
    "gqa",
    "collaborative",
    "shared_identity",
    "gt_mha_exact",
    "gt_mha_residual",
    "gt_mha_quadratic",
]
VISION_ATTENTION_TYPES = {
    "mha",
    "reduced_mha",
    "gqa",
    "collaborative",
    "shared_identity",
    "gt_mha_exact",
    "gt_mha_residual",
    "gt_mha_quadratic",
}


@dataclass(frozen=True)
class DeiTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attention_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    attention_type: VisionAttentionType = "mha"
    reduced_qk_dim: int = 384
    collaborative_qk_dim: int = 384
    num_kv_heads: int = 4
    num_base_heads: int = 4
    num_value_base_heads: int | None = None
    num_generators: int = 8
    generator_mixing: str = "softmax"
    theta_init: str = "balanced_simplex"
    theta_init_scale: float = 4.0
    generator_init_scale: float = 0.02
    use_sdpa: bool = True
    fuse_base_qkv: bool = True
    fold_value_transform_into_output: bool = True
    sdpa_gqa_mode: str = "auto"

    def __post_init__(self) -> None:
        if self.attention_type not in VISION_ATTENTION_TYPES:
            raise ValueError(f"unsupported vision attention type: {self.attention_type}")
        if min(
            self.image_size,
            self.patch_size,
            self.num_classes,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.reduced_qk_dim,
            self.collaborative_qk_dim,
            self.num_kv_heads,
            self.num_base_heads,
            self.num_generators,
        ) <= 0:
            raise ValueError("DeiT dimensions and attention counts must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.reduced_qk_dim % self.num_heads:
            raise ValueError("reduced_qk_dim must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_heads % self.num_base_heads:
            raise ValueError("num_heads must be divisible by num_base_heads")
        if self.num_value_base_heads is not None:
            if self.num_value_base_heads <= 0:
                raise ValueError("num_value_base_heads must be positive")
            if self.num_heads % self.num_value_base_heads:
                raise ValueError("num_heads must be divisible by num_value_base_heads")
        if self.theta_init not in {"balanced_simplex", "random_sphere", "circle"}:
            raise ValueError(f"unsupported theta_init: {self.theta_init}")
        if self.theta_init_scale < 0 or self.generator_init_scale <= 0:
            raise ValueError(
                "theta_init_scale must be non-negative and generator_init_scale positive"
            )
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError("drop_rate must be in [0, 1)")
        if not 0.0 <= self.attention_drop_rate < 1.0:
            raise ValueError("attention_drop_rate must be in [0, 1)")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask.div_(keep)


class PatchEmbed(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} images, got {tuple(x.shape[-2:])}"
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.activation(self.fc1(x)))))


def build_vision_attention(config: DeiTConfig) -> nn.Module:
    head_dim = config.embed_dim // config.num_heads
    common = {
        "d_model": config.embed_dim,
        "num_heads": config.num_heads,
        "dropout": config.attention_drop_rate,
        "bias": config.qkv_bias,
        "causal": False,
    }
    if config.attention_type == "mha":
        return StandardMultiheadAttention(head_dim=head_dim, **common)
    if config.attention_type == "reduced_mha":
        return ReducedDimMultiheadAttention(
            qk_head_dim=config.reduced_qk_dim // config.num_heads,
            value_head_dim=head_dim,
            **common,
        )
    if config.attention_type == "gqa":
        return GroupedQueryAttention(
            head_dim=head_dim,
            num_kv_heads=config.num_kv_heads,
            **common,
        )
    if config.attention_type == "collaborative":
        return CollaborativeAttention(
            head_dim=head_dim,
            base_dim=config.collaborative_qk_dim,
            value_dim=head_dim,
            **common,
        )
    if config.attention_type == "shared_identity":
        return SharedIdentityAttention(head_dim=head_dim, **common)

    mode = {
        "gt_mha_exact": "exp",
        "gt_mha_residual": "residual",
        "gt_mha_quadratic": "quadratic",
    }[config.attention_type]
    attention = LieGeneratedMetricAttention(
        d_model=config.embed_dim,
        num_heads=config.num_heads,
        head_dim=head_dim,
        num_generators=config.num_generators,
        dropout=config.attention_drop_rate,
        bias=config.qkv_bias,
        generator_type="full",
        generator_mixing=config.generator_mixing,
        use_sdpa=config.use_sdpa,
        causal=False,
        stabilize_generators=False,
        normalize_generators=False,
        base_dim=head_dim,
        value_dim=head_dim,
        metric_mode=mode,
        theta_init=config.theta_init,
        theta_init_scale=config.theta_init_scale,
        generator_init_scale=config.generator_init_scale,
        logit_scale_mode="sqrt_dim",
        learn_head_temperature=False,
        value_transform="lie",
        num_base_heads=config.num_base_heads,
        num_value_base_heads=config.num_value_base_heads,
        fuse_base_qkv=config.fuse_base_qkv,
        fold_value_transform_into_output=config.fold_value_transform_into_output,
        sdpa_gqa_mode=config.sdpa_gqa_mode,
    )
    if (
        config.num_base_heads == 4
        and config.num_value_base_heads in (None, config.num_base_heads)
        and config.num_generators == 8
        and config.generator_mixing == "softmax"
    ):
        validate_paper_gt_mha_module(attention)
    return attention


class DeiTBlock(nn.Module):
    def __init__(self, config: DeiTConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = build_vision_attention(config)
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp = Mlp(
            config.embed_dim,
            int(config.embed_dim * config.mlp_ratio),
            config.drop_rate,
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class DeiTClassifier(nn.Module):
    """Non-distilled DeiT classifier with a configurable attention layer."""

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.patch_embed = PatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, config.embed_dim)
        )
        self.pos_drop = nn.Dropout(config.drop_rate)
        drop_paths = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.ModuleList(
            [DeiTBlock(config, probability) for probability in drop_paths]
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(
            self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.shape[0], -1)
        )
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def no_weight_decay(self) -> set[str]:
        return {"cls_token", "pos_embed"}

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls_token, x), dim=1) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))

    def configuration(self) -> dict[str, object]:
        return asdict(self.config)


def deit_base_patch16_224(
    attention_type: VisionAttentionType = "mha", **overrides: object
) -> DeiTClassifier:
    return DeiTClassifier(DeiTConfig(attention_type=attention_type, **overrides))


def vision_parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    attention = sum(
        parameter.numel()
        for block in getattr(model, "blocks", ())
        for parameter in block.attn.parameters()
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "attention_parameters": attention,
        "non_attention_parameters": total - attention,
    }
