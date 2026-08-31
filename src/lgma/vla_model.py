from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from lgma.transformer import TransformerBlock

VLAActionHeadType = Literal["mlp", "flow"]
VLAAttentionType = Literal[
    "mha",
    "collaborative",
    "shared_identity",
    "lgma",
    "lgma_multibase",
    "lgma_residual",
    "lgma_quad",
    "lgma_unconstrained",
    "lgma_value_diag",
    "lgma_multibase_value_diag",
]


@dataclass
class VLAPolicyConfig:
    image_size: int = 128
    num_views: int = 2
    vocab_size: int = 4096
    text_length: int = 32
    proprio_dim: int = 20
    action_dim: int = 20
    action_horizon: int = 10
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 32
    mlp_ratio: int = 4
    dropout: float = 0.0
    attention: VLAAttentionType = "mha"
    num_generators: int = 4
    generator_type: str = "full"
    generator_mixing: str = "softmax"
    theta_init_scale: float = 4.0
    generator_init_scale: float = 0.02
    stabilize_generators: bool = True
    normalize_generators: bool = False
    head_generator_symmetric_cap: float | None = None
    num_base_heads: int = 1
    base_dim: int | None = None
    value_dim: int | None = None
    value_beta: float | None = None
    value_transform: str = "none"
    action_head: VLAActionHeadType = "mlp"
    flow_hidden_mult: int = 2
    flow_sampling_steps: int = 10
    flow_noise_scale: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SimpleVisionEncoder(nn.Module):
    """Small image encoder that emits one token per camera view."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"images must be [B,V,C,H,W], got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        x = images.reshape(batch * views, channels, height, width)
        x = self.net(x).flatten(1)
        x = self.proj(x)
        return x.view(batch, views, -1)


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    if timestep.ndim != 1:
        raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)}")
    half = dim // 2
    if half == 0:
        return timestep[:, None]
    freq = torch.exp(
        -torch.arange(half, device=timestep.device, dtype=torch.float32)
        * (math.log(10000.0) / max(half - 1, 1))
    )
    angles = timestep.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(dtype=timestep.dtype)


class FlowActionHead(nn.Module):
    """Conditional flow-matching head over full action chunks."""

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.net = nn.Sequential(
            nn.Linear(d_model * 2 + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        context: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.shape[:2] != context.shape[:2]:
            raise ValueError(
                "noisy_action and context batch/horizon must match, got "
                f"{tuple(noisy_action.shape)} vs {tuple(context.shape)}"
            )
        time = timestep_embedding(timestep, context.shape[-1])
        time = self.time_mlp(time).unsqueeze(1).expand(-1, context.shape[1], -1)
        return self.net(torch.cat([context, noisy_action, time], dim=-1))


class VLATransformerPolicy(nn.Module):
    """Compact VLA imitation policy for controlled attention ablations."""

    def __init__(self, config: VLAPolicyConfig) -> None:
        super().__init__()
        self.config = config
        if config.attention == "lgma_multibase" and config.num_base_heads == 1:
            config.num_base_heads = 2
        if config.attention == "lgma_unconstrained" and config.generator_type == "diagonal":
            raise ValueError("lgma_unconstrained requires full or symmetric generators")

        self.vision = SimpleVisionEncoder(config.d_model)
        self.text_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.text_pos = nn.Parameter(torch.zeros(1, config.text_length, config.d_model))
        self.view_embedding = nn.Parameter(torch.zeros(1, config.num_views, config.d_model))
        self.proprio_proj = nn.Linear(config.proprio_dim, config.d_model)
        self.action_queries = nn.Parameter(torch.randn(1, config.action_horizon, config.d_model) * 0.02)
        self.type_embedding = nn.Embedding(4, config.d_model)
        max_tokens = config.text_length + config.num_views + 1 + config.action_horizon
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens, config.d_model))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    attention_type=config.attention,  # type: ignore[arg-type]
                    num_generators=config.num_generators,
                    generator_type=config.generator_type,
                    generator_mixing=config.generator_mixing,
                    dropout=config.dropout,
                    mlp_ratio=config.mlp_ratio,
                    causal=False,
                    theta_init_scale=config.theta_init_scale,
                    generator_init_scale=config.generator_init_scale,
                    stabilize_generators=config.stabilize_generators,
                    normalize_generators=config.normalize_generators,
                    head_generator_symmetric_cap=config.head_generator_symmetric_cap,
                    base_dim=config.base_dim,
                    value_dim=config.value_dim,
                    value_beta=config.value_beta,
                    value_transform=config.value_transform,
                    num_base_heads=config.num_base_heads,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        if config.action_head == "mlp":
            self.action_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.action_dim),
            )
        elif config.action_head == "flow":
            if config.flow_hidden_mult <= 0:
                raise ValueError("flow_hidden_mult must be positive")
            if config.flow_noise_scale <= 0.0:
                raise ValueError("flow_noise_scale must be positive")
            self.action_head = FlowActionHead(
                d_model=config.d_model,
                action_dim=config.action_dim,
                hidden_dim=config.d_model * config.flow_hidden_mult,
            )
        else:
            raise ValueError(f"Unsupported action_head `{config.action_head}`")
        nn.init.normal_(self.text_pos, std=0.02)
        nn.init.normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    @property
    def attention_modules(self) -> list[nn.Module]:
        return [block.attn for block in self.blocks]

    def encode_action_context(
        self,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        text_token_ids: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_input.size(0)
        if text_token_ids.shape != (batch, self.config.text_length):
            raise ValueError(
                "text_token_ids must be "
                f"[B,{self.config.text_length}], got {tuple(text_token_ids.shape)}"
            )
        if proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"proprio last dim must be {self.config.proprio_dim}, got {proprio.shape[-1]}"
            )

        text_tokens = self.text_embedding(text_token_ids) + self.text_pos
        text_tokens = text_tokens + self.type_embedding.weight[0].view(1, 1, -1)

        image_tokens = self.vision(image_input) + self.view_embedding[:, : image_input.size(1), :]
        image_tokens = image_tokens + self.type_embedding.weight[1].view(1, 1, -1)
        image_tokens = image_tokens * image_mask.to(image_tokens.dtype).unsqueeze(-1)

        proprio_token = self.proprio_proj(proprio).unsqueeze(1)
        proprio_token = proprio_token + self.type_embedding.weight[2].view(1, 1, -1)

        action_tokens = self.action_queries.expand(batch, -1, -1)
        action_tokens = action_tokens + self.type_embedding.weight[3].view(1, 1, -1)

        x = torch.cat([text_tokens, image_tokens, proprio_token, action_tokens], dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, -self.config.action_horizon :, :]

    def sample_actions_from_context(
        self,
        action_context: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        if self.config.action_head != "flow":
            return self.action_head(action_context)
        steps = int(steps or self.config.flow_sampling_steps)
        if steps <= 0:
            raise ValueError("flow sampling steps must be positive")
        action = torch.randn(
            action_context.shape[0],
            action_context.shape[1],
            self.config.action_dim,
            device=action_context.device,
            dtype=action_context.dtype,
        ) * self.config.flow_noise_scale
        dt = 1.0 / steps
        for idx in range(steps):
            timestep = torch.full(
                (action_context.shape[0],),
                idx / steps,
                device=action_context.device,
                dtype=action_context.dtype,
            )
            velocity = self.action_head(action_context, action, timestep)
            action = action + dt * velocity
        return action

    def forward(
        self,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        text_token_ids: torch.Tensor,
        proprio: torch.Tensor,
        noisy_action: torch.Tensor | None = None,
        flow_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_context = self.encode_action_context(
            image_input=image_input,
            image_mask=image_mask,
            text_token_ids=text_token_ids,
            proprio=proprio,
        )
        if self.config.action_head == "mlp":
            return self.action_head(action_context)
        if noisy_action is None:
            return self.sample_actions_from_context(action_context)
        if flow_timestep is None:
            raise ValueError("flow_timestep is required when noisy_action is provided")
        return self.action_head(action_context, noisy_action, flow_timestep)


def ee6d_loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shapes must match, got {pred.shape} vs {target.shape}")
    if pred.shape[-1] != 20:
        raise ValueError("EE6D loss expects 20D dual-arm padded actions")
    gripper_idx = (9, 19)
    pos_idx = (0, 1, 2, 10, 11, 12)
    rot_idx = (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)
    pos_loss = F.mse_loss(pred[..., pos_idx], target[..., pos_idx]) * 500.0
    rot_loss = F.mse_loss(pred[..., rot_idx], target[..., rot_idx]) * 10.0
    gripper_loss = F.binary_cross_entropy_with_logits(
        pred[..., gripper_idx],
        target[..., gripper_idx].clamp(0.0, 1.0),
    )
    return {
        "position_loss": pos_loss,
        "rotate6D_loss": rot_loss,
        "gripper_loss": gripper_loss,
    }


def ee6d_continuous_loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shapes must match, got {pred.shape} vs {target.shape}")
    if pred.shape[-1] != 20:
        raise ValueError("EE6D loss expects 20D dual-arm padded actions")
    gripper_idx = (9, 19)
    pos_idx = (0, 1, 2, 10, 11, 12)
    rot_idx = (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)
    pos_loss = F.mse_loss(pred[..., pos_idx], target[..., pos_idx]) * 500.0
    rot_loss = F.mse_loss(pred[..., rot_idx], target[..., rot_idx]) * 10.0
    gripper_loss = F.mse_loss(pred[..., gripper_idx], target[..., gripper_idx])
    return {
        "position_loss": pos_loss,
        "rotate6D_loss": rot_loss,
        "gripper_loss": gripper_loss,
    }
