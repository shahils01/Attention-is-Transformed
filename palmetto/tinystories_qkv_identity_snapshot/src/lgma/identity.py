"""QK and V identity ablation of the historical TinyStories residual model."""
import torch

from lgma.attention import LieGeneratedMetricAttention


class QKVIdentityAttention(LieGeneratedMetricAttention):
    def __init__(self, *args, **kwargs):
        if kwargs.get("metric_mode") != "residual":
            raise ValueError("QK/V identity must be constructed from the residual reference")
        super().__init__(*args, **kwargs)
        # Consume exactly the reference initialization RNG, preserving every
        # retained parameter and subsequent layer's initialization for this seed.
        # Keep zero buffers for existing diagnostics, never optimizer parameters.
        for name in ("generators", "theta", "value_generators", "value_theta"):
            value = torch.zeros_like(getattr(self, name))
            delattr(self, name)
            self.register_buffer(name, value)
        self.metric_mode = "identity"
        assert self.value_transform_mode == "residual"

    def compute_metrics(self):
        weight = self.q_proj.weight
        eye = torch.eye(self.base_dim, device=weight.device, dtype=weight.dtype)
        return eye.unsqueeze(0).expand(self.num_heads, -1, -1)

    def compute_head_generators(self):
        return self.q_proj.weight.new_zeros(self.num_heads, self.base_dim, self.base_dim)

    def compute_value_transforms(self):
        weight = self.v_proj.weight
        eye = torch.eye(self.value_dim, device=weight.device, dtype=weight.dtype)
        return eye.unsqueeze(0).expand(self.num_heads, -1, -1)
