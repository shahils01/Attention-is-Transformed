import pytest
import torch
import torch.nn.functional as F

from lgma.baselines import GroupedQueryAttention, MultiQueryAttention, StandardMultiheadAttention


@pytest.mark.skipif(
    not hasattr(F, "scaled_dot_product_attention"),
    reason="PyTorch SDPA is unavailable",
)
@pytest.mark.parametrize(
    "layer",
    [
        StandardMultiheadAttention(32, num_heads=4, head_dim=8, causal=True),
        GroupedQueryAttention(32, num_heads=4, head_dim=8, num_kv_heads=2, causal=True),
        MultiQueryAttention(32, num_heads=4, head_dim=8, causal=True),
    ],
)
def test_sdpa_baselines_match_explicit_attention(layer):
    torch.manual_seed(0)
    x = torch.randn(2, 7, 32)

    sdpa_output = layer(x)
    explicit_output, attention = layer(x, need_weights=True)

    assert attention.shape == (2, 4, 7, 7)
    assert torch.allclose(sdpa_output, explicit_output, atol=1e-5, rtol=1e-5)
