import pytest
import torch
import torch.nn.functional as F

from lgma.baselines import (
    CollaborativeAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    StandardMultiheadAttention,
)


@pytest.mark.skipif(
    not hasattr(F, "scaled_dot_product_attention"),
    reason="PyTorch SDPA is unavailable",
)
@pytest.mark.parametrize(
    "layer",
    [
        StandardMultiheadAttention(32, num_heads=4, head_dim=8, causal=True),
        CollaborativeAttention(32, num_heads=4, head_dim=8, causal=True),
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


def test_collaborative_attention_matches_direct_mixing_vector_equation():
    torch.manual_seed(0)
    layer = CollaborativeAttention(
        d_model=6,
        num_heads=2,
        head_dim=3,
        base_dim=3,
        value_dim=2,
        bias=False,
    )
    with torch.no_grad():
        layer.mixing_vector.copy_(torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 3.0]]))

    x = torch.randn(2, 4, 6)
    output, attention = layer(x, need_weights=True)

    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x).view(2, 4, 2, 2).transpose(1, 2)
    scores = torch.einsum("btd,hd,bsd->bhts", q, layer.mixing_vector, k) / (3**0.5)
    expected_attention = torch.softmax(scores, dim=-1)
    expected_heads = torch.einsum("bhts,bhsv->bhtv", expected_attention, v)
    expected_output = layer.out_proj(expected_heads.transpose(1, 2).reshape(2, 4, 4))

    assert torch.allclose(attention, expected_attention, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-6)
    assert torch.equal(layer.compute_metrics(), torch.diag_embed(layer.mixing_vector))
