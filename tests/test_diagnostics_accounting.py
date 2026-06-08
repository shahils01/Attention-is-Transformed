import torch

from lgma.accounting import (
    attention_accounting,
    count_parameters,
    generator_parameter_count,
    kv_cache_bytes_per_token_per_layer,
    qkv_parameter_count,
)
from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import GroupedQueryAttention, StandardMultiheadAttention
from lgma.diagnostics import (
    attention_cosine_similarity,
    attention_entropy,
    metric_delta_cosine_similarity,
    metric_diversity_loss,
    metric_condition_number,
    metric_cosine_similarity,
)


def test_metric_cosine_similarity_is_square_symmetric_with_unit_diagonal():
    torch.manual_seed(0)
    metrics = torch.randn(4, 8, 8)
    sim = metric_cosine_similarity(metrics)
    assert sim.shape == (4, 4)
    assert torch.allclose(sim, sim.T, atol=1e-6)
    assert torch.allclose(sim.diag(), torch.ones(4), atol=1e-6)


def test_attention_cosine_similarity_is_square_symmetric_with_unit_diagonal():
    torch.manual_seed(0)
    attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    sim = attention_cosine_similarity(attn)
    assert sim.shape == (4, 4)
    assert torch.allclose(sim, sim.T, atol=1e-6)
    assert torch.allclose(sim.diag(), torch.ones(4), atol=1e-6)


def test_attention_entropy_is_finite_and_non_negative():
    torch.manual_seed(0)
    attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    entropy = attention_entropy(attn)
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_metric_condition_number_is_finite_for_identity_metrics():
    metrics = torch.eye(4).expand(3, 4, 4)
    cond = metric_condition_number(metrics)
    assert torch.allclose(cond, torch.ones_like(cond))


def test_metric_diversity_loss_uses_off_diagonal_terms():
    metrics = torch.stack(
        [
            torch.eye(2),
            torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
        ]
    )
    assert torch.allclose(metric_diversity_loss(metrics), torch.tensor(0.0))
    assert torch.allclose(metric_diversity_loss(metrics, squared=True), torch.tensor(0.0))


def test_metric_delta_similarity_removes_identity_component():
    identity = torch.eye(2)
    metrics = torch.stack(
        [
            identity + torch.tensor([[0.1, 0.0], [0.0, 0.0]]),
            identity + torch.tensor([[0.0, 0.0], [0.0, 0.1]]),
        ]
    )
    full_sim = metric_cosine_similarity(metrics)
    delta_sim = metric_delta_cosine_similarity(metrics)
    assert full_sim[0, 1] > 0.9
    assert torch.allclose(delta_sim[0, 1], torch.tensor(0.0))


def test_accounting_matches_hand_computed_lgma_counts():
    layer = LieGeneratedMetricAttention(
        d_model=16,
        num_heads=4,
        head_dim=4,
        num_generators=2,
        generator_type="full",
        bias=False,
    )
    assert qkv_parameter_count(layer) == 3 * 16 * 4
    assert generator_parameter_count(layer) == 2 * 4 * 4 + 4 * 2
    assert kv_cache_bytes_per_token_per_layer(layer, dtype=torch.float16) == 2 * 4 * 2
    assert count_parameters(layer) > qkv_parameter_count(layer)
    report = attention_accounting(layer, sequence_length=8, batch_size=2)
    assert report.attention_maps == 4
    assert report.attention_score_flops == 2 * 2 * 4 * 8 * 8 * 4


def test_kv_cache_accounting_distinguishes_mha_and_gqa():
    mha = StandardMultiheadAttention(16, num_heads=4, head_dim=4)
    gqa = GroupedQueryAttention(16, num_heads=4, head_dim=4, num_kv_heads=2)
    assert kv_cache_bytes_per_token_per_layer(mha, dtype=torch.float16) == 2 * 4 * 4 * 2
    assert kv_cache_bytes_per_token_per_layer(gqa, dtype=torch.float16) == 2 * 2 * 4 * 2
