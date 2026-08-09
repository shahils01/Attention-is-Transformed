import torch

from lgma.accounting import (
    attention_accounting,
    count_parameters,
    generator_parameter_count,
    k_parameter_count,
    kv_cache_bytes_per_token_per_layer,
    q_parameter_count,
    qkv_parameter_count,
    v_parameter_count,
)
from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import (
    CollaborativeAttention,
    GroupedQueryAttention,
    StandardMultiheadAttention,
)
from lgma.diagnostics import (
    attention_cosine_similarity,
    attention_entropy,
    centered_attention_cosine_similarity,
    grouped_similarity_stats,
    induced_bilinear_forms,
    induced_metric_cosine_similarity,
    metric_delta_cosine_similarity,
    metric_diversity_loss,
    metric_distance_from_identity,
    metric_condition_number,
    metric_cosine_similarity,
    score_cosine_similarity,
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


def test_centered_attention_and_score_similarity_are_square_symmetric():
    torch.manual_seed(0)
    attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    scores = torch.randn(2, 4, 8, 8)
    for sim in (centered_attention_cosine_similarity(attn), score_cosine_similarity(scores)):
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


def test_induced_bilinear_diagnostics_are_square_symmetric():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        d_model=8,
        num_heads=3,
        head_dim=4,
        num_generators=2,
    )
    forms = induced_bilinear_forms(layer)
    sim = induced_metric_cosine_similarity(layer)
    assert forms.shape == (3, 8, 8)
    assert sim.shape == (3, 3)
    assert torch.allclose(sim, sim.T, atol=1e-6)
    assert torch.allclose(sim.diag(), torch.ones(3), atol=1e-6)


def test_multibase_induced_bilinear_diagnostics_shape():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        d_model=8,
        num_heads=4,
        head_dim=4,
        base_dim=3,
        num_generators=2,
        num_base_heads=2,
    )
    forms = induced_bilinear_forms(layer)
    sim = induced_metric_cosine_similarity(layer)
    assert forms.shape == (4, 8, 8)
    assert sim.shape == (4, 4)


def test_grouped_similarity_stats_reports_within_and_across_base_means():
    similarity = torch.eye(4)
    similarity[0, 1] = similarity[1, 0] = 0.2
    similarity[2, 3] = similarity[3, 2] = 0.4
    similarity[:2, 2:] = 0.8
    similarity[2:, :2] = 0.8
    stats = grouped_similarity_stats(similarity, num_base_heads=2, generated_heads_per_base=2)
    assert torch.allclose(stats["within_base_offdiag_mean_cosine"], torch.tensor(0.3))
    assert torch.allclose(stats["across_base_mean_cosine"], torch.tensor(0.8))


def test_metric_distance_from_identity_is_zero_for_identity():
    metrics = torch.eye(4).expand(3, 4, 4)
    assert torch.allclose(metric_distance_from_identity(metrics), torch.zeros(3))


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
    assert q_parameter_count(layer) == 16 * 4
    assert k_parameter_count(layer) == 16 * 4
    assert v_parameter_count(layer) == 16 * 4
    assert generator_parameter_count(layer) == 2 * 4 * 4 + 4 * 2
    assert kv_cache_bytes_per_token_per_layer(layer, dtype=torch.float16) == 2 * 4 * 2
    assert count_parameters(layer) > qkv_parameter_count(layer)
    report = attention_accounting(layer, sequence_length=8, batch_size=2)
    assert report.attention_maps == 4
    assert report.attention_score_flops == 2 * 2 * 4 * 8 * 8 * 4


def test_lgma_accounting_uses_base_and_value_dimensions():
    layer = LieGeneratedMetricAttention(
        d_model=16,
        num_heads=4,
        head_dim=4,
        base_dim=6,
        value_dim=3,
        num_generators=2,
        generator_type="full",
        bias=False,
    )
    assert q_parameter_count(layer) == 16 * 6
    assert k_parameter_count(layer) == 16 * 6
    assert v_parameter_count(layer) == 16 * 3
    assert kv_cache_bytes_per_token_per_layer(layer, dtype=torch.float16) == (6 + 3) * 2
    report = attention_accounting(layer, sequence_length=8, batch_size=2)
    assert report.attention_score_flops == 2 * 2 * 4 * 8 * 8 * 6


def test_collaborative_attention_accounting_tracks_shared_keys_and_per_head_values():
    layer = CollaborativeAttention(
        d_model=16,
        num_heads=4,
        head_dim=4,
        base_dim=6,
        value_dim=3,
        bias=False,
    )
    assert q_parameter_count(layer) == 16 * 6
    assert k_parameter_count(layer) == 16 * 6
    assert v_parameter_count(layer) == 16 * 4 * 3
    assert generator_parameter_count(layer) == 4 * 6
    assert kv_cache_bytes_per_token_per_layer(
        layer, dtype=torch.float16
    ) == (6 + 4 * 3) * 2
    report = attention_accounting(layer, sequence_length=8, batch_size=2)
    assert report.base_heads == 1
    assert report.generated_heads_per_base == 4
    assert report.attention_score_flops == 2 * 2 * 4 * 8 * 8 * 6


def test_multibase_lgma_accounting_uses_base_count_for_cache():
    layer = LieGeneratedMetricAttention(
        d_model=16,
        num_heads=8,
        head_dim=4,
        base_dim=6,
        value_dim=3,
        num_generators=2,
        num_base_heads=2,
        generator_type="full",
        bias=False,
    )
    assert q_parameter_count(layer) == 16 * 2 * 6
    assert k_parameter_count(layer) == 16 * 2 * 6
    assert v_parameter_count(layer) == 16 * 2 * 3
    assert kv_cache_bytes_per_token_per_layer(layer, dtype=torch.float16) == 2 * (6 + 3) * 2
    report = attention_accounting(layer, sequence_length=8, batch_size=2)
    assert report.base_heads == 2
    assert report.generated_heads_per_base == 4
    assert report.attention_maps == 8
    assert report.attention_score_flops == 2 * 2 * 8 * 8 * 8 * 6


def test_kv_cache_accounting_distinguishes_mha_and_gqa():
    mha = StandardMultiheadAttention(16, num_heads=4, head_dim=4)
    gqa = GroupedQueryAttention(16, num_heads=4, head_dim=4, num_kv_heads=2)
    assert kv_cache_bytes_per_token_per_layer(mha, dtype=torch.float16) == 2 * 4 * 4 * 2
    assert kv_cache_bytes_per_token_per_layer(gqa, dtype=torch.float16) == 2 * 2 * 4 * 2
