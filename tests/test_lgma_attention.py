import math

import torch

from lgma.attention import LieGeneratedMetricAttention


def test_lgma_shape_for_generator_types():
    torch.manual_seed(0)
    for generator_type in ("full", "diagonal", "symmetric"):
        layer = LieGeneratedMetricAttention(
            d_model=64,
            num_heads=8,
            head_dim=8,
            num_generators=4,
            generator_type=generator_type,
            use_sdpa=False,
        )
        x = torch.randn(2, 16, 64)
        y = layer(x)
        assert y.shape == (2, 16, 64)


def test_lgma_attention_weights_shape():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(64, 8, 8, 4, generator_type="full")
    x = torch.randn(2, 16, 64)
    y, attn = layer(x, need_weights=True)
    assert y.shape == (2, 16, 64)
    assert attn.shape == (2, 8, 16, 16)


def test_lgma_base_dim_and_value_dim_shape_backward():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        32,
        num_heads=3,
        head_dim=8,
        base_dim=12,
        value_dim=5,
        num_generators=2,
        generator_type="full",
        use_sdpa=False,
    )
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 7, 32)
    y.pow(2).mean().backward()
    assert layer.q_proj.weight.grad is not None
    assert layer.k_proj.weight.grad is not None
    assert layer.v_proj.weight.grad is not None
    assert torch.isfinite(layer.q_proj.weight.grad).all()


def test_lgma_multibase_shape_and_projection_widths():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        32,
        num_heads=8,
        head_dim=4,
        base_dim=6,
        value_dim=5,
        num_generators=3,
        num_base_heads=2,
        value_transform="diag",
        use_sdpa=False,
    )
    assert layer.generated_heads_per_base == 4
    assert layer.q_proj.weight.shape == (12, 32)
    assert layer.k_proj.weight.shape == (12, 32)
    assert layer.v_proj.weight.shape == (10, 32)
    assert layer.value_scale.shape == (8, 5)
    x = torch.randn(2, 7, 32)
    y, attn = layer(x, need_weights=True)
    assert y.shape == (2, 7, 32)
    assert attn.shape == (2, 8, 7, 7)


def test_lgma_multibase_requires_heads_divisible_by_bases():
    try:
        LieGeneratedMetricAttention(
            32,
            num_heads=7,
            head_dim=4,
            num_generators=2,
            num_base_heads=2,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_lgma_backward_has_finite_gradients():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(64, 8, 8, 4, generator_type="full")
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in layer.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_metric_relation_matches_original_space_bilinear_form():
    torch.manual_seed(0)
    d_model = 5
    d_head = 3
    wq = torch.randn(d_head, d_model)
    wk = torch.randn(d_head, d_model)
    metric = torch.randn(d_head, d_head)
    xi = torch.randn(d_model)
    xj = torch.randn(d_model)

    q = wq @ xi
    k = wk @ xj
    left = q @ metric @ k
    bilinear = wq.T @ metric @ wk
    right = xi @ bilinear @ xj

    assert torch.allclose(left, right, atol=1e-5)


def test_identity_collapse_gives_identical_attention_maps():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        num_generators=2,
        dropout=0.0,
        generator_type="full",
        use_sdpa=False,
    )
    layer.generators.data.zero_()
    x = torch.randn(2, 8, 32)
    _, attn = layer(x, need_weights=True)
    for head in range(1, layer.num_heads):
        assert torch.allclose(attn[:, 0], attn[:, head], atol=1e-6)


def test_pre_cap_symmetric_norms_identify_clipped_heads():
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=1,
        head_generator_symmetric_cap=1.0,
        stabilize_generators=False,
        use_sdpa=False,
    )
    layer.generators.data.copy_(2.0 * torch.eye(4).unsqueeze(0))

    pre_cap_norms = layer.pre_cap_head_generator_symmetric_norms()
    capped = layer.compute_head_generators()
    capped_symmetric_norms = (
        0.5 * (capped + capped.transpose(-1, -2))
    ).float().norm(dim=(-2, -1))

    assert pre_cap_norms is not None
    assert torch.all(pre_cap_norms > 1.0)
    assert torch.allclose(capped_symmetric_norms, torch.ones(2), atol=1e-6)


def test_residual_metric_mode_returns_identity_plus_delta():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=3,
        metric_mode="residual",
        metric_beta=0.25,
        use_sdpa=False,
    )
    generators = layer._dense_generators()
    delta = torch.einsum("hm,mde->hde", layer.metric_theta_weights(), generators)
    expected = torch.eye(4)[None, :, :] + 0.25 * delta
    assert torch.allclose(layer.compute_metrics(), expected, atol=1e-6)


def test_quadratic_metric_mode_returns_second_order_expansion():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=3,
        metric_mode="quadratic",
        metric_beta=0.25,
        use_sdpa=False,
    )
    generators = layer._dense_generators()
    delta = 0.25 * torch.einsum("hm,mde->hde", layer.metric_theta_weights(), generators)
    expected = torch.eye(4)[None, :, :] + delta + 0.5 * torch.matmul(delta, delta)
    assert torch.allclose(layer.compute_metrics(), expected, atol=1e-6)


def test_exp_metric_mode_matches_batched_matrix_exp():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=3,
        head_dim=4,
        num_generators=2,
        metric_mode="exp",
        stabilize_generators=False,
        use_sdpa=False,
    )
    head_generators = layer.compute_head_generators()
    expected = torch.linalg.matrix_exp(head_generators.float()).to(dtype=head_generators.dtype)
    assert torch.allclose(layer.compute_metrics(), expected, atol=1e-6)


def test_quadratic_metric_is_closer_to_exp_than_residual_for_small_generators():
    torch.manual_seed(0)
    residual = LieGeneratedMetricAttention(
        16,
        num_heads=3,
        head_dim=4,
        num_generators=2,
        metric_mode="residual",
        metric_beta=0.5,
        stabilize_generators=False,
        use_sdpa=False,
    )
    quadratic = LieGeneratedMetricAttention(
        16,
        num_heads=3,
        head_dim=4,
        num_generators=2,
        metric_mode="quadratic",
        metric_beta=0.5,
        stabilize_generators=False,
        use_sdpa=False,
    )
    quadratic.load_state_dict(residual.state_dict())

    exact = torch.linalg.matrix_exp(residual.compute_head_generators().float())
    residual_error = (residual.compute_metrics().float() - exact).norm()
    quadratic_error = (quadratic.compute_metrics().float() - exact).norm()
    assert quadratic_error < residual_error


def test_unconstrained_metric_mode_learns_dense_per_head_metrics():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=3,
        head_dim=4,
        num_generators=1,
        metric_mode="unconstrained",
        metric_beta=0.5,
        use_sdpa=False,
    )
    assert hasattr(layer, "raw_metrics")
    assert layer.raw_metrics.shape == (3, 4, 4)
    assert layer.compute_metrics().shape == (3, 4, 4)
    assert not hasattr(layer, "theta")


def test_rms_metric_scaling_preserves_identity_scale():
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=2,
        logit_scale_mode="rms_metric",
        use_sdpa=False,
    )
    layer.generators.data.zero_()
    scale = layer._score_scale()
    assert torch.allclose(scale, torch.full_like(scale, 2.0), atol=1e-6)


def test_rms_metric_forward_reuses_computed_metrics():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=2,
        metric_mode="exp",
        logit_scale_mode="rms_metric",
        use_sdpa=False,
    )
    original_compute_metrics = layer.compute_metrics
    calls = 0

    def counted_compute_metrics():
        nonlocal calls
        calls += 1
        return original_compute_metrics()

    layer.compute_metrics = counted_compute_metrics
    x = torch.randn(2, 5, 16)
    layer(x)
    assert calls == 1


def test_diagonal_value_transform_changes_values_not_attention_weights():
    torch.manual_seed(0)
    base = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=2,
        dropout=0.0,
        use_sdpa=False,
    )
    transformed = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=2,
        dropout=0.0,
        value_transform="diag",
        use_sdpa=False,
    )
    transformed.load_state_dict(
        {**base.state_dict(), "value_scale": torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])},
        strict=False,
    )
    x = torch.randn(2, 5, 16)
    y_base, attn_base = base(x, need_weights=True)
    y_transformed, attn_transformed = transformed(x, need_weights=True)
    assert torch.allclose(attn_base, attn_transformed, atol=1e-6)
    assert not torch.allclose(y_base, y_transformed)


def test_lie_value_transform_identity_matches_no_transform():
    torch.manual_seed(0)
    base = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=2,
        dropout=0.0,
        use_sdpa=False,
    )
    transformed = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=2,
        dropout=0.0,
        value_transform="lie_exp",
        use_sdpa=False,
    )
    transformed.load_state_dict(base.state_dict(), strict=False)
    transformed.value_generators.data.zero_()
    x = torch.randn(2, 5, 16)
    y_base, attn_base = base(x, need_weights=True)
    y_transformed, attn_transformed = transformed(x, need_weights=True)
    assert torch.allclose(attn_base, attn_transformed, atol=1e-6)
    assert torch.allclose(y_base, y_transformed, atol=1e-6)


def test_lie_value_transform_changes_values_not_attention_weights():
    torch.manual_seed(0)
    base = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=1,
        dropout=0.0,
        stabilize_generators=False,
        use_sdpa=False,
    )
    transformed = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=1,
        dropout=0.0,
        stabilize_generators=False,
        value_transform="lie_residual",
        use_sdpa=False,
    )
    transformed.load_state_dict(base.state_dict(), strict=False)
    transformed.value_theta.data.fill_(1.0)
    transformed.value_generators.data.zero_()
    transformed.value_generators.data[0] = 0.5 * torch.eye(4)
    x = torch.randn(2, 5, 16)
    y_base, attn_base = base(x, need_weights=True)
    y_transformed, attn_transformed = transformed(x, need_weights=True)
    assert torch.allclose(attn_base, attn_transformed, atol=1e-6)
    assert not torch.allclose(y_base, y_transformed)


def test_quadratic_lie_value_transform_returns_second_order_expansion():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        value_dim=3,
        num_generators=2,
        metric_beta=0.25,
        value_transform="lie_quadratic",
        use_sdpa=False,
    )
    generators = layer._dense_value_generators()
    delta = 0.25 * torch.einsum("hm,mde->hde", layer.value_theta_weights(), generators)
    expected = torch.eye(3)[None, :, :] + delta + 0.5 * torch.matmul(delta, delta)
    assert torch.allclose(layer.compute_value_transforms(), expected, atol=1e-6)


def test_value_beta_scales_value_lie_separately_from_metric_beta():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=1,
        head_dim=3,
        base_dim=3,
        value_dim=3,
        num_generators=1,
        generator_type="diagonal",
        metric_mode="residual",
        metric_beta=1.0,
        value_beta=0.25,
        value_transform="lie_residual",
        normalize_generators=True,
        use_sdpa=False,
    )
    layer.theta.data.fill_(1.0)
    layer.generators.data.fill_(2.0)
    layer.value_theta.data.fill_(1.0)
    layer.value_generators.data.fill_(4.0)

    metric_diagonal = torch.einsum(
        "hm,md->hd",
        layer.metric_theta_weights(),
        layer._normalize_generators(layer.generators),
    )
    value_diagonal = torch.einsum(
        "hm,md->hd",
        layer.value_theta_weights(),
        layer._normalize_generators(layer.value_generators),
    )
    expected_metric = torch.diag_embed(1.0 + layer.metric_beta * metric_diagonal)
    expected_value_transform = torch.diag_embed(1.0 + layer.value_beta * value_diagonal)
    assert torch.allclose(layer.compute_metrics(), expected_metric)
    assert torch.allclose(layer.compute_value_transforms(), expected_value_transform)


def test_lie_value_transform_follows_metric_mode():
    for metric_mode, expected_mode in (
        ("exp", "exp"),
        ("residual", "residual"),
        ("quadratic", "quadratic"),
        ("unconstrained", "unconstrained"),
    ):
        layer = LieGeneratedMetricAttention(
            16,
            num_heads=2,
            head_dim=4,
            num_generators=2,
            metric_mode=metric_mode,
            value_transform="lie",
            use_sdpa=False,
        )
        assert layer.value_transform_mode == expected_mode
    assert hasattr(layer, "raw_value_transforms")


def test_circle_theta_init_produces_distinct_head_coordinates():
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=4,
        head_dim=4,
        num_generators=2,
        theta_init="circle",
        theta_init_scale=0.5,
    )
    norms = layer.theta.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.5), atol=1e-6)
    assert torch.unique(layer.theta.round(decimals=4), dim=0).shape[0] == 4


def test_causal_mask_zeros_future_probabilities():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        num_generators=2,
        dropout=0.0,
        causal=True,
        use_sdpa=False,
    )
    x = torch.randn(2, 8, 32)
    _, attn = layer(x, need_weights=True)
    future_mask = torch.ones(8, 8, dtype=torch.bool).triu(1)
    future_values = attn[:, :, future_mask]
    assert torch.allclose(future_values, torch.zeros_like(future_values), atol=1e-7)


def test_key_padding_mask_zeros_padded_key_probabilities():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        num_generators=2,
        dropout=0.0,
        use_sdpa=False,
    )
    x = torch.randn(2, 8, 32)
    key_padding_mask = torch.zeros(2, 8, dtype=torch.bool)
    key_padding_mask[:, -2:] = True
    _, attn = layer(x, key_padding_mask=key_padding_mask, need_weights=True)
    assert torch.allclose(attn[..., -2:], torch.zeros_like(attn[..., -2:]), atol=1e-7)


def test_sdpa_matches_explicit_path_without_dropout_or_masks():
    torch.manual_seed(0)
    explicit = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        num_generators=2,
        dropout=0.0,
        use_sdpa=False,
    )
    sdpa = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        num_generators=2,
        dropout=0.0,
        use_sdpa=True,
    )
    sdpa.load_state_dict(explicit.state_dict())
    explicit.eval()
    sdpa.eval()
    x = torch.randn(2, 8, 32)
    y_explicit = explicit(x)
    y_sdpa = sdpa(x)
    assert torch.allclose(y_explicit, y_sdpa, atol=1e-5)


def test_multibase_sdpa_matches_explicit_path_without_dropout_or_masks():
    torch.manual_seed(0)
    explicit = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        base_dim=8,
        value_dim=8,
        num_generators=2,
        num_base_heads=2,
        dropout=0.0,
        use_sdpa=False,
        logit_scale_mode="rms_metric",
        learn_head_temperature=True,
        value_transform="diag",
    )
    sdpa = LieGeneratedMetricAttention(
        32,
        num_heads=4,
        head_dim=8,
        base_dim=8,
        value_dim=8,
        num_generators=2,
        num_base_heads=2,
        dropout=0.0,
        use_sdpa=True,
        logit_scale_mode="rms_metric",
        learn_head_temperature=True,
        value_transform="diag",
    )
    sdpa.load_state_dict(explicit.state_dict())
    explicit.eval()
    sdpa.eval()
    x = torch.randn(2, 8, 32)
    y_explicit = explicit(x)
    y_sdpa = sdpa(x)
    assert torch.allclose(y_explicit, y_sdpa, atol=1e-5)


def test_diagonal_fast_path_matches_dense_diagonal_metric_scores():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=3,
        head_dim=4,
        num_generators=2,
        dropout=0.0,
        generator_type="diagonal",
        use_sdpa=False,
    )
    x = torch.randn(2, 5, 16)
    q, k, _ = layer._project(x)
    q_fast = layer._apply_metric_to_queries(q)
    metrics = layer.compute_metrics()
    q_dense = torch.einsum("btd,hde->bhte", q, metrics)
    scores_fast = torch.einsum("bhtd,bsd->bhts", q_fast, k)
    scores_dense = torch.einsum("bhtd,bsd->bhts", q_dense, k)
    assert torch.allclose(scores_fast, scores_dense, atol=1e-6)


def test_theta_weights_are_simplex_normalized():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=2,
        head_dim=2,
        num_generators=3,
        generator_type="full",
        use_sdpa=False,
    )
    layer.theta.data = torch.tensor([[2.0, 0.0, -2.0], [-1.0, 1.0, 3.0]])

    weights = layer.metric_theta_weights()
    assert torch.all(weights >= 0.0)
    assert torch.all(weights <= 1.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_compute_head_generators_returns_beta_scaled_softmax_mixture():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=2,
        head_dim=2,
        num_generators=2,
        generator_type="full",
        stabilize_generators=False,
        metric_beta=0.5,
        use_sdpa=False,
    )
    layer.theta.data = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    layer.generators.data.zero_()
    layer.generators.data[0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    layer.generators.data[1] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    weights = layer.metric_theta_weights()
    expected = 0.5 * torch.einsum("hm,mde->hde", weights, layer._dense_generators())
    assert torch.allclose(layer.compute_head_generators(), expected)


def test_dense_generators_are_frobenius_normalized():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=2,
        head_dim=2,
        num_generators=2,
        generator_type="full",
        stabilize_generators=False,
        normalize_generators=True,
        use_sdpa=False,
    )
    layer.generators.data[0] = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    layer.generators.data[1] = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    generators = layer._dense_generators()
    norms = generators.float().reshape(generators.shape[0], -1).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_diagonal_generators_are_l2_normalized_before_dense_embedding():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=1,
        head_dim=3,
        num_generators=1,
        generator_type="diagonal",
        normalize_generators=True,
        use_sdpa=False,
    )
    layer.generators.data[0] = torch.tensor([3.0, 4.0, 0.0])

    generators = layer._dense_generators()
    assert torch.allclose(generators[0].diagonal(), torch.tensor([0.6, 0.8, 0.0]))
    assert torch.allclose(generators.reshape(1, -1).norm(dim=-1), torch.ones(1))


def test_generator_normalization_is_disabled_by_default():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=1,
        head_dim=2,
        num_generators=1,
        generator_type="full",
        stabilize_generators=False,
        use_sdpa=False,
    )
    layer.generators.data[0] = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    assert not layer.normalize_generators
    assert torch.equal(layer._dense_generators(), layer.generators)


def test_metric_singular_value_clipping_bounds_metrics():
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=1,
        head_dim=2,
        num_generators=1,
        generator_type="diagonal",
        metric_clip_min=0.25,
        metric_clip_max=4.0,
        use_sdpa=False,
    )
    layer.theta.data.fill_(1.0)
    layer.generators.data[0] = torch.tensor([3.0, -3.0])

    singular_values = torch.linalg.svdvals(layer.compute_metrics())
    assert torch.all(singular_values >= 0.25 - 1e-6)
    assert torch.all(singular_values <= 4.0 + 1e-6)


def test_head_generator_symmetric_cap_bounds_exp_metric_without_svd():
    cap = math.log(4.0)
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=1,
        head_dim=2,
        num_generators=1,
        generator_type="full",
        stabilize_generators=False,
        head_generator_symmetric_cap=cap,
        use_sdpa=False,
    )
    layer.theta.data.zero_()
    layer.generators.data[0] = torch.tensor([[8.0, 20.0], [-12.0, -6.0]])

    head_generator = layer.compute_head_generators()[0]
    symmetric = 0.5 * (head_generator + head_generator.T)
    assert symmetric.norm() <= cap + 1e-6

    metric = layer.compute_metrics()[0]
    singular_values = torch.linalg.svdvals(metric)
    assert singular_values.min() >= 0.25 - 1e-5
    assert singular_values.max() <= 4.0 + 1e-5
    assert torch.linalg.det(metric) > 0.0


def test_head_generator_symmetric_cap_applies_to_diagonal_fast_path():
    cap = math.log(4.0)
    layer = LieGeneratedMetricAttention(
        8,
        num_heads=1,
        head_dim=3,
        num_generators=1,
        generator_type="diagonal",
        head_generator_symmetric_cap=cap,
        use_sdpa=False,
    )
    layer.theta.data.zero_()
    layer.generators.data[0] = torch.tensor([10.0, -10.0, 5.0])

    assert layer.compute_head_generators()[0].norm() <= cap + 1e-6
    singular_values = torch.linalg.svdvals(layer.compute_metrics())
    assert singular_values.min() >= 0.25 - 1e-5
    assert singular_values.max() <= 4.0 + 1e-5


def test_diagonal_metric_clipping_path_matches_dense_scores():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=3,
        head_dim=4,
        num_generators=2,
        dropout=0.0,
        generator_type="diagonal",
        metric_clip_min=0.25,
        metric_clip_max=4.0,
        use_sdpa=False,
    )
    layer.theta.data.fill_(1.0)
    layer.generators.data.fill_(2.0)
    x = torch.randn(2, 5, 16)
    q, k, _ = layer._project(x)
    q_clipped = layer._apply_metric_to_queries(q)
    metrics = layer.compute_metrics()
    q_dense = torch.einsum("btd,hde->bhte", q, metrics)
    scores_clipped = torch.einsum("bhtd,bsd->bhts", q_clipped, k)
    scores_dense = torch.einsum("bhtd,bsd->bhts", q_dense, k)
    assert torch.allclose(scores_clipped, scores_dense, atol=1e-5)


def test_stabilized_generators_are_trace_zero():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=2,
        head_dim=4,
        num_generators=3,
        generator_type="full",
        stabilize_generators=True,
    )
    traces = layer.effective_generators().diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(traces, torch.zeros_like(traces), atol=1e-7)


def test_theta_init_scale_sets_head_coordinate_norms():
    torch.manual_seed(0)
    layer = LieGeneratedMetricAttention(
        16,
        num_heads=4,
        head_dim=4,
        num_generators=3,
        theta_init_scale=0.25,
    )
    norms = layer.theta.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.25), atol=1e-6)


def test_generator_init_scale_changes_generator_std():
    torch.manual_seed(0)
    small = LieGeneratedMetricAttention(
        16,
        num_heads=4,
        head_dim=4,
        num_generators=32,
        generator_init_scale=0.01,
    )
    torch.manual_seed(0)
    large = LieGeneratedMetricAttention(
        16,
        num_heads=4,
        head_dim=4,
        num_generators=32,
        generator_init_scale=0.2,
    )
    assert large.generators.std() > small.generators.std()
