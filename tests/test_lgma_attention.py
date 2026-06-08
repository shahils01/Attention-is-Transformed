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
    layer.theta.data.zero_()
    x = torch.randn(2, 8, 32)
    _, attn = layer(x, need_weights=True)
    for head in range(1, layer.num_heads):
        assert torch.allclose(attn[:, 0], attn[:, head], atol=1e-6)


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
    delta = torch.einsum("hm,mde->hde", layer.theta, generators)
    expected = torch.eye(4)[None, :, :] + 0.25 * delta
    assert torch.allclose(layer.compute_metrics(), expected, atol=1e-6)


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
    layer.theta.data.zero_()
    scale = layer._score_scale()
    assert torch.allclose(scale, torch.full_like(scale, 2.0), atol=1e-6)


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
