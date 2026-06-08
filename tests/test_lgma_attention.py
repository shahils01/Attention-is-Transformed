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
