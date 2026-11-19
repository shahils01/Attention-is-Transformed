from __future__ import annotations

import io
import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from lgma import attention as attention_module
from lgma.accounting import attention_accounting
from lgma.attention import LieGeneratedMetricAttention
from lgma.transformer import TinyTransformerLM, build_attention


def make_lu_attention(**overrides):
    settings = {
        "d_model": 13,
        "num_heads": 6,
        "head_dim": 4,
        "num_base_heads": 2,
        "num_generators": 0,
        "metric_mode": "lu",
        "value_transform": "none",
    }
    settings.update(overrides)
    return LieGeneratedMetricAttention(**settings)


def dense_reference_metrics(layer):
    """Assemble L diag(exp(s)) U independently, one matrix at a time."""
    eye = torch.eye(layer.base_dim, device=layer.q_proj.weight.device, dtype=layer.q_proj.weight.dtype)
    lower_indices = torch.tril_indices(layer.base_dim, layer.base_dim, offset=-1)
    upper_indices = torch.triu_indices(layer.base_dim, layer.base_dim, offset=1)
    matrices = []
    for base in range(layer.num_base_heads):
        matrices.append(eye)
        for head in range(layer.generated_heads_per_base - 1):
            lower = eye.clone()
            upper = eye.clone()
            lower[tuple(lower_indices)] = layer.lu_lower[base, head]
            upper[tuple(upper_indices)] = layer.lu_upper[base, head]
            diagonal = torch.diag(layer.lu_log_diag[base, head].exp())
            matrices.append(lower @ diagonal @ upper)
    return torch.stack(matrices)


def dense_reference_attention(layer, inputs, context, attention_mask, padding_mask):
    batch, query_length, _ = inputs.shape
    key_length = context.shape[1]
    w_q = layer.q_proj.weight.view(layer.num_base_heads, layer.base_dim, layer.d_model)
    w_k = layer.k_proj.weight.view(layer.num_base_heads, layer.base_dim, layer.d_model)
    w_v = layer.v_proj.weight.view(layer.num_value_base_heads, layer.value_dim, layer.d_model)
    matrices = dense_reference_metrics(layer)
    heads = []
    weights = []
    for head, metric in enumerate(matrices):
        base = head // layer.generated_heads_per_base
        value_base = head // layer.generated_value_heads_per_base
        query = F.linear(inputs, w_q[base], layer.q_proj.bias.split(layer.base_dim)[base])
        keys = F.linear(context, w_k[base], layer.k_proj.bias.split(layer.base_dim)[base])
        values = F.linear(context, w_v[value_base], layer.v_proj.bias.split(layer.value_dim)[value_base])
        scores = query @ metric @ keys.transpose(-2, -1) / math.sqrt(layer.base_dim)
        if layer.causal:
            scores = scores.masked_fill(
                torch.ones(query_length, key_length, dtype=torch.bool).triu(key_length - query_length + 1),
                -torch.inf,
            )
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, -torch.inf)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, :], -torch.inf)
        probabilities = scores.softmax(-1)
        heads.append(probabilities @ values)
        weights.append(probabilities)
    output = F.linear(torch.cat(heads, dim=-1), layer.out_proj.weight, layer.out_proj.bias)
    return output.reshape(batch, query_length, layer.d_model), torch.stack(weights, dim=1)


def test_fixed_heads_are_exact_identities_and_only_other_heads_own_parameters():
    torch.manual_seed(8)
    layer = make_lu_attention(num_heads=16, num_base_heads=4, d_model=1024, head_dim=64)
    assert layer.fuse_lu_queries is True
    assert layer.num_generators == 0
    assert layer.lu_init_scale == 0.02
    assert set(dict(layer.named_parameters())) == {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
        "lu_lower", "lu_upper", "lu_log_diag",
    }
    expected_shape = (4, 3, 64 * 63 // 2)
    assert layer.lu_lower.shape == layer.lu_upper.shape == expected_shape
    assert layer.lu_log_diag.shape == (4, 3, 64)
    accounting = attention_accounting(layer, sequence_length=512, dtype=torch.bfloat16)
    assert accounting.lu_metric_parameters == 12 * 64**2
    assert accounting.generator_parameters == 0
    assert accounting.total_parameters == 1_884_160
    assert accounting.kv_cache_bytes_per_token_per_layer == 2 * 4 * 64 * 2
    assert accounting.attention_maps == 16
    assert layer.lu_lower.std().item() == pytest.approx(0.02 / 8, rel=0.05)
    assert layer.lu_upper.std().item() == pytest.approx(0.02 / 8, rel=0.05)
    assert layer.lu_log_diag.std().item() == pytest.approx(0.02, rel=0.12)

    metrics = layer.compute_metrics(dtype=torch.float64)
    layer = layer.double()
    torch.testing.assert_close(metrics, dense_reference_metrics(layer), rtol=1e-12, atol=1e-12)
    assert torch.equal(metrics[::4], torch.eye(64, dtype=torch.float64).expand(4, -1, -1))
    log_det_sign, log_det_magnitude = torch.linalg.slogdet(metrics.view(4, 4, 64, 64)[:, 1:])
    assert torch.equal(log_det_sign, torch.ones_like(log_det_sign))
    torch.testing.assert_close(log_det_magnitude, layer.lu_log_diag.sum(-1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["causal", "masked_causal", "masked_cross"])
@pytest.mark.parametrize("need_weights", [False, True])
def test_fixed_base_lu_matches_dense_fp64_forward_and_gradients(case, need_weights):
    torch.manual_seed(13)
    cross_attention = case == "masked_cross"
    layer = make_lu_attention(
        causal=not cross_attention, bias=True, value_dim=3, num_value_base_heads=3,
        fuse_base_qkv=True, fuse_lu_queries=False, sdpa_gqa_mode="expand",
    ).double()
    # Nontrivial factors distinguish ordering and transpose mistakes.
    with torch.no_grad():
        for parameter in (layer.lu_lower, layer.lu_upper, layer.lu_log_diag):
            parameter.normal_(std=0.4)
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            projection.bias.normal_(std=0.2)

    reference = make_lu_attention(
        causal=not cross_attention, bias=True, value_dim=3, num_value_base_heads=3,
    ).double()
    reference.load_state_dict(layer.state_dict())
    x = torch.randn(2, 5, 13, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    context = ref_context = mask = padding = None
    if cross_attention:
        context = torch.randn(2, 7, 13, dtype=torch.float64, requires_grad=True)
        ref_context = context.detach().clone().requires_grad_()
    if case != "causal":
        key_length = 7 if cross_attention else 5
        mask = torch.zeros(5, key_length, dtype=torch.bool)
        mask[2, 1] = True
        padding = torch.zeros(2, key_length, dtype=torch.bool)
        padding[0, 3] = padding[1, 4] = True

    actual = layer(x, context=context, attn_mask=mask, key_padding_mask=padding, need_weights=need_weights)
    expected, ref_weights = dense_reference_attention(
        reference, ref_x, ref_x if ref_context is None else ref_context, mask, padding,
    )
    if need_weights:
        actual, weights = actual
        torch.testing.assert_close(weights, ref_weights, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    loss_gradient = torch.randn_like(actual)
    actual.backward(loss_gradient)
    expected.backward(loss_gradient)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-12, atol=1e-12)
    if context is not None:
        torch.testing.assert_close(context.grad, ref_context.grad, rtol=1e-12, atol=1e-12)
    for (name, parameter), (ref_name, ref_parameter) in zip(layer.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("num_heads,head_dim", [(2, 3), (6, 1)])
def test_degenerate_fixed_base_lu_shapes(num_heads, head_dim):
    layer = make_lu_attention(num_heads=num_heads, head_dim=head_dim, lu_init_scale=0.0).double()
    x = torch.randn(2, 5, 13, dtype=torch.float64, requires_grad=True)
    assert torch.equal(layer.compute_metrics(), torch.eye(head_dim, dtype=x.dtype).expand(num_heads, -1, -1))
    layer(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    if num_heads == layer.num_base_heads:
        assert all(not name.startswith("lu_") for name in layer.state_dict())
    else:
        assert layer.lu_lower.numel() == layer.lu_upper.numel() == 0
        assert torch.isfinite(layer.lu_log_diag.grad).all()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"value_transform": "diag"}, "value_transform"),
        ({"metric_beta": 0.5}, "metric_beta"),
        ({"metric_clip_min": 0.1}, "clip"),
        ({"metric_clip_max": 2.0}, "clip"),
        ({"head_generator_symmetric_cap": 0.5}, "head_generator_symmetric_cap"),
        ({"lu_init_scale": -0.02}, "lu_init_scale"),
    ],
)
def test_fixed_base_lu_rejects_incompatible_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_lu_attention(**overrides)


@pytest.mark.parametrize("overrides", [{"num_generators": 8}, {"value_transform": "lie"}])
def test_fixed_base_lu_builder_rejects_legacy_generator_and_value_options(overrides):
    with pytest.raises(ValueError, match="requires zero generators"):
        build_attention("gt_mha_lu", d_model=8, num_heads=4, head_dim=4, **overrides)


def test_tiny_transformer_lu_round_trip_and_base_sized_decode_caches():
    torch.manual_seed(24)
    settings = {
        "vocab_size": 19, "d_model": 16, "num_layers": 2, "num_heads": 6,
        "num_base_heads": 2, "head_dim": 4, "attention_type": "gt_mha_lu", "num_generators": 0,
        "metric_mode": "lu", "value_transform": "none", "context_length": 8,
        "lu_init_scale": 0.3, "sdpa_gqa_mode": "expand",
    }
    model = TinyTransformerLM(**settings).double().eval()
    assert model.first_attention.lu_init_scale == 0.3
    assert model.first_attention.fuse_lu_queries
    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)
    restored = TinyTransformerLM(**settings, fuse_lu_queries=False).double().eval()
    assert not restored.first_attention.fuse_lu_queries
    restored.load_state_dict(torch.load(serialized), strict=True)
    tokens = torch.randint(0, 19, (2, 8))

    with torch.no_grad():
        logits = model(tokens)
        torch.testing.assert_close(logits, restored(tokens), rtol=0, atol=0)
        prefix, cache = model(tokens[:, :3], use_cache=True)
        torch.testing.assert_close(prefix, logits[:, :3], rtol=1e-12, atol=1e-12)
        assert all(k.shape == v.shape == (2, 2, 3, 4) for k, v in cache)
        static_cache = model.allocate_kv_cache(batch_size=2)
        static_prefix, _ = model(tokens[:, :3], past_key_values=static_cache, use_cache=True)
        torch.testing.assert_close(static_prefix, prefix, rtol=0, atol=0)
        for start, end in [(3, 4), (4, 6), (6, 8)]:
            chunk, cache = model(tokens[:, start:end], past_key_values=cache, use_cache=True)
            static_chunk, returned = model(
                tokens[:, start:end], past_key_values=static_cache, use_cache=True,
            )
            assert all(k.shape == v.shape == (2, 2, end, 4) for k, v in cache)
            assert all(a is b and a.length == end for a, b in zip(returned, static_cache))
            torch.testing.assert_close(chunk, logits[:, start:end], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(static_chunk, chunk, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_fixed_base_lu_runs_fused_forward_and_backward(dtype):
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires NVIDIA compute capability 8.0 or greater")
    torch.manual_seed(32)
    settings = {
        "d_model": 128, "num_heads": 8, "head_dim": 64, "num_base_heads": 2,
        "num_generators": 0, "metric_mode": "lu", "value_transform": "none",
        "fuse_base_qkv": True, "sdpa_gqa_mode": "expand",
    }
    fused = build_attention("gt_mha_lu", **settings).cuda().train()
    unfused = build_attention("gt_mha_lu", **settings, fuse_lu_queries=False).cuda().train()
    unfused.load_state_dict(fused.state_dict())
    x = torch.randn(2, 65, 128, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    upstream = torch.randn(x.shape, device="cuda", dtype=dtype)
    executed = {"forward": 0, "backward": 0}
    fused_queries = attention_module.fused_lu_queries

    def observe_backward(grad_inputs, grad_outputs):
        executed["backward"] += 1

    def observe_forward(*args, **kwargs):
        result = fused_queries(*args, **kwargs)
        if result is not None:
            executed["forward"] += 1
            assert "FusedLUQueries" in result.grad_fn.name()
            result.grad_fn.register_hook(observe_backward)
        return result

    with patch.object(attention_module, "fused_lu_queries", observe_forward):
        with torch.autocast("cuda", dtype=dtype):
            actual = fused(x)
            expected = unfused(ref_x)
        actual.backward(upstream)
        expected.backward(upstream)
    assert executed == {"forward": 1, "backward": 1}
    assert actual.dtype == expected.dtype == dtype
    gradients = [(x.grad, ref_x.grad)]
    gradients.extend((p.grad, q.grad) for p, q in zip(fused.parameters(), unfused.parameters()))
    for result, reference in [(actual, expected), *gradients]:
        result, reference = result.float(), reference.float()
        max_relative_error = 0.035 if dtype == torch.bfloat16 else 0.012
        assert (result - reference).norm().item() / reference.norm().item() < max_relative_error
