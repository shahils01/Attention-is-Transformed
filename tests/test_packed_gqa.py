import pytest
import torch

from lgma.baselines import GroupedQueryAttention
from lgma.packed_gqa import PackedGroupedQueryAttention, pack_gqa_model
from lgma.transformer import TinyTransformerLM


def _make_model(attention_type="gqa"):
    return TinyTransformerLM(
        vocab_size=37,
        d_model=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        attention_type=attention_type,
        context_length=12,
        dropout=0.2,
        num_kv_heads=2,
    )


def _parameter_bytes(model):
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


@torch.inference_mode()
@pytest.mark.parametrize("static", [False, True])
def test_packed_gqa_model_matches_full_and_cached_logits(static):
    torch.manual_seed(0)
    ordinary = _make_model().eval()
    packed = pack_gqa_model(ordinary)
    tokens = torch.randint(0, ordinary.vocab_size, (2, 8))

    assert isinstance(ordinary.first_attention, GroupedQueryAttention)
    assert isinstance(packed.first_attention, PackedGroupedQueryAttention)
    assert _parameter_bytes(packed) == _parameter_bytes(ordinary)
    assert ordinary.token_embedding.weight.data_ptr() != packed.token_embedding.weight.data_ptr()
    assert ordinary.token_embedding.weight.requires_grad
    assert all(not parameter.requires_grad for parameter in packed.parameters())

    expected = ordinary(tokens)
    torch.testing.assert_close(packed(tokens), expected)

    initial = packed.allocate_kv_cache(batch_size=2, max_length=12) if static else None
    prefill, cache = packed(tokens[:, :4], use_cache=True, past_key_values=initial)
    if static:
        assert cache[0] is initial[0]
        assert cache[0].allocated_bytes == 2 * 2 * 2 * 12 * 4 * prefill.element_size()
    else:
        for layer_cache in cache:
            for projection in layer_cache:
                if hasattr(projection, "untyped_storage"):
                    storage_bytes = projection.untyped_storage().nbytes()
                else:
                    storage_bytes = projection.storage().size() * projection.element_size()
                assert storage_bytes == projection.numel() * projection.element_size()
    decode, cache = packed(tokens[:, 4:5], use_cache=True, past_key_values=cache)
    chunk, _ = packed(tokens[:, 5:], use_cache=True, past_key_values=cache)

    torch.testing.assert_close(torch.cat((prefill, decode, chunk), dim=1), expected)


@torch.inference_mode()
@pytest.mark.parametrize("cross_attention", [False, True])
@pytest.mark.parametrize("need_weights", [False, True])
def test_packed_gqa_layer_matches_masked_and_weighted_paths(cross_attention, need_weights):
    torch.manual_seed(0)
    ordinary = GroupedQueryAttention(
        12, num_heads=4, head_dim=4, num_kv_heads=2, dropout=0.3, bias=True, causal=True
    ).eval()
    packed = PackedGroupedQueryAttention(ordinary)
    query = torch.randn(2, 3, 12)
    context = torch.randn(2, 5, 12) if cross_attention else None
    source_len = 5 if cross_attention else 3
    mask = torch.zeros(3, source_len, dtype=torch.bool)
    mask[-1, 0] = True
    padding = torch.zeros(2, source_len, dtype=torch.bool)
    padding[1, -1] = True
    kwargs = {
        "context": context,
        "attn_mask": mask,
        "key_padding_mask": padding,
        "need_weights": need_weights,
        "use_cache": True,
    }

    torch.testing.assert_close(packed(query, **kwargs), ordinary(query, **kwargs))


def test_packed_gqa_decode_skips_the_offset_mask(monkeypatch):
    import lgma.packed_gqa as packed_gqa

    ordinary = GroupedQueryAttention(16, num_heads=4, head_dim=4, num_kv_heads=2, causal=True).eval()
    packed = PackedGroupedQueryAttention(ordinary)
    received = []
    grouped_sdpa = packed_gqa._grouped_sdpa_attention

    def observe_sdpa(q, k, v, *, dropout_p, causal):
        received.append((q.shape[-2], k.shape[-2], causal))
        return grouped_sdpa(q, k, v, dropout_p=dropout_p, causal=causal)

    monkeypatch.setattr(packed_gqa, "_grouped_sdpa_attention", observe_sdpa)
    x = torch.randn(1, 4, 16)
    with torch.inference_mode():
        _, cache = packed(x[:, :3], use_cache=True)
        packed(x[:, 3:], past_key_value=cache)

    assert received == [(3, 3, True), (1, 4, False)]


def test_pack_gqa_model_rejects_training_or_non_gqa():
    with pytest.raises(ValueError, match="eval mode"):
        pack_gqa_model(_make_model())
    with pytest.raises(TypeError, match="ordinary GroupedQueryAttention"):
        pack_gqa_model(_make_model("mha").eval())

    packed = pack_gqa_model(_make_model().eval()).train()
    with pytest.raises(RuntimeError, match="inference-only"):
        packed(torch.zeros(1, 2, dtype=torch.long))
