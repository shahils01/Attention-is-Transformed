import pytest
import torch

from lgma.transformer import TinyTransformerLM
from lgma.kv_cache import StaticKVCache


def build_model(attention_type: str, **overrides) -> TinyTransformerLM:
    config = {
        "vocab_size": 23,
        "d_model": 24,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 6,
        "attention_type": attention_type,
        "num_generators": 3 if attention_type.startswith("lgma") else 0,
        "context_length": 12,
        "dropout": 0.0,
    }
    config.update(overrides)
    model = TinyTransformerLM(**config)
    model.eval()
    return model


@pytest.mark.parametrize(
    ("attention_type", "overrides"),
    [
        ("mha", {}),
        ("reduced_mha", {}),
        ("collaborative", {}),
        ("shared_identity", {}),
        ("mqa", {}),
        ("gqa", {"num_kv_heads": 2}),
        (
            "lgma_quad",
            {
                "num_base_heads": 2,
                "value_transform": "lie_quadratic",
                "base_dim": 6,
                "value_dim": 6,
            },
        ),
    ],
)
def test_cached_logits_match_full_context(attention_type, overrides):
    torch.manual_seed(0)
    model = build_model(attention_type, **overrides)
    input_ids = torch.randint(0, model.vocab_size, (2, 8))

    with torch.no_grad():
        full_logits = model(input_ids)
        cached_prefix_logits, cache = model(input_ids[:, :3], use_cache=True)

        assert torch.allclose(
            cached_prefix_logits,
            full_logits[:, :3],
            atol=2e-5,
            rtol=2e-5,
        )
        for position in range(3, input_ids.shape[1]):
            step_logits, cache = model(
                input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            assert torch.allclose(
                step_logits[:, 0],
                full_logits[:, position],
                atol=2e-5,
                rtol=2e-5,
            )

    assert len(cache) == len(model.blocks)
    assert all(key.shape[-2] == input_ids.shape[1] for key, _ in cache)
    assert all(value.shape[-2] == input_ids.shape[1] for _, value in cache)


def test_cache_shapes_distinguish_mha_and_multibase_gt_mha():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 23, (2, 5))
    mha = build_model("mha")
    gt_mha = build_model(
        "lgma_quad",
        num_base_heads=2,
        value_transform="lie_quadratic",
    )

    with torch.no_grad():
        _, mha_cache = mha(input_ids, use_cache=True)
        _, gt_cache = gt_mha(input_ids, use_cache=True)

    mha_key, mha_value = mha_cache[0]
    gt_key, gt_value = gt_cache[0]
    assert mha_key.shape == mha_value.shape == (2, 4, 5, 6)
    assert gt_key.shape == gt_value.shape == (2, 2, 5, 6)
    assert sum(t.numel() for t in gt_cache[0]) * 2 == sum(
        t.numel() for t in mha_cache[0]
    )


def test_using_cache_does_not_change_state_dict():
    torch.manual_seed(0)
    model = build_model(
        "lgma_quad",
        num_base_heads=2,
        value_transform="lie_quadratic",
    )
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    with torch.no_grad():
        _, cache = model(torch.tensor([[1, 2, 3]]), use_cache=True)
        model(torch.tensor([[4]]), past_key_values=cache, use_cache=True)

    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


@pytest.mark.parametrize(
    ("attention_type", "overrides"),
    [
        ("mha", {}),
        ("reduced_mha", {}),
        ("collaborative", {}),
        ("shared_identity", {}),
        ("mqa", {}),
        ("gqa", {"num_kv_heads": 2}),
        (
            "lgma_quad",
            {"num_base_heads": 2, "value_transform": "lie_quadratic"},
        ),
    ],
)
def test_static_cache_matches_full_context_without_reallocating(attention_type, overrides):
    torch.manual_seed(12)
    model = build_model(attention_type, **overrides)
    input_ids = torch.randint(0, model.vocab_size, (2, 8))
    cache = model.allocate_kv_cache(batch_size=2, max_length=8)

    with torch.no_grad():
        full_logits = model(input_ids)
        prefix_logits, returned = model(
            input_ids[:, :3], past_key_values=cache, use_cache=True
        )
        assert returned[0] is cache[0]
        assert torch.allclose(prefix_logits, full_logits[:, :3], atol=2e-5, rtol=2e-5)
        pointers = [(layer[0].data_ptr(), layer[1].data_ptr()) for layer in cache]
        allocated = [layer.allocated_bytes for layer in cache]

        for position in range(3, input_ids.shape[1]):
            step_logits, returned = model(
                input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            assert returned[0] is cache[0]
            assert torch.allclose(
                step_logits[:, 0], full_logits[:, position], atol=2e-5, rtol=2e-5
            )
            assert [layer.length for layer in cache] == [position + 1] * len(cache)
            assert [(layer[0].data_ptr(), layer[1].data_ptr()) for layer in cache] == pointers
            assert [layer.allocated_bytes for layer in cache] == allocated


def test_static_cache_capacity_and_inference_only():
    model = build_model("lgma_quad", num_base_heads=2)
    cache = model.allocate_kv_cache(batch_size=1, max_length=3)
    with pytest.raises(RuntimeError, match="requires torch.no_grad"):
        model(torch.tensor([[1]]), past_key_values=cache, use_cache=True)
    assert all(layer.length == 0 for layer in cache)

    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="exceeds max_length"):
            model(torch.tensor([[4]]), past_key_values=cache, use_cache=True)
        for layer in cache:
            assert isinstance(layer, StaticKVCache)
            layer.reset()
        model(torch.tensor([[5]]), past_key_values=cache, use_cache=True)
    assert all(layer.length == 1 for layer in cache)
