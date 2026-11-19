from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import GroupedQueryAttention
from lgma.checkpointing import load_full_checkpoint
from lgma.lu_export import export_lu_model, fold_lu_attention
from lgma.transformer import TinyTransformerLM


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = {
    "attention_type": "gt_mha_lu",
    "d_model": 13,
    "num_layers": 2,
    "num_heads": 6,
    "head_dim": 4,
    "num_base_heads": 2,
    "num_generators": 0,
    "metric_mode": "lu",
    "value_transform": "none",
    "context_length": 8,
    "dropout": 0.2,
    "fuse_base_qkv": True,
    "sdpa_gqa_mode": "expand",
}


@torch.no_grad()
def set_nontrivial_lu(layer):
    for parameter in (layer.lu_lower, layer.lu_upper, layer.lu_log_diag):
        parameter.normal_(std=0.4)
    if layer.q_proj.bias is not None:
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            projection.bias.normal_(std=0.3)
    if layer.learn_head_temperature:
        # Vary temperatures independently of the metrics, including the clamp.
        temperatures = layer.head_logit_scale.new_tensor([0.7, 1.4, 2.0, 0.9, 0.4, -0.2])
        layer.head_logit_scale.copy_(temperatures)


def make_lu_model(vocab_size):
    model = TinyTransformerLM(vocab_size=vocab_size, **MODEL_CONFIG)
    for block in model.blocks:
        set_nontrivial_lu(block.attn)
    return model


@pytest.mark.parametrize(
    ("logit_scale_mode", "learn_head_temperature", "mask_kind"),
    [("sqrt_dim", False, "boolean"), ("rms_metric", True, "additive")],
)
@torch.no_grad()
def test_folded_queries_preserve_projection_orientation_and_attention(
    logit_scale_mode, learn_head_temperature, mask_kind,
):
    torch.manual_seed(18)
    cross_attention = mask_kind == "additive"
    source = LieGeneratedMetricAttention(
        d_model=13, num_heads=6, head_dim=4, num_base_heads=2, num_generators=0,
        metric_mode="lu", value_transform="none", bias=True, causal=not cross_attention,
        dropout=0.25, logit_scale_mode=logit_scale_mode,
        learn_head_temperature=learn_head_temperature, sdpa_gqa_mode="expand",
    ).double()
    set_nontrivial_lu(source)
    original = {name: tensor.clone() for name, tensor in source.state_dict().items()}

    folded = fold_lu_attention(source)

    assert type(folded) is GroupedQueryAttention
    assert not folded.training
    assert source.training
    source_state = source.state_dict()
    assert source_state.keys() == original.keys()
    for name, tensor in source_state.items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    assert folded.q_proj.weight.shape == (24, 13)
    assert set(folded.state_dict()) == {
        f"{projection}.{part}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
        for part in ("weight", "bias")
    }
    for name, tensor in folded.state_dict().items():
        assert tensor.dtype == torch.float64
        assert tensor.device == source.q_proj.weight.device
        if not name.startswith("q_proj."):
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
            assert tensor.data_ptr() != source_state[name].data_ptr()

    source.eval()
    x = torch.randn(2, 5, 13, dtype=torch.float64)
    x[0, 0].zero_()  # A zero query input exposes how the query bias was folded.
    metrics = source.compute_metrics()
    base_q = source.q_proj(x).reshape(2, 5, 2, 4)
    expected = []
    for head, metric in enumerate(metrics):
        base = head // source.generated_heads_per_base
        query = base_q[:, :, base, :] @ metric
        if logit_scale_mode == "rms_metric":
            rms = metric.float().square().sum().div(source.base_dim).sqrt().double()
            query = query / rms.clamp_min(1e-6)
        if learn_head_temperature:
            query = query * source.head_logit_scale[head].clamp_min(1e-6)
        expected.append(query)
    torch.testing.assert_close(
        folded.q_proj(x), torch.cat(expected, dim=-1), rtol=1e-12, atol=1e-12,
    )

    context = torch.randn(2, 7, 13, dtype=x.dtype) if cross_attention else None
    key_length = context.shape[1] if cross_attention else x.shape[1]
    key_padding_mask = torch.zeros(2, key_length, dtype=torch.bool)
    key_padding_mask[0, 2] = key_padding_mask[1, 3] = True
    if cross_attention:
        mask = torch.randn(2, 6, 5, key_length, dtype=x.dtype) * 0.3
    else:
        mask = torch.zeros(5, key_length, dtype=torch.bool)
        mask[3, 1] = True

    kwargs = {"context": context, "attn_mask": mask, "key_padding_mask": key_padding_mask}
    actual_out, actual_weights = folded(x, **kwargs, need_weights=True)
    expected_out, expected_weights = source(x, **kwargs, need_weights=True)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(folded(x, **kwargs), source(x, **kwargs), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        folded(x, context=context), source(x, context=context), rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("cache_mode", ["dynamic", "static"])
@torch.no_grad()
def test_lu_model_export_preserves_full_logits_and_base_sized_decode(cache_mode):
    torch.manual_seed(29)
    source = make_lu_model(vocab_size=19).double()
    source_attentions = [block.attn for block in source.blocks]
    exported = export_lu_model(source)
    assert source.training
    assert not exported.training
    assert [block.attn for block in source.blocks] == source_attentions
    assert all(type(block.attn) is GroupedQueryAttention for block in exported.blocks)
    assert not any("lu_" in name or "head_logit_scale" in name for name in exported.state_dict())
    source_state = source.state_dict()
    for name, tensor in exported.state_dict().items():
        if ".attn." not in name:
            torch.testing.assert_close(tensor, source_state[name], rtol=0, atol=0)
            assert tensor.data_ptr() != source_state[name].data_ptr()

    source.eval()
    tokens = torch.randint(0, source.vocab_size, (2, source.context_length))
    full_logits = source(tokens)
    torch.testing.assert_close(exported(tokens), full_logits, rtol=1e-12, atol=1e-12)
    cache = exported.allocate_kv_cache(batch_size=2) if cache_mode == "static" else None
    for start, end in [(0, 3), (3, 4), (4, 6), (6, 8)]:
        logits, present = exported(tokens[:, start:end], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(logits, full_logits[:, start:end], rtol=1e-12, atol=1e-12)
        assert len(present) == len(source.blocks)
        assert all(layer[0].shape == layer[1].shape == (2, 2, end, 4) for layer in present)
        if cache_mode == "static":
            assert all(
                current is layer and layer.length == end for current, layer in zip(present, cache)
            )
        cache = present


def test_export_cli_checkpoint_reloads_with_tinystories_runtime(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    from export_fixed_base_lu import main as export_main
    from tinystories_runtime import build_tokenizer, load_tinystories_checkpoint

    torch.manual_seed(37)
    train_text = "Once upon a time, Tim made cookies.\n"
    val_text = "Tim was quiet!\n"
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    tokenizer = build_tokenizer(train_text, val_text)
    source = make_lu_model(vocab_size=tokenizer.vocab_size)
    checkpoint_path = tmp_path / "trained.pt"
    export_path = tmp_path / "folded.pt"
    run_manifest = {"seed": 37, "attention_type": "gt_mha_lu"}
    torch.save(
        {
            "model_config": MODEL_CONFIG,
            "model_state": source.state_dict(),
            "step": 123,
            "args": {"data_path": train_path, "val_data_path": val_path},
            "optimizer_state": {"state": {}, "param_groups": []},
            "scaler_state": None,
            "data_generator_states": [torch.Generator().get_state()],
            "run_manifest": run_manifest,
        },
        checkpoint_path,
    )
    source_bytes = checkpoint_path.read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_fixed_base_lu.py",
            "--checkpoint", str(checkpoint_path), "--output", str(export_path),
        ],
    )
    export_main()
    assert checkpoint_path.read_bytes() == source_bytes

    exported_payload = load_full_checkpoint(export_path, map_location="cpu")
    assert exported_payload["inference_only"] is True
    training_state_keys = {"optimizer_state", "scaler_state", "data_generator_states"}
    assert training_state_keys.isdisjoint(exported_payload)
    assert exported_payload["deployment"]["format"] == "fixed_base_lu_to_gqa"
    assert exported_payload["deployment"]["source_checkpoint"] == str(checkpoint_path.resolve())
    assert exported_payload["deployment"]["source_model_config"] == MODEL_CONFIG
    assert exported_payload["deployment"]["source_run_manifest"] == run_manifest
    exported, restored_tokenizer, train_ids, val_ids, config, step = load_tinystories_checkpoint(
        export_path, torch.device("cpu"),
    )
    assert config["attention_type"] == "gqa"
    assert config["num_kv_heads"] == 2
    assert step == 123
    assert all(type(block.attn) is GroupedQueryAttention for block in exported.blocks)
    torch.testing.assert_close(train_ids, tokenizer.encode(train_text))
    torch.testing.assert_close(val_ids, tokenizer.encode(val_text))
    assert restored_tokenizer.stoi == tokenizer.stoi
    with torch.no_grad():
        sample = train_ids[:8].unsqueeze(0)
        expected = source.eval()(sample)
        logits, cache = exported(sample[:, :5], use_cache=True)
        tail, _ = exported(sample[:, 5:], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(torch.cat((logits, tail), dim=1), expected, rtol=2e-5, atol=2e-6)
