from __future__ import annotations

from types import SimpleNamespace
import json

import torch
from torch import nn
import pytest

from lgma.bert import (
    BertCollaborativeConfig,
    BertCollaborativeSelfAttention,
    BertGqaConfig,
    BertGqaSelfAttention,
    BertGtMhaConfig,
    BertGtMhaSelfAttention,
    bert_parameter_counts,
    initialize_collaborative_from_bert_attention,
    initialize_gqa_from_bert_attention,
    initialize_gt_mha_from_bert_attention,
    load_bert_masked_lm,
    load_bert_sequence_classifier,
    replace_bert_self_attention,
)


class FakeBertSelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)


class FakeBertLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = FakeBertSelfAttention(hidden_size)


class FakeBertModel(nn.Module):
    def __init__(self, hidden_size: int = 96, num_heads: int = 12, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0.0,
            is_decoder=False,
        )
        self.bert = nn.Module()
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList(
            [FakeBertLayer(hidden_size) for _ in range(num_layers)]
        )


def test_bert_gt_mha_forward_and_padding_mask() -> None:
    torch.manual_seed(0)
    module = BertGtMhaSelfAttention(
        BertGtMhaConfig(
            hidden_size=96,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.0,
        )
    )
    hidden = torch.randn(2, 5, 96)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    context, probabilities = module(
        hidden,
        attention_mask=mask,
        output_attentions=True,
    )
    assert context.shape == (2, 5, 96)
    assert probabilities.shape == (2, 12, 5, 5)
    assert torch.allclose(probabilities[0, :, :, 3:], torch.zeros_like(probabilities[0, :, :, 3:]))
    assert torch.allclose(probabilities[1, :, :, 4:], torch.zeros_like(probabilities[1, :, :, 4:]))


def test_bert_gt_mha_supports_wider_qk_bases_with_standard_values() -> None:
    torch.manual_seed(0)
    module = BertGtMhaSelfAttention(
        BertGtMhaConfig(
            hidden_size=96,
            num_attention_heads=12,
            num_base_heads=6,
            num_generators=8,
            qk_base_dim=14,
            value_head_dim=8,
            attention_probs_dropout_prob=0.0,
            attention_type="gt_mha_residual",
        )
    )
    hidden = torch.randn(2, 5, 96, requires_grad=True)
    context, probabilities = module(
        hidden,
        attention_mask=torch.ones(2, 5),
        output_attentions=True,
    )
    context.square().mean().backward()

    assert module.query.out_features == 6 * 14
    assert module.key.out_features == 6 * 14
    assert module.value.out_features == 6 * 8
    assert context.shape == (2, 5, 96)
    assert probabilities.shape == (2, 12, 5, 5)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_bert_gqa_forward_padding_mask_and_backward() -> None:
    torch.manual_seed(0)
    module = BertGqaSelfAttention(
        BertGqaConfig(
            hidden_size=96,
            num_attention_heads=12,
            num_kv_heads=4,
            attention_probs_dropout_prob=0.0,
        )
    )
    hidden = torch.randn(2, 5, 96, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    context, probabilities = module(hidden, attention_mask=mask, output_attentions=True)
    context.square().mean().backward()

    assert context.shape == (2, 5, 96)
    assert probabilities.shape == (2, 12, 5, 5)
    assert torch.allclose(probabilities[0, :, :, 3:], torch.zeros_like(probabilities[0, :, :, 3:]))
    assert torch.allclose(probabilities[1, :, :, 4:], torch.zeros_like(probabilities[1, :, :, 4:]))
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_bert_collaborative_forward_padding_mask_and_backward() -> None:
    torch.manual_seed(0)
    module = BertCollaborativeSelfAttention(
        BertCollaborativeConfig(
            hidden_size=96,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.0,
        )
    )
    hidden = torch.randn(2, 5, 96, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    context, probabilities = module(hidden, attention_mask=mask, output_attentions=True)
    context.square().mean().backward()

    assert context.shape == (2, 5, 96)
    assert probabilities.shape == (2, 12, 5, 5)
    assert torch.allclose(probabilities[0, :, :, 3:], torch.zeros_like(probabilities[0, :, :, 3:]))
    assert torch.allclose(probabilities[1, :, :, 4:], torch.zeros_like(probabilities[1, :, :, 4:]))
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_bert_gt_mha_value_lie_transform_participates_in_backward() -> None:
    torch.manual_seed(0)
    module = BertGtMhaSelfAttention(
        BertGtMhaConfig(
            hidden_size=96,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.0,
            attention_type="gt_mha_residual",
        )
    )
    hidden = torch.randn(2, 5, 96)
    context = module(hidden, attention_mask=torch.ones(2, 5))[0]
    context.square().mean().backward()

    missing_gradients = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing_gradients == []
    for parameter_name in ("value_generators", "value_theta"):
        gradient = getattr(module.gt_attention, parameter_name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_bert_optimized_sdpa_fused_qkv_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(3)
    common = {
        "hidden_size": 96,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.0,
        "attention_type": "gt_mha_residual",
    }
    reference = BertGtMhaSelfAttention(BertGtMhaConfig(**common))
    optimized = BertGtMhaSelfAttention(
        BertGtMhaConfig(
            **common,
            use_sdpa=True,
            fuse_base_qkv=True,
            sdpa_gqa_mode="native",
        )
    )
    optimized.load_state_dict(reference.state_dict())
    reference.eval()
    optimized.eval()
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    upstream = torch.randn(2, 5, 96)
    reference_input = torch.randn(2, 5, 96, requires_grad=True)
    optimized_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input, attention_mask=mask)[0]
    optimized_output = optimized(optimized_input, attention_mask=mask)[0]
    (reference_output * upstream).sum().backward()
    (optimized_output * upstream).sum().backward()

    assert torch.allclose(reference_output, optimized_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reference_input.grad, optimized_input.grad, atol=1e-5, rtol=1e-5)
    reference_gradients = dict(reference.named_parameters())
    for name, parameter in optimized.named_parameters():
        assert parameter.grad is not None
        assert torch.allclose(
            reference_gradients[name].grad,
            parameter.grad,
            atol=2e-5,
            rtol=2e-4,
        ), name


def test_teacher_head_averaging_initializes_all_three_base_projections() -> None:
    teacher = FakeBertSelfAttention(96)
    with torch.no_grad():
        for offset, projection in enumerate((teacher.query, teacher.key, teacher.value)):
            projection.weight.copy_(
                torch.arange(projection.weight.numel()).view_as(projection.weight) + offset
            )
            projection.bias.copy_(
                torch.arange(projection.bias.numel()).view_as(projection.bias) + offset
            )
    student = BertGtMhaSelfAttention(
        BertGtMhaConfig(hidden_size=96, num_attention_heads=12)
    )
    initialize_gt_mha_from_bert_attention(student, teacher)

    expected_query = teacher.query.weight.view(4, 3, 8, 96).mean(dim=1).reshape(32, 96)
    expected_key = teacher.key.weight.view(4, 3, 8, 96).mean(dim=1).reshape(32, 96)
    expected_value = teacher.value.weight.view(4, 3, 8, 96).mean(dim=1).reshape(32, 96)
    assert torch.equal(student.query.weight, expected_query)
    assert torch.equal(student.key.weight, expected_key)
    assert torch.equal(student.value.weight, expected_value)


def test_widened_qk_bases_require_random_initialization() -> None:
    teacher = FakeBertSelfAttention(96)
    student = BertGtMhaSelfAttention(
        BertGtMhaConfig(
            hidden_size=96,
            num_attention_heads=12,
            num_base_heads=6,
            qk_base_dim=14,
            value_head_dim=8,
        )
    )
    with pytest.raises(ValueError, match="use random initialization"):
        initialize_gt_mha_from_bert_attention(student, teacher)


def test_parameter_matched_gt_mha_configuration_is_audited() -> None:
    model = FakeBertModel(num_layers=1)
    audit = replace_bert_self_attention(
        model,
        attention_type="gt_mha_residual",
        num_base_heads=6,
        num_generators=8,
        qk_base_dim=14,
        value_head_dim=8,
        initialize_from_mha=False,
        enforce_paper_gt_mha=False,
    )
    replacement = model.bert.encoder.layer[0].attention.self
    assert isinstance(replacement, BertGtMhaSelfAttention)
    assert replacement.gt_attention.base_dim == 14
    assert replacement.gt_attention.value_dim == 8
    assert audit[0]["gt_mha_config"]["qk_base_dim"] == 14
    assert audit[0]["gt_mha_config"]["value_head_dim"] == 8


def test_gqa_initialization_copies_queries_and_averages_kv_heads() -> None:
    teacher = FakeBertSelfAttention(96)
    with torch.no_grad():
        for offset, projection in enumerate((teacher.query, teacher.key, teacher.value)):
            projection.weight.copy_(
                torch.arange(projection.weight.numel()).view_as(projection.weight) + offset
            )
            projection.bias.copy_(
                torch.arange(projection.bias.numel()).view_as(projection.bias) + offset
            )
    student = BertGqaSelfAttention(
        BertGqaConfig(hidden_size=96, num_attention_heads=12, num_kv_heads=4)
    )
    initialize_gqa_from_bert_attention(student, teacher)

    assert torch.equal(student.query.weight, teacher.query.weight)
    assert torch.equal(student.query.bias, teacher.query.bias)
    expected_key = teacher.key.weight.view(4, 3, 8, 96).mean(dim=1).reshape(32, 96)
    expected_value = teacher.value.weight.view(4, 3, 8, 96).mean(dim=1).reshape(32, 96)
    assert torch.equal(student.key.weight, expected_key)
    assert torch.equal(student.value.weight, expected_value)


def test_collaborative_initialization_averages_query_key_and_copies_value() -> None:
    teacher = FakeBertSelfAttention(96)
    with torch.no_grad():
        for offset, projection in enumerate((teacher.query, teacher.key, teacher.value)):
            projection.weight.copy_(
                torch.arange(projection.weight.numel()).view_as(projection.weight) + offset
            )
            projection.bias.copy_(
                torch.arange(projection.bias.numel()).view_as(projection.bias) + offset
            )
    student = BertCollaborativeSelfAttention(
        BertCollaborativeConfig(hidden_size=96, num_attention_heads=12)
    )
    initialize_collaborative_from_bert_attention(student, teacher)

    expected_query = teacher.query.weight.view(1, 12, 8, 96).mean(dim=1).reshape(8, 96)
    expected_key = teacher.key.weight.view(1, 12, 8, 96).mean(dim=1).reshape(8, 96)
    assert torch.equal(student.query.weight, expected_query)
    assert torch.equal(student.key.weight, expected_key)
    assert torch.equal(student.value.weight, teacher.value.weight)
    assert torch.equal(
        student.collaborative_attention.mixing_vector,
        torch.ones_like(student.collaborative_attention.mixing_vector),
    )


def test_replace_bert_attention_is_audited_and_reduces_parameters() -> None:
    model = FakeBertModel()
    audit = replace_bert_self_attention(model, attention_type="gt_mha_exact")
    assert len(audit) == 2
    assert all(entry["initialization"] == "mean_teacher_heads" for entry in audit)
    assert all(entry["attention_parameter_reduction"] > 0 for entry in audit)
    assert all(
        isinstance(layer.attention.self, BertGtMhaSelfAttention)
        for layer in model.bert.encoder.layer
    )


def test_replace_bert_attention_with_gqa_is_audited_and_reduces_parameters() -> None:
    model = FakeBertModel()
    audit = replace_bert_self_attention(
        model, attention_type="gqa", num_kv_heads=4
    )
    assert len(audit) == 2
    assert all(entry["gqa_config"]["num_kv_heads"] == 4 for entry in audit)
    assert all(entry["attention_parameter_reduction"] > 0 for entry in audit)
    assert all(
        isinstance(layer.attention.self, BertGqaSelfAttention)
        for layer in model.bert.encoder.layer
    )


def test_replace_bert_attention_with_mqa_uses_one_kv_head() -> None:
    model = FakeBertModel(num_layers=1)
    audit = replace_bert_self_attention(model, attention_type="mqa", num_kv_heads=4)
    replacement = model.bert.encoder.layer[0].attention.self
    assert isinstance(replacement, BertGqaSelfAttention)
    assert replacement.gqa_config.num_kv_heads == 1
    assert audit[0]["attention_type"] == "mqa"
    assert audit[0]["attention_parameter_reduction"] > 0


def test_replace_bert_attention_with_collaborative_is_audited() -> None:
    model = FakeBertModel(num_layers=1)
    audit = replace_bert_self_attention(model, attention_type="collaborative")
    replacement = model.bert.encoder.layer[0].attention.self
    assert isinstance(replacement, BertCollaborativeSelfAttention)
    assert audit[0]["initialization"] == "mean_teacher_query_key_copy_value"
    assert audit[0]["attention_parameter_reduction"] > 0
    assert "collaborative_config" in audit[0]


def test_gqa_rejects_incompatible_kv_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        replace_bert_self_attention(
            FakeBertModel(num_layers=1), attention_type="gqa", num_kv_heads=5
        )


def test_mha_replacement_is_a_noop() -> None:
    model = FakeBertModel(num_layers=1)
    original = model.bert.encoder.layer[0].attention.self
    assert replace_bert_self_attention(model, attention_type="mha") == []
    assert model.bert.encoder.layer[0].attention.self is original


def test_paper_guard_rejects_noncanonical_counts() -> None:
    model = FakeBertModel()
    try:
        replace_bert_self_attention(
            model,
            attention_type="gt_mha_exact",
            num_base_heads=3,
            num_generators=4,
        )
    except ValueError as exc:
        assert "paper GT-MHA" in str(exc)
    else:
        raise AssertionError("expected the paper GT-MHA guard to reject noncanonical counts")


def test_bert_generator_mixing_none_requires_disabling_paper_guard() -> None:
    guarded = FakeBertModel(num_layers=1)
    with pytest.raises(ValueError, match="generator_mixing"):
        replace_bert_self_attention(
            guarded,
            attention_type="gt_mha_residual",
            generator_mixing="none",
        )

    experimental = FakeBertModel(num_layers=1)
    audit = replace_bert_self_attention(
        experimental,
        attention_type="gt_mha_residual",
        generator_mixing="none",
        enforce_paper_gt_mha=False,
    )
    replacement = experimental.bert.encoder.layer[0].attention.self
    assert replacement.gt_attention.generator_mixing == "none"
    assert audit[0]["gt_mha_config"]["generator_mixing"] == "none"


def test_hugging_face_bert_forward_after_replacement() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    replace_bert_self_attention(model, attention_type="gt_mha_exact")
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]]
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        output_attentions=True,
    )
    assert outputs.logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(outputs.loss)
    assert outputs.attentions[0].shape == (2, 12, 8, 8)


def test_hugging_face_bert_forward_after_gqa_replacement() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    replace_bert_self_attention(model, attention_type="gqa", num_kv_heads=4)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]]
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        output_attentions=True,
    )
    assert outputs.logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(outputs.loss)
    assert outputs.attentions[0].shape == (2, 12, 8, 8)


@pytest.mark.parametrize("attention_type", ["mqa", "collaborative"])
def test_hugging_face_bert_forward_after_additional_baseline_replacement(
    attention_type: str,
) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    replace_bert_self_attention(model, attention_type=attention_type)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        output_attentions=True,
    )
    assert outputs.logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(outputs.loss)
    assert outputs.attentions[0].shape == (2, 12, 8, 8)


def test_bert_parameter_counts_include_self_and_output_attention() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    counts = bert_parameter_counts(model)
    layers = model.bert.encoder.layer
    expected_self_attention = sum(
        parameter.numel()
        for layer in layers
        for parameter in layer.attention.self.parameters()
    )
    expected_output = sum(
        parameter.numel()
        for layer in layers
        for parameter in layer.attention.output.parameters()
    )
    assert counts["model_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert counts["trainable_parameter_count"] == counts["model_parameter_count"]
    assert counts["self_attention_parameter_count"] == expected_self_attention
    assert counts["attention_output_parameter_count"] == expected_output
    assert counts["attention_block_parameter_count"] == expected_self_attention + expected_output


def test_saved_gt_mha_checkpoint_round_trip(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    replace_bert_self_attention(model, attention_type="gt_mha_exact")
    config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "gt_mha_state_dict.pt")
    (tmp_path / "bert_gt_mha_manifest.json").write_text(
        json.dumps(
            {
                "attention_type": "gt_mha_exact",
                "num_base_heads": 4,
                "num_generators": 8,
            }
        ),
        encoding="utf-8",
    )

    loaded, audit = load_bert_masked_lm(str(tmp_path))
    assert len(audit) == 1
    expected = model.bert.encoder.layer[0].attention.self.gt_attention.q_proj.weight
    actual = loaded.bert.encoder.layer[0].attention.self.gt_attention.q_proj.weight
    assert torch.equal(actual, expected)


def test_saved_gqa_checkpoint_round_trip(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    replace_bert_self_attention(model, attention_type="gqa", num_kv_heads=4)
    config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "gt_mha_state_dict.pt")
    (tmp_path / "bert_gt_mha_manifest.json").write_text(
        json.dumps({"attention_type": "gqa", "num_kv_heads": 4}),
        encoding="utf-8",
    )

    loaded, audit = load_bert_masked_lm(str(tmp_path))
    assert len(audit) == 1
    assert isinstance(loaded.bert.encoder.layer[0].attention.self, BertGqaSelfAttention)
    expected = model.bert.encoder.layer[0].attention.self.key.weight
    actual = loaded.bert.encoder.layer[0].attention.self.key.weight
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("attention_type", ["mqa", "collaborative"])
def test_saved_additional_baseline_checkpoint_round_trip(
    tmp_path, attention_type: str
) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=192,
    )
    model = transformers.BertForMaskedLM(config)
    replace_bert_self_attention(model, attention_type=attention_type)
    config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "gt_mha_state_dict.pt")
    (tmp_path / "bert_gt_mha_manifest.json").write_text(
        json.dumps(
            {
                "attention_type": attention_type,
                "num_kv_heads": 1 if attention_type == "mqa" else 4,
            }
        ),
        encoding="utf-8",
    )

    loaded, audit = load_bert_masked_lm(str(tmp_path))
    assert len(audit) == 1
    assert type(loaded.bert.encoder.layer[0].attention.self) is type(
        model.bert.encoder.layer[0].attention.self
    )
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(actual, expected)


def test_local_sequence_classifier_checkpoint_converts_to_gt_mha(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=101,
        hidden_size=96,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=192,
        num_labels=2,
    )
    original = transformers.BertForSequenceClassification(config)
    original.save_pretrained(tmp_path)
    converted, audit = load_bert_sequence_classifier(
        str(tmp_path),
        num_labels=2,
        attention_type="gt_mha_exact",
    )
    assert len(audit) == 1
    assert isinstance(converted.bert.encoder.layer[0].attention.self, BertGtMhaSelfAttention)
