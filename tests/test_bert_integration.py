from __future__ import annotations

from types import SimpleNamespace
import json

import torch
from torch import nn
import pytest

from lgma.bert import (
    BertGtMhaConfig,
    BertGtMhaSelfAttention,
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
