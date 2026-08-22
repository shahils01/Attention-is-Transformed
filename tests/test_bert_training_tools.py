from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_bert_mlm import (  # noqa: E402
    TrainEvalMLMCollator,
    fixed_validation_mlm_mask,
    optimizer_parameter_groups,
)
from lgma.bert import (  # noqa: E402
    BertGqaConfig,
    BertGqaSelfAttention,
    BertGtMhaConfig,
    BertGtMhaSelfAttention,
)


def test_bert_gt_mha_base_projections_use_bert_initializer() -> None:
    torch.manual_seed(0)
    module = BertGtMhaSelfAttention(
        BertGtMhaConfig(
            hidden_size=768,
            num_attention_heads=12,
            initializer_range=0.02,
        )
    )

    for projection in (module.query, module.key, module.value):
        assert abs(float(projection.weight.mean())) < 5e-4
        assert torch.isclose(
            projection.weight.std(), torch.tensor(0.02), rtol=0.03, atol=0.0
        )
        assert torch.count_nonzero(projection.bias) == 0


def test_bert_gqa_projections_use_bert_initializer() -> None:
    torch.manual_seed(0)
    module = BertGqaSelfAttention(
        BertGqaConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_kv_heads=4,
            initializer_range=0.02,
        )
    )

    for projection in (module.query, module.key, module.value):
        assert abs(float(projection.weight.mean())) < 8e-4
        assert torch.isclose(
            projection.weight.std(), torch.tensor(0.02), rtol=0.04, atol=0.0
        )
        assert torch.count_nonzero(projection.bias) == 0


def test_gqa_optimizer_has_no_head_coordinates() -> None:
    module = BertGqaSelfAttention(
        BertGqaConfig(hidden_size=96, num_attention_heads=12, num_kv_heads=4)
    )

    def all_parameter_names(model: nn.Module, _forbidden_layer_types):
        return [name for name, _ in model.named_parameters()]

    groups, coordinate_names = optimizer_parameter_groups(
        module,
        weight_decay=0.01,
        get_parameter_names=all_parameter_names,
    )
    parameter_decay = {
        id(parameter): group["weight_decay"]
        for group in groups
        for parameter in group["params"]
    }

    assert coordinate_names == []
    assert parameter_decay[id(module.query.weight)] == 0.01
    assert parameter_decay[id(module.key.weight)] == 0.01
    assert parameter_decay[id(module.value.weight)] == 0.01


def test_fixed_validation_mask_is_repeatable_and_preserves_special_tokens() -> None:
    batch = {
        "input_ids": [list(range(10)), list(range(10, 20))],
        "special_tokens_mask": [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    }
    kwargs = {
        "mask_token_id": 99,
        "vocab_size": 128,
        "mlm_probability": 0.5,
        "seed": 17_029,
    }

    first = fixed_validation_mlm_mask(batch, [4, 5], **kwargs)
    second = fixed_validation_mlm_mask(batch, [4, 5], **kwargs)
    changed_seed = fixed_validation_mlm_mask(batch, [4, 5], **{**kwargs, "seed": 17_030})

    assert first == second
    assert first != changed_seed
    for original, masked, labels in zip(batch["input_ids"], first["input_ids"], first["labels"]):
        assert masked[0] == original[0]
        assert masked[-1] == original[-1]
        assert labels[0] == -100
        assert labels[-1] == -100
        assert any(label != -100 for label in labels[1:-1])


def test_train_eval_collator_routes_precomputed_labels_to_fixed_collator() -> None:
    calls: list[str] = []

    def train_collator(features):
        calls.append("train")
        return {"route": torch.tensor(1)}

    def fixed_collator(features):
        calls.append("fixed")
        return {"route": torch.tensor(2)}

    collator = TrainEvalMLMCollator(train_collator, fixed_collator)
    assert int(collator([{"input_ids": [1, 2]}])["route"]) == 1
    assert int(collator([{"input_ids": [1, 2], "labels": [-100, 2]}])["route"]) == 2
    assert calls == ["train", "fixed"]


def test_head_coordinates_are_in_zero_weight_decay_group() -> None:
    module = BertGtMhaSelfAttention(
        BertGtMhaConfig(hidden_size=96, num_attention_heads=12)
    )

    def all_parameter_names(model: nn.Module, _forbidden_layer_types):
        return [name for name, _ in model.named_parameters()]

    groups, coordinate_names = optimizer_parameter_groups(
        module,
        weight_decay=0.01,
        get_parameter_names=all_parameter_names,
    )
    parameter_decay = {
        id(parameter): group["weight_decay"]
        for group in groups
        for parameter in group["params"]
    }

    assert coordinate_names
    for name, parameter in module.named_parameters():
        if name.endswith((".theta", ".value_theta")):
            assert parameter_decay[id(parameter)] == 0.0
    assert parameter_decay[id(module.query.weight)] == 0.01


def test_hugging_face_parameter_name_helper_keeps_coordinates_out_of_decay() -> None:
    pytest.importorskip("transformers")
    from transformers.trainer_pt_utils import get_parameter_names

    module = BertGtMhaSelfAttention(
        BertGtMhaConfig(hidden_size=96, num_attention_heads=12)
    )
    groups, coordinate_names = optimizer_parameter_groups(
        module,
        weight_decay=0.01,
        get_parameter_names=get_parameter_names,
    )
    parameter_decay = {
        id(parameter): group["weight_decay"]
        for group in groups
        for parameter in group["params"]
    }

    assert coordinate_names == ["gt_attention.theta", "gt_attention.value_theta"]
    assert parameter_decay[id(module.gt_attention.theta)] == 0.0
    assert parameter_decay[id(module.gt_attention.value_theta)] == 0.0
    assert parameter_decay[id(module.query.weight)] == 0.01
