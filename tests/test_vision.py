from __future__ import annotations

from pathlib import Path
import sys
import json

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lgma.baselines import ReducedDimMultiheadAttention
from lgma.vision import (
    VISION_ATTENTION_TYPES,
    DeiTClassifier,
    DeiTConfig,
    vision_parameter_counts,
)


def test_webdataset_rejects_repeat_augmentation() -> None:
    from argparse import Namespace

    from experiments.train_imagenet_deit import create_data_loaders

    args = Namespace(dataset_format="wds", repeated_augmentation=3)
    with pytest.raises(ValueError, match="requires an indexable dataset"):
        create_data_loaders(args, world_size=1, device=torch.device("cpu"))


def test_exact_wds_validation_resets_distributed_reader() -> None:
    from types import SimpleNamespace

    from experiments.train_imagenet_deit import configure_exact_wds_validation

    reader = SimpleNamespace(
        ds=None,
        dist_rank=2,
        dist_num_replicas=4,
        global_worker_id=2,
        global_num_workers=48,
    )
    configure_exact_wds_validation(SimpleNamespace(reader=reader))

    assert reader.dist_rank == 0
    assert reader.dist_num_replicas == 1
    assert reader.global_worker_id == 0
    assert reader.global_num_workers == 1


def test_wds_split_rejects_class_homogeneous_training_shards(tmp_path: Path) -> None:
    from experiments.train_imagenet_deit import _wds_split

    manifest = {
        "format": "webdataset",
        "num_classes": 1000,
        "train_order": "deterministic_random_shuffle",
        "splits": {
            "train": {
                "pattern": "train-{00000..00000}.tar",
                "samples": 1024,
                "shards": [
                    {"name": "train-00000.tar", "samples": 1024, "unique_classes": 1}
                ],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="insufficiently class-mixed"):
        _wds_split(tmp_path, "train")


def tiny_config(attention_type: str) -> DeiTConfig:
    return DeiTConfig(
        image_size=32,
        patch_size=8,
        num_classes=10,
        embed_dim=64,
        depth=1,
        num_heads=4,
        reduced_qk_dim=32,
        collaborative_qk_dim=32,
        num_kv_heads=2,
        num_base_heads=2,
        num_generators=2,
        attention_type=attention_type,
        use_sdpa=False,
    )


@pytest.mark.parametrize("attention_type", sorted(VISION_ATTENTION_TYPES))
def test_deit_attention_variants_forward_backward(attention_type: str) -> None:
    model = DeiTClassifier(tiny_config(attention_type))
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)

    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert model.patch_embed.proj.weight.grad is not None


def test_reduced_mha_only_reduces_query_and_key_dimensions() -> None:
    attention = ReducedDimMultiheadAttention(
        d_model=64,
        num_heads=4,
        qk_head_dim=8,
        value_head_dim=16,
        bias=True,
    )

    assert attention.q_proj.out_features == 32
    assert attention.k_proj.out_features == 32
    assert attention.v_proj.out_features == 64
    assert attention.out_proj.in_features == 64
    assert attention(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_controlled_parameter_ordering() -> None:
    mha = vision_parameter_counts(DeiTClassifier(tiny_config("mha")))
    reduced = vision_parameter_counts(DeiTClassifier(tiny_config("reduced_mha")))
    gt = vision_parameter_counts(DeiTClassifier(tiny_config("gt_mha_residual")))

    assert reduced["non_attention_parameters"] == mha["non_attention_parameters"]
    assert gt["non_attention_parameters"] == mha["non_attention_parameters"]
    assert gt["attention_parameters"] < reduced["attention_parameters"]
    assert reduced["attention_parameters"] < mha["attention_parameters"]


def test_deit_rejects_invalid_attention_counts() -> None:
    with pytest.raises(ValueError, match="positive"):
        DeiTConfig(num_base_heads=0)


def test_deit_rejects_wrong_image_size_at_runtime() -> None:
    model = DeiTClassifier(tiny_config("mha"))
    with pytest.raises(ValueError, match="expected 32x32"):
        model(torch.randn(1, 3, 24, 32))


def test_gt_generator_no_decay_ablation_is_opt_in() -> None:
    from experiments.train_imagenet_deit import optimizer_groups

    model = DeiTClassifier(tiny_config("gt_mha_residual"))

    default_groups = optimizer_groups(model, 0.05)
    ablation_groups = optimizer_groups(
        model,
        0.05,
        no_decay_gt_generators=True,
    )
    default_decay = {
        id(parameter): group["weight_decay"]
        for group in default_groups
        for parameter in group["params"]
    }
    ablation_decay = {
        id(parameter): group["weight_decay"]
        for group in ablation_groups
        for parameter in group["params"]
    }

    attention = model.blocks[0].attn
    assert default_decay[id(attention.generators)] == 0.05
    assert default_decay[id(attention.value_generators)] == 0.05
    assert ablation_decay[id(attention.generators)] == 0.0
    assert ablation_decay[id(attention.value_generators)] == 0.0
    assert ablation_decay[id(attention.theta)] == 0.0
    assert ablation_decay[id(attention.value_theta)] == 0.0
    assert ablation_decay[id(attention.q_proj.weight)] == 0.05
    assert ablation_decay[id(attention.k_proj.weight)] == 0.05
    assert ablation_decay[id(attention.v_proj.weight)] == 0.05
