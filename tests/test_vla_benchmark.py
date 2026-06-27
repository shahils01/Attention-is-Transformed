from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_vla import load_checkpoint, save_checkpoint
from lgma.vla_data import XVLAMetaDataset
from lgma.vla_model import VLAPolicyConfig, VLATransformerPolicy, ee6d_continuous_loss, ee6d_loss


def _write_libero_fixture(root: Path, length: int = 8) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    h5_path = data_dir / "episode_000000.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as f:
        action = rng.normal(size=(length, 10)).astype(np.float32)
        action[:, 9] = (rng.random(length) > 0.5).astype(np.float32)
        f.create_dataset("abs_action_6d", data=action)
        f.create_dataset("agentview_rgb", data=rng.integers(0, 255, size=(length, 24, 24, 3), dtype=np.uint8))
        f.create_dataset("eye_in_hand_rgb", data=rng.integers(0, 255, size=(length, 24, 24, 3), dtype=np.uint8))
        f.create_dataset("language_instruction", data=np.bytes_("pick up the block"))
    meta = {
        "dataset_name": "tiny_libero",
        "robot_type": "libero",
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
        "datalist": [str(h5_path)],
    }
    meta_path = root / "libero_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


def _write_encoded_libero_fixture(root: Path, length: int = 8) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    h5_path = data_dir / "episode_000000.hdf5"
    rng = np.random.default_rng(1)
    encoded = []
    for _ in range(length):
        image = Image.fromarray(rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
    max_len = max(x.size for x in encoded)
    padded = np.zeros((length, max_len), dtype=np.uint8)
    for idx, item in enumerate(encoded):
        padded[idx, : item.size] = item

    with h5py.File(h5_path, "w") as f:
        action = rng.normal(size=(length, 10)).astype(np.float32)
        action[:, 9] = (rng.random(length) > 0.5).astype(np.float32)
        f.create_dataset("abs_action_6d", data=action)
        f.create_dataset("agentview_rgb", data=padded)
        f.create_dataset("eye_in_hand_rgb", data=padded)
        f.create_dataset("language_instruction", data=np.bytes_("pick up the block"))
    meta = {
        "dataset_name": "tiny_libero_encoded",
        "robot_type": "libero",
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
        "datalist": [str(h5_path)],
    }
    meta_path = root / "libero_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


def _config(
    attention: str,
    num_base_heads: int = 1,
    value_transform: str = "none",
) -> VLAPolicyConfig:
    return VLAPolicyConfig(
        image_size=16,
        num_views=2,
        vocab_size=128,
        text_length=8,
        action_horizon=3,
        d_model=32,
        num_layers=1,
        num_heads=4,
        head_dim=8,
        attention=attention,  # type: ignore[arg-type]
        num_generators=2,
        num_base_heads=num_base_heads,
        value_transform=value_transform,
    )


@pytest.mark.parametrize(
    ("attention", "num_base_heads"),
    [
        ("mha", 1),
        ("shared_identity", 1),
        ("lgma", 1),
        ("lgma_multibase", 2),
        ("lgma_residual", 2),
        ("lgma_quad", 2),
        ("lgma_unconstrained", 2),
    ],
)
def test_vla_forward_backward_for_attention_variants(attention: str, num_base_heads: int):
    torch.manual_seed(0)
    model = VLATransformerPolicy(_config(attention, num_base_heads=num_base_heads))
    batch = {
        "image_input": torch.randn(2, 2, 3, 16, 16),
        "image_mask": torch.ones(2, 2, dtype=torch.bool),
        "text_token_ids": torch.randint(0, 128, (2, 8)),
        "proprio": torch.randn(2, 20),
        "action": torch.randn(2, 3, 20),
    }
    batch["action"][..., (9, 19)] = torch.randint(0, 2, (2, 3, 2)).float()
    pred = model(
        image_input=batch["image_input"],
        image_mask=batch["image_mask"],
        text_token_ids=batch["text_token_ids"],
        proprio=batch["proprio"],
    )
    assert pred.shape == (2, 3, 20)
    loss = sum(ee6d_loss(pred, batch["action"]).values())
    loss.backward()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


@pytest.mark.parametrize(
    ("attention", "value_transform"),
    [
        ("lgma", "lie"),
        ("lgma_residual", "lie"),
        ("lgma_quad", "lie"),
        ("lgma_unconstrained", "lie"),
    ],
)
def test_vla_forward_backward_for_value_lie_variants(attention: str, value_transform: str):
    torch.manual_seed(0)
    model = VLATransformerPolicy(
        _config(attention, num_base_heads=2, value_transform=value_transform)
    )
    batch = {
        "image_input": torch.randn(2, 2, 3, 16, 16),
        "image_mask": torch.ones(2, 2, dtype=torch.bool),
        "text_token_ids": torch.randint(0, 128, (2, 8)),
        "proprio": torch.randn(2, 20),
        "action": torch.randn(2, 3, 20),
    }
    batch["action"][..., (9, 19)] = torch.randint(0, 2, (2, 3, 2)).float()
    pred = model(
        image_input=batch["image_input"],
        image_mask=batch["image_mask"],
        text_token_ids=batch["text_token_ids"],
        proprio=batch["proprio"],
    )
    assert pred.shape == (2, 3, 20)
    loss = sum(ee6d_loss(pred, batch["action"]).values())
    loss.backward()
    first_attn = model.attention_modules[0]
    assert getattr(first_attn, "value_transform_mode") in {
        "exp",
        "residual",
        "quadratic",
        "unconstrained",
    }
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def test_vla_flow_action_head_forward_backward():
    torch.manual_seed(0)
    config = _config("mha")
    config.action_head = "flow"
    config.flow_sampling_steps = 2
    model = VLATransformerPolicy(config)
    batch = {
        "image_input": torch.randn(2, 2, 3, 16, 16),
        "image_mask": torch.ones(2, 2, dtype=torch.bool),
        "text_token_ids": torch.randint(0, 128, (2, 8)),
        "proprio": torch.randn(2, 20),
        "action": torch.randn(2, 3, 20),
    }
    timestep = torch.rand(2)
    noise = torch.randn_like(batch["action"])
    mix = timestep.view(-1, 1, 1)
    noisy_action = (1.0 - mix) * noise + mix * batch["action"]
    target_velocity = batch["action"] - noise
    pred_velocity = model(
        image_input=batch["image_input"],
        image_mask=batch["image_mask"],
        text_token_ids=batch["text_token_ids"],
        proprio=batch["proprio"],
        noisy_action=noisy_action,
        flow_timestep=timestep,
    )
    assert pred_velocity.shape == (2, 3, 20)
    loss = sum(ee6d_continuous_loss(pred_velocity, target_velocity).values())
    loss.backward()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def test_vla_flow_action_head_samples_actions():
    torch.manual_seed(0)
    config = _config("mha")
    config.action_head = "flow"
    config.flow_sampling_steps = 2
    model = VLATransformerPolicy(config)
    pred = model(
        image_input=torch.randn(2, 2, 3, 16, 16),
        image_mask=torch.ones(2, 2, dtype=torch.bool),
        text_token_ids=torch.randint(0, 128, (2, 8)),
        proprio=torch.randn(2, 20),
    )
    assert pred.shape == (2, 3, 20)
    assert torch.isfinite(pred).all()


def test_invalid_multibase_head_count_raises():
    cfg = _config("lgma_residual", num_base_heads=3)
    with pytest.raises(ValueError, match="divisible"):
        VLATransformerPolicy(cfg)


def test_xvla_meta_dataset_loads_tiny_libero(tmp_path: Path):
    meta_path = _write_libero_fixture(tmp_path)
    dataset = XVLAMetaDataset(
        meta_path,
        action_horizon=3,
        num_views=2,
        image_size=16,
        vocab_size=128,
        text_length=8,
    )
    sample = dataset[0]
    assert sample["image_input"].shape == (2, 3, 16, 16)
    assert sample["image_mask"].tolist() == [True, True]
    assert sample["text_token_ids"].shape == (8,)
    assert sample["proprio"].shape == (20,)
    assert sample["action"].shape == (3, 20)


def test_xvla_meta_dataset_decodes_encoded_image_bytes(tmp_path: Path):
    meta_path = _write_encoded_libero_fixture(tmp_path)
    dataset = XVLAMetaDataset(
        meta_path,
        action_horizon=3,
        num_views=2,
        image_size=16,
        vocab_size=128,
        text_length=8,
    )
    sample = dataset[0]
    assert sample["image_input"].shape == (2, 3, 16, 16)
    assert torch.isfinite(sample["image_input"]).all()


def test_vla_train_smoke_two_steps(tmp_path: Path):
    meta_path = _write_libero_fixture(tmp_path / "dataset", length=8)
    output_dir = tmp_path / "run"
    dataset = XVLAMetaDataset(
        meta_path,
        action_horizon=3,
        num_views=2,
        image_size=16,
        vocab_size=128,
        text_length=8,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    config = _config("shared_identity")
    model = VLATransformerPolicy(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = 0
    for batch in loader:
        pred = model(
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
            text_token_ids=batch["text_token_ids"],
            proprio=batch["proprio"],
        )
        loss = sum(ee6d_loss(pred, batch["action"]).values())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        if steps == 2:
            break
    assert steps == 2
    save_checkpoint(model, optimizer, config, output_dir, steps)
    assert (output_dir / "checkpoint_step_2" / "checkpoint.pt").exists()


def test_vla_checkpoint_resume_restores_model_and_optimizer(tmp_path: Path):
    config = _config("mha")
    model = VLATransformerPolicy(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "image_input": torch.randn(2, 2, 3, 16, 16),
        "image_mask": torch.ones(2, 2, dtype=torch.bool),
        "text_token_ids": torch.randint(0, 128, (2, 8)),
        "proprio": torch.randn(2, 20),
        "action": torch.randn(2, 3, 20),
    }
    loss = sum(
        ee6d_loss(
            model(
                image_input=batch["image_input"],
                image_mask=batch["image_mask"],
                text_token_ids=batch["text_token_ids"],
                proprio=batch["proprio"],
            ),
            batch["action"],
        ).values()
    )
    loss.backward()
    optimizer.step()
    save_checkpoint(model, optimizer, config, tmp_path, step=7)

    resumed = VLATransformerPolicy(config)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    step = load_checkpoint(
        tmp_path / "checkpoint_step_7",
        resumed,
        resumed_optimizer,
        scaler=None,
        config=config,
        device=torch.device("cpu"),
    )

    assert step == 7
    for original, loaded in zip(model.parameters(), resumed.parameters()):
        assert torch.allclose(original, loaded)
    assert resumed_optimizer.state_dict()["state"]


def test_vla_checkpoint_resume_rejects_config_mismatch(tmp_path: Path):
    config = _config("mha")
    model = VLATransformerPolicy(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, config, tmp_path, step=3)

    mismatched = _config("mha")
    mismatched.dropout = 0.2
    mismatched_model = VLATransformerPolicy(mismatched)
    with pytest.raises(ValueError, match="config does not match"):
        load_checkpoint(
            tmp_path / "checkpoint_step_3" / "checkpoint.pt",
            mismatched_model,
            torch.optim.AdamW(mismatched_model.parameters(), lr=1e-3),
            scaler=None,
            config=mismatched,
            device=torch.device("cpu"),
        )


def test_vla_checkpoint_resume_can_reset_action_head(tmp_path: Path):
    config = _config("mha")
    model = VLATransformerPolicy(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, config, tmp_path, step=5)

    flow_config = _config("mha")
    flow_config.action_head = "flow"
    flow_config.flow_sampling_steps = 2
    flow_model = VLATransformerPolicy(flow_config)
    flow_optimizer = torch.optim.AdamW(flow_model.parameters(), lr=1e-3)
    step = load_checkpoint(
        tmp_path / "checkpoint_step_5",
        flow_model,
        flow_optimizer,
        scaler=None,
        config=flow_config,
        device=torch.device("cpu"),
        reset_action_head=True,
    )

    assert step == 5
    assert not flow_optimizer.state_dict()["state"]
    assert torch.allclose(model.text_embedding.weight, flow_model.text_embedding.weight)
