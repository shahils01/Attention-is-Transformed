from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_vla import save_checkpoint
from lgma.vla_eval import (
    VLAActionPlanner,
    euler_xyz_to_rot6d,
    rot6d_to_axis_angle,
    rot6d_to_matrix,
    rot6d_to_quat_xyzw,
    single_arm_to_proprio20,
)
from lgma.vla_model import VLAPolicyConfig, VLATransformerPolicy


def _tiny_config() -> VLAPolicyConfig:
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
        attention="mha",
    )


def test_rotation_conversion_identity_shapes():
    rot6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mat = rot6d_to_matrix(rot6d)
    axis_angle = rot6d_to_axis_angle(rot6d)
    quat = rot6d_to_quat_xyzw(rot6d)
    assert mat.shape == (3, 3)
    assert np.allclose(mat, np.eye(3), atol=1e-5)
    assert axis_angle.shape == (3,)
    assert np.allclose(axis_angle, np.zeros(3), atol=1e-5)
    assert quat.shape == (4,)
    assert np.allclose(quat, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5)


def test_euler_and_proprio_helpers():
    rot6d = euler_xyz_to_rot6d(np.zeros(3, dtype=np.float32))
    proprio = single_arm_to_proprio20(np.concatenate([np.zeros(3), rot6d, np.array([1.0])]))
    assert rot6d.shape == (6,)
    assert proprio.shape == (20,)
    assert np.allclose(proprio[10:], 0.0)


def test_action_planner_loads_checkpoint_and_pads_missing_view(tmp_path: Path):
    config = _tiny_config()
    model = VLATransformerPolicy(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, config, tmp_path, step=1)
    planner = VLAActionPlanner(tmp_path / "checkpoint_step_1" / "checkpoint.pt", device="cpu")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    proprio = np.zeros(20, dtype=np.float32)
    plan = planner.predict_plan([image], "pick up block", proprio)
    assert plan.shape == (3, 20)
    assert np.isfinite(plan).all()
    assert np.all((0.0 <= plan[:, [9, 19]]) & (plan[:, [9, 19]] <= 1.0))
