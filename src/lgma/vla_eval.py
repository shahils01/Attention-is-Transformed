from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Literal

import numpy as np
import torch

from lgma.vla_data import _load_image_tensor, hash_instruction
from lgma.vla_model import VLAPolicyConfig, VLATransformerPolicy

Rot6DLayout = Literal["row", "column"]


def load_vla_policy(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[VLATransformerPolicy, VLAPolicyConfig]:
    payload = torch.load(checkpoint, map_location="cpu")
    config = VLAPolicyConfig(**payload["config"])
    model = VLATransformerPolicy(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, config


def euler_xyz_to_rot6d(euler: np.ndarray) -> np.ndarray:
    e = np.asarray(euler, dtype=np.float64).reshape(-1, 3)
    cx, cy, cz = np.cos(e[:, 0]), np.cos(e[:, 1]), np.cos(e[:, 2])
    sx, sy, sz = np.sin(e[:, 0]), np.sin(e[:, 1]), np.sin(e[:, 2])
    rot = np.empty((e.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = cy * cz
    rot[:, 0, 1] = -cy * sz
    rot[:, 0, 2] = sy
    rot[:, 1, 0] = sx * sy * cz + cx * sz
    rot[:, 1, 1] = -sx * sy * sz + cx * cz
    rot[:, 1, 2] = -sx * cy
    rot[:, 2, 0] = -cx * sy * cz + sx * sz
    rot[:, 2, 1] = cx * sy * sz + sx * cz
    rot[:, 2, 2] = cx * cy
    out = rot[:, :, :2].reshape(e.shape[0], 6).astype(np.float32)
    return out[0] if np.asarray(euler).ndim == 1 else out


def rotmat_to_rot6d(rot: np.ndarray, layout: Rot6DLayout = "row") -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    if layout not in {"row", "column"}:
        raise ValueError("layout must be 'row' or 'column'")
    if r.shape == (3, 3):
        if layout == "column":
            return np.concatenate([r[:, 0], r[:, 1]], axis=0).astype(np.float32)
        return r[:, :2].reshape(6).astype(np.float32)
    if layout == "column":
        return np.concatenate([r[:, :, 0], r[:, :, 1]], axis=-1).astype(np.float32)
    return r[:, :, :2].reshape(r.shape[0], 6).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray, layout: Rot6DLayout = "row") -> np.ndarray:
    if layout not in {"row", "column"}:
        raise ValueError("layout must be 'row' or 'column'")
    r6 = np.asarray(rot6d, dtype=np.float64)
    single = r6.ndim == 1
    r6 = r6.reshape(-1, 6)
    if layout == "column":
        a1 = r6[:, :3]
        a2 = r6[:, 3:6]
    else:
        a1 = r6[:, 0:5:2]
        a2 = r6[:, 1:6:2]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    mat = np.stack((b1, b2, b3), axis=-1).astype(np.float32)
    return mat[0] if single else mat


def rot6d_to_axis_angle(rot6d: np.ndarray, layout: Rot6DLayout = "row") -> np.ndarray:
    mat = rot6d_to_matrix(rot6d, layout=layout)
    single = mat.ndim == 2
    mats = mat[None] if single else mat
    output = []
    for rot in mats:
        trace = float(np.trace(rot))
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        angle = math.acos(cos_angle)
        if angle < 1e-7:
            output.append(np.zeros(3, dtype=np.float32))
            continue
        denom = 2.0 * math.sin(angle)
        axis = np.array(
            [
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ],
            dtype=np.float64,
        ) / denom
        output.append((axis * angle).astype(np.float32))
    arr = np.stack(output, axis=0)
    return arr[0] if single else arr


def rot6d_to_quat_xyzw(rot6d: np.ndarray, layout: Rot6DLayout = "row") -> np.ndarray:
    mat = rot6d_to_matrix(rot6d, layout=layout)
    single = mat.ndim == 2
    mats = mat[None] if single else mat
    quats = []
    for rot in mats:
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(rot)))
            if idx == 0:
                s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                qw = (rot[2, 1] - rot[1, 2]) / s
                qx = 0.25 * s
                qy = (rot[0, 1] + rot[1, 0]) / s
                qz = (rot[0, 2] + rot[2, 0]) / s
            elif idx == 1:
                s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                qw = (rot[0, 2] - rot[2, 0]) / s
                qx = (rot[0, 1] + rot[1, 0]) / s
                qy = 0.25 * s
                qz = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                qw = (rot[1, 0] - rot[0, 1]) / s
                qx = (rot[0, 2] + rot[2, 0]) / s
                qy = (rot[1, 2] + rot[2, 1]) / s
                qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw], dtype=np.float32)
        quat = quat / np.clip(np.linalg.norm(quat), 1e-8, None)
        quats.append(quat)
    arr = np.stack(quats, axis=0)
    return arr[0] if single else arr


def single_arm_to_proprio20(left_state: np.ndarray) -> np.ndarray:
    left = np.asarray(left_state, dtype=np.float32)
    if left.shape != (10,):
        raise ValueError(f"left_state must have shape (10,), got {left.shape}")
    return np.concatenate([left, np.zeros_like(left)], axis=0).astype(np.float32)


class VLAActionPlanner:
    """Checkpoint-backed policy wrapper that queues predicted action horizons."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
        action_chunk: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.config = load_vla_policy(checkpoint, self.device)
        self.action_chunk = action_chunk
        self.plan: Deque[np.ndarray] = deque()
        self.proprio: np.ndarray | None = None

    def reset(self, proprio: np.ndarray | None = None) -> None:
        self.plan.clear()
        self.proprio = None if proprio is None else np.asarray(proprio, dtype=np.float32).copy()

    @torch.no_grad()
    def predict_plan(
        self,
        images: Iterable[np.ndarray],
        language_instruction: str,
        proprio: np.ndarray,
    ) -> np.ndarray:
        image_tensors = [
            _load_image_tensor(np.asarray(image), self.config.image_size)
            for image in list(images)[: self.config.num_views]
        ]
        if not image_tensors:
            raise ValueError("at least one image is required")
        valid_views = len(image_tensors)
        while len(image_tensors) < self.config.num_views:
            image_tensors.append(torch.zeros_like(image_tensors[0]))
        image_mask = torch.zeros(self.config.num_views, dtype=torch.bool)
        image_mask[: min(valid_views, self.config.num_views)] = True

        batch = {
            "image_input": torch.stack(image_tensors, dim=0).unsqueeze(0).to(self.device),
            "image_mask": image_mask.unsqueeze(0).to(self.device),
            "text_token_ids": hash_instruction(
                language_instruction,
                self.config.vocab_size,
                self.config.text_length,
            )
            .unsqueeze(0)
            .to(self.device),
            "proprio": torch.from_numpy(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(self.device),
        }
        pred = self.model(**batch).squeeze(0).float().cpu().numpy()
        pred[:, [9, 19]] = 1.0 / (1.0 + np.exp(-pred[:, [9, 19]]))
        return pred

    def next_action_plan(
        self,
        images: Iterable[np.ndarray],
        language_instruction: str,
        proprio: np.ndarray,
    ) -> np.ndarray:
        if not self.plan:
            plan = self.predict_plan(images, language_instruction, proprio)
            if self.action_chunk is not None:
                plan = plan[: self.action_chunk]
            for row in plan:
                self.plan.append(row.astype(np.float32))
        action = self.plan.popleft()
        self.proprio = action.copy()
        return action
