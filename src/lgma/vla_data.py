from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class VLAMetaEntry:
    path: str
    robot_type: str
    observation_key: tuple[str, ...]
    language_instruction_key: str


def hash_instruction(text: str, vocab_size: int, text_length: int) -> torch.LongTensor:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    words = text.lower().replace("_", " ").split()
    token_ids = torch.zeros(text_length, dtype=torch.long)
    for i, word in enumerate(words[:text_length]):
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
        token_ids[i] = int.from_bytes(digest, byteorder="little") % (vocab_size - 1) + 1
    return token_ids


def load_meta_entries(metas_path: str | Path) -> list[VLAMetaEntry]:
    path = Path(metas_path)
    if path.is_dir():
        meta_paths = sorted(path.rglob("*.json"))
    else:
        meta_paths = [path]
    if not meta_paths:
        raise FileNotFoundError(f"No meta JSON files found at {path}")

    entries: list[VLAMetaEntry] = []
    for meta_path in meta_paths:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if "datalist" not in meta:
            raise ValueError(f"Meta file {meta_path} is missing `datalist`")
        robot_type = str(meta.get("robot_type") or meta.get("dataset_name"))
        if robot_type not in {"libero", "Calvin"}:
            raise ValueError(f"Unsupported robot_type `{robot_type}` in {meta_path}")
        observation_key = tuple(meta.get("observation_key") or _default_observation_keys(robot_type))
        language_key = str(meta.get("language_instruction_key", "language_instruction"))
        for item in meta["datalist"]:
            if isinstance(item, dict):
                item_path = item.get("path")
            else:
                item_path = item
            if not item_path:
                raise ValueError(f"Invalid datalist item in {meta_path}: {item!r}")
            entries.append(
                VLAMetaEntry(
                    path=str(item_path),
                    robot_type=robot_type,
                    observation_key=observation_key,
                    language_instruction_key=language_key,
                )
            )
    return entries


def _default_observation_keys(robot_type: str) -> tuple[str, ...]:
    if robot_type == "libero":
        return ("agentview_rgb", "eye_in_hand_rgb")
    if robot_type == "Calvin":
        return ("rgb_static", "rgb_gripper")
    raise ValueError(f"Unsupported robot_type `{robot_type}`")


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_string(value.item())
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="ignore")
    return str(value)


def _read_instruction(f: h5py.File, key: str, fallback: str) -> str:
    if key in f:
        return _decode_string(f[key][()])
    return fallback


def _axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    aa = axis_angle.astype(np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.clip(theta, 1e-12, None)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    k = np.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros],
        axis=1,
    ).reshape(-1, 3, 3)
    eye = np.eye(3, dtype=np.float64)[None, :, :]
    sin = np.sin(theta)[:, None]
    cos = np.cos(theta)[:, None]
    rot = eye + sin * k + (1.0 - cos) * (k @ k)
    small = theta[:, 0] < 1e-8
    if np.any(small):
        rot[small] = np.eye(3, dtype=np.float64)
    return _rotmat_to_rot6d(rot).astype(np.float32)


def _euler_xyz_to_rot6d(euler: np.ndarray) -> np.ndarray:
    e = euler.astype(np.float64)
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
    return _rotmat_to_rot6d(rot).astype(np.float32)


def _rotmat_to_rot6d(rot: np.ndarray) -> np.ndarray:
    return rot[:, :, :2].reshape(rot.shape[0], 6)


def _pad_single_arm(left: np.ndarray) -> np.ndarray:
    right = np.zeros_like(left)
    return np.concatenate([left, right], axis=-1).astype(np.float32)


def _libero_trajectory(f: h5py.File) -> np.ndarray:
    if "abs_action_6d" in f:
        left = f["abs_action_6d"][()].astype(np.float32)
        left = left.copy()
        left[:, 9:10] = (left[:, 9:10] > 0.0).astype(np.float32)
        return _pad_single_arm(left)
    if "proprio" in f:
        proprio = f["proprio"][()].astype(np.float32)
        if proprio.shape[-1] == 10:
            return _pad_single_arm(proprio)
    raise KeyError("LIBERO HDF5 requires `abs_action_6d` or 10D `proprio`")


def _calvin_trajectory(f: h5py.File) -> np.ndarray:
    proprio = f["proprio"][()].astype(np.float32)
    if proprio.shape[-1] < 7:
        raise ValueError(f"CALVIN proprio must have at least 7 dims, got {proprio.shape}")
    grip = (proprio[:, 6:7] < 0.0).astype(np.float32)
    left = np.concatenate([proprio[:, :3], _euler_xyz_to_rot6d(proprio[:, 3:6]), grip], axis=-1)
    return _pad_single_arm(left)


def _trajectory_for_entry(f: h5py.File, entry: VLAMetaEntry) -> np.ndarray:
    if entry.robot_type == "libero":
        return _libero_trajectory(f)
    if entry.robot_type == "Calvin":
        return _calvin_trajectory(f)
    raise ValueError(f"Unsupported robot_type `{entry.robot_type}`")


def _image_count(f: h5py.File, entry: VLAMetaEntry) -> int:
    for key in entry.observation_key:
        if key in f:
            return int(f[key].shape[0])
    raise KeyError(f"No observation keys {entry.observation_key} found in {entry.path}")


def _load_image_tensor(array: np.ndarray, image_size: int) -> torch.Tensor:
    image = np.asarray(array)
    if image.ndim != 3:
        raise ValueError(f"image must be [H,W,C], got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        tensor = torch.from_numpy(image).float()
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class XVLAMetaDataset(Dataset):
    """Finite X-VLA-format HDF5 dataset for LIBERO/CALVIN attention ablations."""

    def __init__(
        self,
        metas_path: str | Path,
        action_horizon: int = 10,
        num_views: int = 2,
        image_size: int = 128,
        vocab_size: int = 4096,
        text_length: int = 32,
        stride: int = 1,
        max_episodes: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        self.entries = load_meta_entries(metas_path)
        if max_episodes is not None:
            self.entries = self.entries[:max_episodes]
        self.action_horizon = int(action_horizon)
        self.num_views = int(num_views)
        self.image_size = int(image_size)
        self.vocab_size = int(vocab_size)
        self.text_length = int(text_length)
        self.index: list[tuple[int, int]] = []
        for entry_idx, entry in enumerate(self.entries):
            with h5py.File(entry.path, "r") as f:
                traj_len = _trajectory_for_entry(f, entry).shape[0]
                image_len = _image_count(f, entry)
            usable = min(traj_len, image_len) - self.action_horizon
            if usable <= 0:
                continue
            for start in range(0, usable, stride):
                self.index.append((entry_idx, start))
                if max_samples is not None and len(self.index) >= max_samples:
                    return
        if not self.index:
            raise ValueError(f"No usable samples found in {metas_path}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry_idx, start = self.index[idx]
        entry = self.entries[entry_idx]
        with h5py.File(entry.path, "r") as f:
            traj = _trajectory_for_entry(f, entry)
            fallback = Path(entry.path).stem.replace("_", " ")
            instruction = _read_instruction(f, entry.language_instruction_key, fallback)
            images = []
            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            for view_idx, key in enumerate(entry.observation_key[: self.num_views]):
                if key not in f:
                    continue
                images.append(_load_image_tensor(f[key][start], self.image_size))
                image_mask[view_idx] = True
            if not images:
                raise KeyError(f"No requested image views found in {entry.path}")
            while len(images) < self.num_views:
                images.append(torch.zeros_like(images[0]))

        proprio = torch.from_numpy(traj[start].astype(np.float32))
        action = torch.from_numpy(traj[start + 1 : start + 1 + self.action_horizon].astype(np.float32))
        return {
            "image_input": torch.stack(images, dim=0),
            "image_mask": image_mask,
            "text_token_ids": hash_instruction(instruction, self.vocab_size, self.text_length),
            "proprio": proprio,
            "action": action,
        }
