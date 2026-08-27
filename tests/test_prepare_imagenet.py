from __future__ import annotations

import json
import io
from pathlib import Path
import sys
import tarfile
import zipfile
from collections import Counter

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prepare_imagenet_kaggle import TRAIN_PREFIX, VAL_PREFIX, prepare
from experiments.prepare_imagenet_shared import prepare_shared


def _write_fake_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ILSVRC/Data/CLS-LOC/train/n00000001/a.JPEG", b"train-a")
        archive.writestr("ILSVRC/Data/CLS-LOC/train/n00000002/b.JPEG", b"train-b")
        archive.writestr("ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG", b"val-a")
        archive.writestr(
            "LOC_val_solution.csv",
            "ImageId,PredictionString\n"
            "ILSVRC2012_val_00000001,n00000002 0 0 1 1\n",
        )
        archive.writestr("ILSVRC/Annotations/CLS-LOC/train/ignored.xml", b"ignored")


def _labels_in_shard(path: Path) -> list[int]:
    labels: list[int] = []
    with tarfile.open(path) as shard:
        for member in shard.getmembers():
            if member.name.endswith(".cls"):
                payload = shard.extractfile(member)
                assert payload is not None
                labels.append(int(payload.read()))
    return labels


def test_prepare_imagenet_archive_as_webdataset(tmp_path: Path) -> None:
    archive_path = tmp_path / "imagenet.zip"
    output_dir = tmp_path / "wds"
    _write_fake_archive(archive_path)

    manifest = prepare(
        archive_path,
        output_dir,
        samples_per_shard=1,
        max_train_samples=2,
        max_val_samples=1,
    )

    assert manifest["num_classes"] == 2
    assert manifest["train_order"] == "deterministic_random_shuffle"
    assert manifest["train_shuffle_seed"] == 0
    assert manifest["splits"]["train"]["samples"] == 2
    assert all(
        shard["unique_classes"] == 1
        for shard in manifest["splits"]["train"]["shards"]
    )
    assert manifest["splits"]["val"]["samples"] == 1
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest
    timm_info = json.loads((output_dir / "_info.json").read_text())
    assert timm_info["splits"]["train"]["shard_lengths"] == [1, 1]

    with tarfile.open(output_dir / "val-00000.tar") as shard:
        names = set(shard.getnames())
        assert "ILSVRC2012_val_00000001.jpg" in names
        assert "ILSVRC2012_val_00000001.cls" in names
        label = shard.extractfile("ILSVRC2012_val_00000001.cls")
        assert label is not None
        assert label.read() == b"1"


def test_prepare_reuses_complete_shards(tmp_path: Path) -> None:
    archive_path = tmp_path / "imagenet.zip"
    output_dir = tmp_path / "wds"
    _write_fake_archive(archive_path)
    kwargs = {
        "samples_per_shard": 2,
        "max_train_samples": 2,
        "max_val_samples": 1,
    }
    prepare(archive_path, output_dir, **kwargs)
    train_shard = output_dir / "train-00000.tar"
    original_mtime = train_shard.stat().st_mtime_ns

    prepare(archive_path, output_dir, **kwargs)

    assert train_shard.stat().st_mtime_ns == original_mtime


def test_prepare_rejects_incompatible_existing_shard_protocol(tmp_path: Path) -> None:
    archive_path = tmp_path / "imagenet.zip"
    output_dir = tmp_path / "wds"
    _write_fake_archive(archive_path)
    prepare(
        archive_path,
        output_dir,
        samples_per_shard=2,
        max_train_samples=2,
        max_val_samples=1,
        train_shuffle_seed=3,
    )

    with pytest.raises(FileExistsError, match="existing shard protocol is incompatible"):
        prepare(
            archive_path,
            output_dir,
            samples_per_shard=2,
            max_train_samples=2,
            max_val_samples=1,
            train_shuffle_seed=4,
        )


def test_training_shards_are_deterministically_mixed_across_classes(tmp_path: Path) -> None:
    archive_path = tmp_path / "imagenet.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for class_index in range(8):
            wnid = f"n{class_index:08d}"
            for image_index in range(16):
                archive.writestr(
                    f"{TRAIN_PREFIX}{wnid}/{wnid}_{image_index:04d}.JPEG",
                    f"{class_index}-{image_index}".encode(),
                )
        archive.writestr(
            f"{VAL_PREFIX}ILSVRC2012_val_00000001.JPEG", b"validation"
        )
        archive.writestr(
            "LOC_val_solution.csv",
            "ImageId,PredictionString\n"
            "ILSVRC2012_val_00000001,n00000000 0 0 1 1\n",
        )

    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "samples_per_shard": 32,
        "max_train_samples": 128,
        "max_val_samples": 1,
        "train_shuffle_seed": 17,
    }
    prepare(archive_path, first, **kwargs)
    prepare(archive_path, second, **kwargs)

    first_labels = _labels_in_shard(first / "train-00000.tar")
    second_labels = _labels_in_shard(second / "train-00000.tar")
    assert first_labels == second_labels
    assert len(Counter(first_labels)) >= 6
    first_manifest = json.loads((first / "manifest.json").read_text())
    assert first_manifest["splits"]["train"]["shards"][0]["unique_classes"] >= 6


def test_prepare_shared_filters_official_images_and_falls_back(tmp_path: Path) -> None:
    archive_path = tmp_path / "imagenet.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for wnid, filename, payload in (
            ("n00000001", "a.JPEG", b"archive-a"),
            ("n00000002", "b.JPEG", b"archive-b"),
            ("n00000003", "c.JPEG", b"archive-c"),
        ):
            archive.writestr(f"{TRAIN_PREFIX}{wnid}/{filename}", payload)
        archive.writestr(
            f"{VAL_PREFIX}ILSVRC2012_val_00000001.JPEG", b"validation"
        )
        archive.writestr(
            "LOC_val_solution.csv",
            "ImageId,PredictionString\n"
            "ILSVRC2012_val_00000001,n00000001 0 0 1 1\n",
        )

    shared = tmp_path / "shared"
    shared.mkdir()
    with tarfile.open(shared / "n00000001.tar", "w") as source:
        for name, payload in (("a.JPEG", b"shared-a"), ("extra.JPEG", b"extra")):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            source.addfile(member, fileobj=io.BytesIO(payload))
    shared_directory = shared / "train" / "n00000002"
    shared_directory.mkdir(parents=True)
    (shared_directory / "b.JPEG").write_bytes(b"shared-b")
    (shared_directory / "extra.JPEG").write_bytes(b"extra")

    output = tmp_path / "wds"
    manifest = prepare_shared(
        archive_path,
        shared,
        output,
        samples_per_shard=3,
        max_train_samples=3,
        max_val_samples=1,
    )

    assert manifest["source_counts"] == {
        "shared_tar": 1,
        "shared_directory": 1,
        "archive_fallback": 1,
    }
    with tarfile.open(output / "train-00000.tar") as shard:
        payloads = {
            member.name: shard.extractfile(member).read()
            for member in shard.getmembers()
            if member.name.endswith(".jpg")
        }
    assert set(payloads.values()) == {b"shared-a", b"shared-b", b"archive-c"}
