from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prepare_imagenet_kaggle import prepare


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
    assert manifest["splits"]["train"]["samples"] == 2
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
