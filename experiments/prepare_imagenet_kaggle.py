from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import tarfile
import time
import zipfile


TRAIN_PREFIX = "ILSVRC/Data/CLS-LOC/train/"
VAL_PREFIX = "ILSVRC/Data/CLS-LOC/val/"
VAL_LABELS = "LOC_val_solution.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Convert the authorized Kaggle ImageNet archive into deterministic WebDataset shards"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument(
        "--train-shuffle-seed",
        type=int,
        default=0,
        help="Deterministically shuffle training images before forming shards.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jpeg_entries(archive: zipfile.ZipFile, prefix: str) -> list[zipfile.ZipInfo]:
    return [
        info
        for info in archive.infolist()
        if info.filename.startswith(prefix)
        and info.filename.lower().endswith((".jpeg", ".jpg"))
        and not info.is_dir()
    ]


def _validation_labels(archive: zipfile.ZipFile) -> dict[str, str]:
    with archive.open(VAL_LABELS) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        labels: dict[str, str] = {}
        for row in reader:
            image_id = row["ImageId"]
            prediction = row["PredictionString"].split()
            if not prediction:
                raise ValueError(f"validation image {image_id} has no label")
            labels[image_id] = prediction[0]
    return labels


def _tar_member(name: str, payload: bytes) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name=name)
    member.size = len(payload)
    member.mtime = 0
    member.mode = 0o640
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def _add_sample(
    output: tarfile.TarFile,
    key: str,
    image: bytes,
    label: int,
    wnid: str,
) -> None:
    fields = {
        f"{key}.jpg": image,
        f"{key}.cls": str(label).encode("ascii"),
        f"{key}.wnid": wnid.encode("ascii"),
    }
    for name, payload in fields.items():
        output.addfile(_tar_member(name, payload), io.BytesIO(payload))


def _write_split(
    archive: zipfile.ZipFile,
    entries: list[zipfile.ZipInfo],
    output_dir: Path,
    split: str,
    class_to_idx: dict[str, int],
    samples_per_shard: int,
    validation_labels: dict[str, str] | None,
    overwrite: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    started = time.time()
    for shard_index, start in enumerate(range(0, len(entries), samples_per_shard)):
        shard_entries = entries[start : start + samples_per_shard]
        shard_name = f"{split}-{shard_index:05d}.tar"
        final_path = output_dir / shard_name
        partial_path = output_dir / f".{shard_name}.partial"
        if final_path.exists() and not overwrite:
            records.append({"name": shard_name, "samples": len(shard_entries)})
            continue
        partial_path.unlink(missing_ok=True)
        with tarfile.open(partial_path, mode="w") as output:
            for info in shard_entries:
                path = PurePosixPath(info.filename)
                image_id = path.stem
                if split == "train":
                    wnid = path.parent.name
                else:
                    assert validation_labels is not None
                    wnid = validation_labels[image_id]
                label = class_to_idx[wnid]
                _add_sample(output, image_id, archive.read(info), label, wnid)
        os.replace(partial_path, final_path)
        records.append({"name": shard_name, "samples": len(shard_entries)})
        processed = min(start + len(shard_entries), len(entries))
        elapsed = max(time.time() - started, 1.0)
        print(
            f"{split}: {processed:,}/{len(entries):,} images "
            f"({processed / elapsed:.1f} images/s)",
            flush=True,
        )
    return records


def _write_shuffled_train_split(
    archive: zipfile.ZipFile,
    entries: list[zipfile.ZipInfo],
    output_dir: Path,
    class_to_idx: dict[str, int],
    samples_per_shard: int,
    shuffle_seed: int,
    max_samples: int | None,
    overwrite: bool,
) -> tuple[list[dict[str, object]], int]:
    """Read class-sorted ZIP entries sequentially into randomized shards."""
    source_indices = list(range(len(entries)))
    random.Random(shuffle_seed).shuffle(source_indices)
    selected_count = len(source_indices) if max_samples is None else min(
        max_samples, len(source_indices)
    )
    source_indices = source_indices[:selected_count]
    destination_by_source = [-1] * len(entries)
    for destination, source_index in enumerate(source_indices):
        destination_by_source[source_index] = destination
    del source_indices

    num_shards = (selected_count + samples_per_shard - 1) // samples_per_shard
    records = [
        {
            "name": f"train-{shard_index:05d}.tar",
            "samples": min(
                samples_per_shard,
                selected_count - shard_index * samples_per_shard,
            ),
        }
        for shard_index in range(num_shards)
    ]
    classes_by_shard = [set() for _ in range(num_shards)]
    for source_index, destination in enumerate(destination_by_source):
        if destination >= 0:
            shard_index = destination // samples_per_shard
            wnid = PurePosixPath(entries[source_index].filename).parent.name
            classes_by_shard[shard_index].add(class_to_idx[wnid])
    for record, classes in zip(records, classes_by_shard):
        record["unique_classes"] = len(classes)
    incomplete: set[int] = set()
    partial_paths: dict[int, Path] = {}
    outputs: dict[int, tarfile.TarFile] = {}
    for shard_index, record in enumerate(records):
        final_path = output_dir / str(record["name"])
        if final_path.exists() and not overwrite:
            continue
        incomplete.add(shard_index)
        partial_path = output_dir / f'.{record["name"]}.partial'
        partial_path.unlink(missing_ok=True)
        partial_paths[shard_index] = partial_path

    started = time.time()
    written = 0
    try:
        for source_index, info in enumerate(entries):
            destination = destination_by_source[source_index]
            if destination < 0:
                continue
            shard_index = destination // samples_per_shard
            if shard_index not in incomplete:
                continue
            output = outputs.get(shard_index)
            if output is None:
                output = tarfile.open(partial_paths[shard_index], mode="w")
                outputs[shard_index] = output
            path = PurePosixPath(info.filename)
            wnid = path.parent.name
            _add_sample(
                output,
                path.stem,
                archive.read(info),
                class_to_idx[wnid],
                wnid,
            )
            written += 1
            if written % 10_000 == 0:
                elapsed = max(time.time() - started, 1.0)
                print(
                    f"train: {written:,}/{selected_count:,} images "
                    f"({written / elapsed:.1f} images/s)",
                    flush=True,
                )
    finally:
        for output in outputs.values():
            output.close()

    for shard_index in sorted(incomplete):
        os.replace(
            partial_paths[shard_index],
            output_dir / str(records[shard_index]["name"]),
        )
    return records, selected_count


def prepare(
    archive_path: Path,
    output_dir: Path,
    *,
    samples_per_shard: int = 1024,
    train_shuffle_seed: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    if train_shuffle_seed < 0:
        raise ValueError("train_shuffle_seed must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text())
        expected = {
            "samples_per_shard": samples_per_shard,
            "train_order": "deterministic_random_shuffle",
            "train_shuffle_seed": train_shuffle_seed,
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise FileExistsError(
                f"existing shard protocol is incompatible: {mismatches}; "
                "use a new output directory or --overwrite"
            )
    with zipfile.ZipFile(archive_path) as archive:
        train_entries = _jpeg_entries(archive, TRAIN_PREFIX)
        val_entries = _jpeg_entries(archive, VAL_PREFIX)
        train_entries.sort(key=lambda info: info.filename)
        val_entries.sort(key=lambda info: info.filename)
        wnids = sorted(
            {PurePosixPath(info.filename).parent.name for info in train_entries}
        )
        if max_train_samples is None and len(wnids) != 1000:
            raise ValueError(f"expected 1000 ImageNet classes, found {len(wnids)}")
        class_to_idx = {wnid: index for index, wnid in enumerate(wnids)}
        if max_val_samples is not None:
            val_entries = val_entries[:max_val_samples]
        validation_labels = _validation_labels(archive)
        missing = {
            validation_labels[PurePosixPath(info.filename).stem]
            for info in val_entries
            if validation_labels[PurePosixPath(info.filename).stem] not in class_to_idx
        }
        if missing:
            raise ValueError(f"validation labels missing from training classes: {sorted(missing)}")
        # The source ZIP is class ordered. Assign each sequentially read entry
        # to a deterministic randomized destination so shards are well mixed
        # without imposing random seeks on the large HDD archive.
        train_shards, train_sample_count = _write_shuffled_train_split(
            archive,
            train_entries,
            output_dir,
            class_to_idx,
            samples_per_shard,
            train_shuffle_seed,
            max_train_samples,
            overwrite,
        )
        val_shards = _write_split(
            archive,
            val_entries,
            output_dir,
            "val",
            class_to_idx,
            samples_per_shard,
            validation_labels,
            overwrite,
        )

    manifest: dict[str, object] = {
        "format": "webdataset",
        "source_archive": str(archive_path),
        "source_archive_bytes": archive_path.stat().st_size,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "samples_per_shard": samples_per_shard,
        "train_order": "deterministic_random_shuffle",
        "train_shuffle_seed": train_shuffle_seed,
        "splits": {
            "train": {
                "samples": train_sample_count,
                "shards": train_shards,
                "pattern": f"train-{{00000..{len(train_shards) - 1:05d}}}.tar",
            },
            "val": {
                "samples": len(val_entries),
                "shards": val_shards,
                "pattern": f"val-{{00000..{len(val_shards) - 1:05d}}}.tar",
            },
        },
    }
    temporary = output_dir / ".manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    timm_info = {
        "splits": {
            name: {
                "name": name,
                "num_samples": split_manifest["samples"],
                "filenames": [record["name"] for record in split_manifest["shards"]],
                "shard_lengths": [
                    record["samples"] for record in split_manifest["shards"]
                ],
            }
            for name, split_manifest in manifest["splits"].items()
        }
    }
    temporary_info = output_dir / "._info.json.partial"
    temporary_info.write_text(json.dumps(timm_info, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_info, output_dir / "_info.json")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare(
        args.archive,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        train_shuffle_seed=args.train_shuffle_seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "num_classes": manifest["num_classes"],
        "splits": {
            name: {"samples": value["samples"], "pattern": value["pattern"]}
            for name, value in manifest["splits"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
