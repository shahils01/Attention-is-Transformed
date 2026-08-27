from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import tarfile
import time
import zipfile

from experiments.prepare_imagenet_kaggle import (
    TRAIN_PREFIX,
    VAL_PREFIX,
    _add_sample,
    _jpeg_entries,
    _validation_labels,
    _write_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Build exact ImageNet-1K WebDataset shards from NCSA's shared image tree"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--shared-images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--train-shuffle-seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _open_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    overwrite: bool,
) -> tuple[set[int], dict[int, Path]]:
    incomplete: set[int] = set()
    partial_paths: dict[int, Path] = {}
    for shard_index, record in enumerate(records):
        final_path = output_dir / str(record["name"])
        if final_path.exists() and not overwrite:
            continue
        incomplete.add(shard_index)
        partial_path = output_dir / f'.{record["name"]}.partial'
        partial_path.unlink(missing_ok=True)
        partial_paths[shard_index] = partial_path
    return incomplete, partial_paths


def _write_shared_train_split(
    archive: zipfile.ZipFile,
    entries: list[zipfile.ZipInfo],
    shared_images_root: Path,
    output_dir: Path,
    class_to_idx: dict[str, int],
    samples_per_shard: int,
    shuffle_seed: int,
    max_samples: int | None,
    overwrite: bool,
) -> tuple[list[dict[str, object]], int, dict[str, int]]:
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
    expected_by_class: dict[str, dict[str, tuple[int, zipfile.ZipInfo]]] = {}
    for source_index, info in enumerate(entries):
        destination = destination_by_source[source_index]
        if destination < 0:
            continue
        path = PurePosixPath(info.filename)
        wnid = path.parent.name
        expected_by_class.setdefault(wnid, {})[path.name.lower()] = (
            destination,
            info,
        )
        classes_by_shard[destination // samples_per_shard].add(class_to_idx[wnid])
    for record, classes in zip(records, classes_by_shard):
        record["unique_classes"] = len(classes)

    incomplete, partial_paths = _open_outputs(output_dir, records, overwrite)
    outputs: dict[int, tarfile.TarFile] = {}
    seen = bytearray(selected_count)
    source_counts = {"shared_tar": 0, "shared_directory": 0, "archive_fallback": 0}
    started = time.time()
    written = 0

    def write(wnid: str, filename: str, payload: bytes) -> None:
        nonlocal written
        match = expected_by_class[wnid].get(Path(filename).name.lower())
        if match is None:
            return
        destination, info = match
        if seen[destination]:
            return
        seen[destination] = 1
        shard_index = destination // samples_per_shard
        if shard_index in incomplete:
            output = outputs.get(shard_index)
            if output is None:
                output = tarfile.open(partial_paths[shard_index], mode="w")
                outputs[shard_index] = output
            _add_sample(
                output,
                PurePosixPath(info.filename).stem,
                payload,
                class_to_idx[wnid],
                wnid,
            )
        written += 1
        if written % 10_000 == 0:
            elapsed = max(time.time() - started, 1.0)
            print(
                f"train: {written:,}/{selected_count:,} official images "
                f"({written / elapsed:.1f} images/s)",
                flush=True,
            )

    try:
        for wnid in sorted(expected_by_class):
            expected = expected_by_class[wnid]
            tar_path = shared_images_root / f"{wnid}.tar"
            directory = shared_images_root / "train" / wnid
            before = written
            if tar_path.is_file():
                with tarfile.open(tar_path) as source:
                    for member in source:
                        if not member.isfile() or Path(member.name).name.lower() not in expected:
                            continue
                        extracted = source.extractfile(member)
                        if extracted is not None:
                            write(wnid, member.name, extracted.read())
                source_counts["shared_tar"] += written - before
            elif directory.is_dir():
                for path in directory.iterdir():
                    if path.is_file() and path.name.lower() in expected:
                        write(wnid, path.name, path.read_bytes())
                source_counts["shared_directory"] += written - before
            fallback_before = written
            for _, info in expected.values():
                destination, _ = expected[PurePosixPath(info.filename).name.lower()]
                if not seen[destination]:
                    write(wnid, PurePosixPath(info.filename).name, archive.read(info))
            source_counts["archive_fallback"] += written - fallback_before
    finally:
        for output in outputs.values():
            output.close()

    missing = [index for index, value in enumerate(seen) if not value]
    if missing:
        examples = [entries[index].filename for index in missing[:10]]
        raise RuntimeError(
            f"shared sources are missing {len(missing)} official training images: "
            + ", ".join(examples)
        )
    for shard_index in sorted(incomplete):
        os.replace(
            partial_paths[shard_index],
            output_dir / str(records[shard_index]["name"]),
        )
    return records, selected_count, source_counts


def _write_metadata(output_dir: Path, manifest: dict[str, object]) -> None:
    temporary = output_dir / ".manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "manifest.json")
    timm_info = {
        "splits": {
            name: {
                "name": name,
                "num_samples": split_manifest["samples"],
                "filenames": [row["name"] for row in split_manifest["shards"]],
                "shard_lengths": [row["samples"] for row in split_manifest["shards"]],
            }
            for name, split_manifest in manifest["splits"].items()
        }
    }
    temporary_info = output_dir / "._info.json.partial"
    temporary_info.write_text(json.dumps(timm_info, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_info, output_dir / "_info.json")


def prepare_shared(
    archive_path: Path,
    shared_images_root: Path,
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
    with zipfile.ZipFile(archive_path) as archive:
        train_entries = sorted(
            _jpeg_entries(archive, TRAIN_PREFIX), key=lambda info: info.filename
        )
        val_entries = sorted(
            _jpeg_entries(archive, VAL_PREFIX), key=lambda info: info.filename
        )
        wnids = sorted(
            {PurePosixPath(info.filename).parent.name for info in train_entries}
        )
        if max_train_samples is None and len(wnids) != 1000:
            raise ValueError(f"expected 1000 ImageNet classes, found {len(wnids)}")
        class_to_idx = {wnid: index for index, wnid in enumerate(wnids)}
        if max_val_samples is not None:
            val_entries = val_entries[:max_val_samples]
        validation_labels = _validation_labels(archive)
        train_shards, train_samples, source_counts = _write_shared_train_split(
            archive,
            train_entries,
            shared_images_root,
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
        "source_shared_images_root": str(shared_images_root),
        "source_counts": source_counts,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "samples_per_shard": samples_per_shard,
        "train_order": "deterministic_random_shuffle",
        "train_shuffle_seed": train_shuffle_seed,
        "splits": {
            "train": {
                "samples": train_samples,
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
    _write_metadata(output_dir, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_shared(
        args.archive,
        args.shared_images_root,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        train_shuffle_seed=args.train_shuffle_seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "source_counts": manifest["source_counts"],
                "splits": {
                    name: {"samples": value["samples"], "pattern": value["pattern"]}
                    for name, value in manifest["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
