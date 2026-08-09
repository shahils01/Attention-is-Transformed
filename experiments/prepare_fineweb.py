from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
HASH_PERSON = b"lgma-fw-v1"
SPLIT_BUCKETS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible FineWeb-Edu splits, train a byte-level BPE tokenizer, "
            "and pack documents into resumable uint16 token shards."
        )
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["inspect", "tokenizer", "pack", "all"],
        default="all",
    )
    parser.add_argument("--vocab_size", type=int, default=32_768)
    parser.add_argument("--validation_basis_points", type=int, default=10)
    parser.add_argument("--test_basis_points", type=int, default=10)
    parser.add_argument(
        "--tokenizer_sample_modulus",
        type=int,
        default=50,
        help="Use approximately one training document in N for tokenizer training.",
    )
    parser.add_argument(
        "--tokenizer_max_documents",
        type=int,
        default=0,
        help="Optional cap after deterministic sampling; zero means no cap.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--force_tokenizer",
        action="store_true",
        help="Replace an existing tokenizer instead of verifying and reusing it.",
    )
    return parser.parse_args()


def require_data_dependencies():
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "FineWeb preprocessing requires numpy, pyarrow, and tokenizers; "
            "install the project data extras."
        ) from exc
    return np, pq, Tokenizer


def stable_bucket(value: str, buckets: int = SPLIT_BUCKETS) -> int:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=HASH_PERSON
    ).digest()
    return int.from_bytes(digest, "big") % buckets


def assign_split(
    document_id: str,
    validation_basis_points: int = 10,
    test_basis_points: int = 10,
) -> str:
    if validation_basis_points < 0 or test_basis_points < 0:
        raise ValueError("split basis points must be non-negative")
    if validation_basis_points + test_basis_points >= SPLIT_BUCKETS:
        raise ValueError("validation and test splits must leave training data")
    bucket = stable_bucket(document_id)
    if bucket < validation_basis_points:
        return "validation"
    if bucket < validation_basis_points + test_basis_points:
        return "test"
    return "train"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no Parquet files found in {input_dir}")
    return files


def document_key(raw_id: Any, source: Path, row_index: int) -> str:
    if raw_id is not None and str(raw_id):
        return str(raw_id)
    return f"{source.name}:{row_index}"


def scan_splits(
    files: list[Path],
    output_dir: Path,
    *,
    validation_basis_points: int,
    test_basis_points: int,
    batch_size: int,
) -> dict[str, Any]:
    _, pq, _ = require_data_dependencies()
    totals = {"train": 0, "validation": 0, "test": 0}
    shard_rows = []
    for source in files:
        counts = {"train": 0, "validation": 0, "test": 0}
        row_index = 0
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["id"]):
            for raw_id in batch.column(0).to_pylist():
                split = assign_split(
                    document_key(raw_id, source, row_index),
                    validation_basis_points,
                    test_basis_points,
                )
                counts[split] += 1
                row_index += 1
        if row_index != parquet.metadata.num_rows:
            raise RuntimeError(f"row count mismatch while scanning {source}")
        for split, count in counts.items():
            totals[split] += count
        shard_rows.append(
            {
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "documents": counts,
            }
        )
        print(json.dumps({"event": "split_scan", "source": str(source), **counts}))

    total_documents = sum(totals.values())
    payload = {
        "schema_version": 1,
        "algorithm": "blake2b-64-document-id",
        "hash_person": HASH_PERSON.decode("ascii"),
        "split_buckets": SPLIT_BUCKETS,
        "validation_basis_points": validation_basis_points,
        "test_basis_points": test_basis_points,
        "train_basis_points": SPLIT_BUCKETS
        - validation_basis_points
        - test_basis_points,
        "total_documents": total_documents,
        "documents": totals,
        "document_fractions": {
            split: count / total_documents for split, count in totals.items()
        },
        "source_files": shard_rows,
    }
    atomic_write_json(output_dir / "split_manifest.json", payload)
    return payload


def iter_tokenizer_documents(
    files: list[Path],
    *,
    validation_basis_points: int,
    test_basis_points: int,
    batch_size: int,
    sample_modulus: int,
    max_documents: int,
    stats: dict[str, int],
) -> Iterator[str]:
    _, pq, _ = require_data_dependencies()
    for source in files:
        row_index = 0
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "text"]):
            ids = batch.column(0).to_pylist()
            texts = batch.column(1).to_pylist()
            for raw_id, text in zip(ids, texts):
                key = document_key(raw_id, source, row_index)
                row_index += 1
                if (
                    assign_split(key, validation_basis_points, test_basis_points)
                    != "train"
                ):
                    continue
                if stable_bucket(f"tokenizer:{key}", sample_modulus) != 0:
                    continue
                if max_documents > 0 and stats["documents"] >= max_documents:
                    return
                value = text or ""
                stats["documents"] += 1
                stats["characters"] += len(value)
                yield value


def train_tokenizer(
    files: list[Path],
    output_dir: Path,
    *,
    vocab_size: int,
    validation_basis_points: int,
    test_basis_points: int,
    batch_size: int,
    sample_modulus: int,
    max_documents: int,
    force: bool,
) -> dict[str, Any]:
    _, _, Tokenizer = require_data_dependencies()
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tokenizer_path = output_dir / "tokenizer.json"
    metadata_path = output_dir / "tokenizer_metadata.json"
    if tokenizer_path.exists() and not force:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if tokenizer.get_vocab_size() != vocab_size:
            raise SystemExit(
                f"existing tokenizer has {tokenizer.get_vocab_size()} entries, "
                f"not requested {vocab_size}; use --force_tokenizer to replace it"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(json.dumps({"event": "tokenizer_reused", "path": str(tokenizer_path)}))
        return metadata

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    stats = {"documents": 0, "characters": 0}
    iterator = iter_tokenizer_documents(
        files,
        validation_basis_points=validation_basis_points,
        test_basis_points=test_basis_points,
        batch_size=batch_size,
        sample_modulus=sample_modulus,
        max_documents=max_documents,
        stats=stats,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    if tokenizer.get_vocab_size() > 65_536:
        raise SystemExit("tokenizer vocabulary does not fit in uint16")
    token_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in token_ids.values()):
        raise RuntimeError("trained tokenizer is missing a required special token")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = tokenizer_path.with_name(f".{tokenizer_path.name}.tmp")
    tokenizer.save(str(temporary))
    os.replace(temporary, tokenizer_path)
    metadata = {
        "schema_version": 1,
        "type": "byte_level_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": token_ids,
        "sample_modulus": sample_modulus,
        "max_documents": max_documents,
        "training_documents": stats["documents"],
        "training_characters": stats["characters"],
        "trained_on_split": "train",
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"event": "tokenizer_trained", **metadata}))
    return metadata


def shard_is_complete(metadata_path: Path, output_paths: dict[str, Path]) -> bool:
    if not metadata_path.exists() or not all(path.exists() for path in output_paths.values()):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for split, path in output_paths.items():
        expected = int(metadata["splits"][split]["bytes"])
        if path.stat().st_size != expected:
            return False
    return True


def pack_shard(
    source: Path,
    output_dir: Path,
    tokenizer,
    *,
    eos_token_id: int,
    validation_basis_points: int,
    test_basis_points: int,
    batch_size: int,
) -> dict[str, Any]:
    np, pq, _ = require_data_dependencies()
    stem = source.stem
    output_paths = {
        split: output_dir / "shards" / split / f"{stem}.bin"
        for split in ("train", "validation", "test")
    }
    metadata_path = output_dir / "shard_metadata" / f"{stem}.json"
    if shard_is_complete(metadata_path, output_paths):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(json.dumps({"event": "pack_shard_reused", "source": str(source)}))
        return metadata

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        split: path.with_name(f".{path.name}.tmp") for split, path in output_paths.items()
    }
    handles = {split: path.open("wb") for split, path in temporary_paths.items()}
    counts = {
        split: {"documents": 0, "tokens": 0, "bytes": 0}
        for split in output_paths
    }
    row_index = 0
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "text"]):
            raw_ids = batch.column(0).to_pylist()
            texts = [text or "" for text in batch.column(1).to_pylist()]
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            buffers: dict[str, list[Any]] = {split: [] for split in output_paths}
            for raw_id, encoding in zip(raw_ids, encodings):
                key = document_key(raw_id, source, row_index)
                row_index += 1
                split = assign_split(
                    key, validation_basis_points, test_basis_points
                )
                values = np.empty(len(encoding.ids) + 1, dtype=np.dtype("<u2"))
                values[:-1] = encoding.ids
                values[-1] = eos_token_id
                buffers[split].append(values)
                counts[split]["documents"] += 1
                counts[split]["tokens"] += int(values.size)
            for split, arrays in buffers.items():
                if not arrays:
                    continue
                packed = np.concatenate(arrays)
                handles[split].write(packed.tobytes(order="C"))
                counts[split]["bytes"] += int(packed.nbytes)
    finally:
        for handle in handles.values():
            handle.close()

    if row_index != pq.ParquetFile(source).metadata.num_rows:
        raise RuntimeError(f"row count mismatch while packing {source}")
    for split, temporary in temporary_paths.items():
        if temporary.stat().st_size != counts[split]["bytes"]:
            raise RuntimeError(f"packed byte count mismatch for {temporary}")
        os.replace(temporary, output_paths[split])
    metadata = {
        "schema_version": 1,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "dtype": "uint16-le",
        "eos_token_id": eos_token_id,
        "splits": counts,
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"event": "pack_shard_complete", "source": str(source), **counts}))
    return metadata


def aggregate_packed_metadata(
    output_dir: Path,
    files: list[Path],
    tokenizer_metadata: dict[str, Any],
    shard_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    totals = {
        split: {"documents": 0, "tokens": 0, "bytes": 0}
        for split in ("train", "validation", "test")
    }
    for shard in shard_metadata:
        for split in totals:
            for key in totals[split]:
                totals[split][key] += int(shard["splits"][split][key])
    payload = {
        "schema_version": 1,
        "format": "contiguous_token_ids_with_eos_document_separator",
        "dtype": "uint16-le",
        "source_files": len(files),
        "tokenizer": "tokenizer.json",
        "vocab_size": tokenizer_metadata["vocab_size"],
        "special_token_ids": tokenizer_metadata["special_token_ids"],
        "splits": totals,
        "total_tokens": sum(row["tokens"] for row in totals.values()),
        "total_bytes": sum(row["bytes"] for row in totals.values()),
    }
    atomic_write_json(output_dir / "dataset_metadata.json", payload)
    return payload


def pack_dataset(
    files: list[Path],
    output_dir: Path,
    *,
    validation_basis_points: int,
    test_basis_points: int,
    batch_size: int,
) -> dict[str, Any]:
    _, _, Tokenizer = require_data_dependencies()
    tokenizer_path = output_dir / "tokenizer.json"
    metadata_path = output_dir / "tokenizer_metadata.json"
    if not tokenizer_path.exists() or not metadata_path.exists():
        raise SystemExit("run the tokenizer phase before packing")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    eos_token_id = int(tokenizer_metadata["special_token_ids"]["<eos>"])
    shard_metadata = [
        pack_shard(
            source,
            output_dir,
            tokenizer,
            eos_token_id=eos_token_id,
            validation_basis_points=validation_basis_points,
            test_basis_points=test_basis_points,
            batch_size=batch_size,
        )
        for source in files
    ]
    return aggregate_packed_metadata(
        output_dir, files, tokenizer_metadata, shard_metadata
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.vocab_size <= len(SPECIAL_TOKENS) or args.vocab_size > 65_536:
        raise SystemExit("--vocab_size must be greater than 4 and at most 65536")
    if args.batch_size <= 0 or args.tokenizer_sample_modulus <= 0:
        raise SystemExit("--batch_size and --tokenizer_sample_modulus must be positive")
    if args.tokenizer_max_documents < 0:
        raise SystemExit("--tokenizer_max_documents must be non-negative")
    try:
        assign_split(
            "argument-check",
            args.validation_basis_points,
            args.test_basis_points,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    args = parse_args()
    validate_args(args)
    files = source_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "source_files": [str(path) for path in files],
        "vocab_size": args.vocab_size,
        "validation_basis_points": args.validation_basis_points,
        "test_basis_points": args.test_basis_points,
        "tokenizer_sample_modulus": args.tokenizer_sample_modulus,
        "tokenizer_max_documents": args.tokenizer_max_documents,
        "batch_size": args.batch_size,
    }
    atomic_write_json(args.output_dir / "preprocessing_config.json", config)

    if args.phase in {"inspect", "all"}:
        manifest = scan_splits(
            files,
            args.output_dir,
            validation_basis_points=args.validation_basis_points,
            test_basis_points=args.test_basis_points,
            batch_size=max(args.batch_size, 4096),
        )
        print(json.dumps({"event": "split_manifest_complete", **manifest["documents"]}))
    if args.phase in {"tokenizer", "all"}:
        train_tokenizer(
            files,
            args.output_dir,
            vocab_size=args.vocab_size,
            validation_basis_points=args.validation_basis_points,
            test_basis_points=args.test_basis_points,
            batch_size=args.batch_size,
            sample_modulus=args.tokenizer_sample_modulus,
            max_documents=args.tokenizer_max_documents,
            force=args.force_tokenizer,
        )
    if args.phase in {"pack", "all"}:
        metadata = pack_dataset(
            files,
            args.output_dir,
            validation_basis_points=args.validation_basis_points,
            test_basis_points=args.test_basis_points,
            batch_size=args.batch_size,
        )
        print(json.dumps({"event": "dataset_pack_complete", **metadata}))


if __name__ == "__main__":
    main()
