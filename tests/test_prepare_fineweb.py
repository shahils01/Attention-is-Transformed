import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from prepare_fineweb import (  # noqa: E402
    SPECIAL_TOKENS,
    assign_split,
    pack_dataset,
    scan_splits,
    stable_bucket,
    train_tokenizer,
)


def test_stable_document_splits_are_deterministic_and_disjoint():
    ids = [f"document-{index}" for index in range(20_000)]
    first = [assign_split(value) for value in ids]
    second = [assign_split(value) for value in ids]

    assert first == second
    assert set(first) == {"train", "validation", "test"}
    assert 5 <= first.count("validation") <= 40
    assert 5 <= first.count("test") <= 40
    assert stable_bucket("document-7") == stable_bucket("document-7")


def test_invalid_split_configuration_is_rejected():
    with pytest.raises(ValueError):
        assign_split("document", validation_basis_points=6_000, test_basis_points=4_000)
    with pytest.raises(ValueError):
        stable_bucket("document", buckets=0)


def test_tiny_parquet_pipeline(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("tokenizers")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = 400
    table = pa.table(
        {
            "id": [f"doc-{index}" for index in range(rows)],
            "text": [
                f"Document number {index}. Robots learn from carefully prepared data."
                for index in range(rows)
            ],
        }
    )
    source = input_dir / "000.parquet"
    pq.write_table(table, source, row_group_size=50)

    manifest = scan_splits(
        [source],
        output_dir,
        validation_basis_points=2_000,
        test_basis_points=2_000,
        batch_size=64,
    )
    tokenizer_metadata = train_tokenizer(
        [source],
        output_dir,
        vocab_size=300,
        validation_basis_points=2_000,
        test_basis_points=2_000,
        batch_size=64,
        sample_modulus=1,
        max_documents=0,
        force=False,
    )
    dataset_metadata = pack_dataset(
        [source],
        output_dir,
        validation_basis_points=2_000,
        test_basis_points=2_000,
        batch_size=64,
    )

    assert manifest["total_documents"] == rows
    assert sum(manifest["documents"].values()) == rows
    assert tokenizer_metadata["vocab_size"] == 300
    assert set(tokenizer_metadata["special_token_ids"]) == set(SPECIAL_TOKENS)
    assert dataset_metadata["total_tokens"] > rows
    assert sum(
        split["documents"] for split in dataset_metadata["splits"].values()
    ) == rows
    assert dataset_metadata["total_bytes"] == dataset_metadata["total_tokens"] * 2
    for split in ("train", "validation", "test"):
        binary = output_dir / "shards" / split / "000.bin"
        assert binary.stat().st_size == dataset_metadata["splits"][split]["bytes"]
    saved = json.loads((output_dir / "dataset_metadata.json").read_text())
    assert saved == dataset_metadata
