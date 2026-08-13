import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from lgma.packed_data import load_packed_token_corpus  # noqa: E402
from lgma.checkpointing import load_full_checkpoint  # noqa: E402
from train_tinystories import (  # noqa: E402
    normalize_data_generator_state,
    restore_data_generator_state,
)


def make_packed_corpus(root: Path, vocab_size: int = 32) -> Path:
    split_values = {
        "train": np.arange(500, dtype=np.uint16) % vocab_size,
        "validation": np.arange(120, dtype=np.uint16) % vocab_size,
        "test": np.arange(100, dtype=np.uint16) % vocab_size,
    }
    split_metadata = {}
    for split, values in split_values.items():
        directory = root / "shards" / split
        directory.mkdir(parents=True)
        path = directory / "000.bin"
        values.astype("<u2").tofile(path)
        split_metadata[split] = {
            "documents": 5,
            "tokens": int(values.size),
            "bytes": int(values.nbytes),
        }
    metadata = {
        "schema_version": 1,
        "format": "contiguous_token_ids_with_eos_document_separator",
        "dtype": "uint16-le",
        "source_files": 1,
        "tokenizer": "tokenizer.json",
        "vocab_size": vocab_size,
        "special_token_ids": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
        "splits": split_metadata,
        "total_tokens": sum(row["tokens"] for row in split_metadata.values()),
        "total_bytes": sum(row["bytes"] for row in split_metadata.values()),
    }
    (root / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_generator_state_is_normalized_for_cpu_generator_restore():
    generator = torch.Generator().manual_seed(123)
    original_state = generator.get_state()
    non_byte_state = original_state.to(dtype=torch.int64)

    normalized_state = normalize_data_generator_state(non_byte_state)

    assert normalized_state.device.type == "cpu"
    assert normalized_state.dtype == torch.uint8
    restored = torch.Generator()
    restored.set_state(normalized_state)
    assert torch.equal(restored.get_state(), original_state)


def test_new_ddp_rank_retains_its_independent_seeded_generator_state():
    saved_generator = torch.Generator().manual_seed(123)
    new_rank_generator = torch.Generator().manual_seed(456)
    initial_new_rank_state = new_rank_generator.get_state().clone()

    restored = restore_data_generator_state(
        new_rank_generator, [saved_generator.get_state()], data_rank=1
    )

    assert restored is False
    assert torch.equal(new_rank_generator.get_state(), initial_new_rank_state)


def test_packed_corpus_validates_counts_and_samples_deterministically(tmp_path):
    corpus = load_packed_token_corpus(make_packed_corpus(tmp_path / "packed"))
    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)
    first = corpus.train.sample_batch(4, 16, generator=generator_a)
    second = corpus.train.sample_batch(4, 16, generator=generator_b)

    assert corpus.tokenizer.vocab_size == 32
    assert corpus.train.numel() == 500
    assert first.input_ids.shape == (4, 16)
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.targets, second.targets)
    assert torch.equal(first.input_ids[:, 1:], first.targets[:, :-1])


def test_packed_corpus_rejects_metadata_file_mismatch(tmp_path):
    root = make_packed_corpus(tmp_path / "packed")
    metadata_path = root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["splits"]["train"]["tokens"] += 1
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="token count mismatch"):
        load_packed_token_corpus(root)


def test_existing_training_runner_trains_on_packed_data(tmp_path, monkeypatch):
    import train_tinystories

    packed = make_packed_corpus(tmp_path / "packed")
    output = tmp_path / "run"
    argv = [
        "train_tinystories.py",
        "--packed_data_dir",
        str(packed),
        "--attention",
        "mha",
        "--steps",
        "2",
        "--batch_size",
        "2",
        "--eval_batches",
        "2",
        "--context_length",
        "8",
        "--d_model",
        "32",
        "--num_layers",
        "1",
        "--num_heads",
        "2",
        "--head_dim",
        "8",
        "--output_dir",
        str(output),
        "--save_every",
        "1",
        "--diagnostic_batches",
        "1",
        "--wandb_mode",
        "disabled",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_tinystories.main()

    report = json.loads((output / "final_report.json").read_text())
    assert report["data_backend"] == "packed_uint16"
    assert report["train_tokens"] == 500
    assert report["validation_tokens"] == 120
    assert report["vocab_size"] == 32
    assert report["data_sampling"] == "uniform_random_with_replacement"
    checkpoint = load_full_checkpoint(output / "checkpoint_step_2.pt")
    assert len(checkpoint["data_generator_states"]) == 1
