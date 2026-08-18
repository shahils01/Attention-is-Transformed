from __future__ import annotations

from pathlib import Path

import torch

from lgma.synthetic import CharTokenizer
from ospool.train_tinystories_compat import (
    compact_build_tokenizer,
    compact_encode,
    compact_make_lm_batch,
)


def test_compact_encoding_matches_existing_character_ids() -> None:
    text = "tiny stories\nare useful"
    tokenizer = CharTokenizer(text)
    expected = torch.tensor([tokenizer.stoi[character] for character in text])

    encoded = compact_encode(tokenizer, text)

    assert encoded.dtype == torch.uint8
    assert torch.equal(encoded.long(), expected)


def test_compact_tokenizer_uses_train_and_validation_vocabularies() -> None:
    tokenizer = compact_build_tokenizer("abc", "cd!")

    assert set(tokenizer.itos) == {"a", "b", "c", "d", "!"}


def test_compact_batches_are_long_and_shift_targets() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    tokenizer = CharTokenizer(text)
    encoded = compact_encode(tokenizer, text)

    batch = compact_make_lm_batch(encoded, batch_size=3, seq_len=8)

    assert batch.input_ids.dtype == torch.long
    assert batch.targets.dtype == torch.long
    assert batch.input_ids.shape == (3, 8)
    assert torch.equal(batch.input_ids[:, 1:], batch.targets[:, :-1])


def test_single_gpu_wrapper_uses_an_indexed_cuda_device() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "ospool" / "run_tinystories_hopper_checkpointed.sh").read_text()

    assert "--device cuda:0" in wrapper


def test_wandb_wrapper_preserves_run_id_across_checkpoint_segments() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "ospool" / "run_tinystories_hopper_checkpointed.sh").read_text()

    assert "wandb_run_id_file=\"${RESULTS_DIR}/wandb_run_id\"" in wrapper
    assert "export WANDB_RESUME=allow" in wrapper
    assert "WANDB_API_KEY" in wrapper


def test_wandb_submit_enables_online_mode_and_transfers_bundle() -> None:
    root = Path(__file__).resolve().parents[1]
    submit = (root / "ospool" / "tinystories_hopper_wandb.sub").read_text()

    assert "arguments = online" in submit
    assert "wandb-py311.tar.gz" in submit
    assert "container_image = osdf://" in submit
    assert "gpus_minimum_capability = 8.0" in submit
    assert "gpus_maximum_capability = 10.0" in submit


def test_wandb_smoke_does_not_request_a_gpu() -> None:
    root = Path(__file__).resolve().parents[1]
    submit = (root / "ospool" / "wandb_smoke.sub").read_text()

    assert "request_gpus" not in submit
    assert "wandb-py311.tar.gz" in submit


def test_wandb_smoke_verifies_authentication_without_printing_key() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "ospool" / "run_wandb_smoke.sh").read_text()

    assert "wandb.login(key=os.environ[\"WANDB_API_KEY\"], verify=True)" in wrapper
    assert "characters" in wrapper
