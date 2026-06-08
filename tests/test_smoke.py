from pathlib import Path

import torch

from lgma.synthetic import make_synthetic_batch
from lgma.transformer import TinyTransformerLM, tiny_lm_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_copy_and_reverse_batches_have_expected_shapes():
    for task in ("copy", "reverse"):
        batch = make_synthetic_batch(task, batch_size=2, seq_len=8, vocab_size=16)
        assert batch.input_ids.shape == (2, 8)
        assert batch.targets.shape == (2, 8)


def test_tiny_lm_forward_loss_for_mha_and_lgma():
    torch.manual_seed(0)
    for attention_type in ("mha", "lgma"):
        model = TinyTransformerLM(
            vocab_size=16,
            d_model=32,
            num_layers=1,
            num_heads=4,
            head_dim=8,
            attention_type=attention_type,
            num_generators=2 if attention_type == "lgma" else 0,
            context_length=8,
        )
        batch = make_synthetic_batch("copy", batch_size=2, seq_len=8, vocab_size=16)
        logits, loss = model(batch.input_ids, batch.targets)
        assert logits.shape == (2, 8, 16)
        assert torch.isfinite(loss)


def test_config_loading_instantiates_first_phase_variants():
    for name in ("tiny_mha.json", "tiny_lgma_diag.json", "tiny_lgma_full.json"):
        model = tiny_lm_from_config(ROOT / "experiments" / "configs" / name, vocab_size=32)
        assert isinstance(model, TinyTransformerLM)


def test_train_synthetic_smoke_runs_for_one_step():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_synthetic

    old_argv = sys.argv
    sys.argv = [
        "train_synthetic.py",
        "--task",
        "copy",
        "--attention",
        "lgma",
        "--steps",
        "1",
        "--batch_size",
        "2",
        "--seq_len",
        "8",
        "--vocab_size",
        "16",
        "--d_model",
        "32",
        "--num_layers",
        "1",
        "--num_heads",
        "2",
        "--head_dim",
        "8",
    ]
    try:
        train_synthetic.main()
    finally:
        sys.argv = old_argv
