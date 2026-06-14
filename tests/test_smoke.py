from pathlib import Path

import torch

from lgma.synthetic import make_synthetic_batch
from lgma.transformer import TinyTransformerLM, tiny_lm_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_batches_have_expected_shapes():
    for task in ("copy", "reverse", "modular", "previous", "cumsum_mod", "multi_relation"):
        batch = make_synthetic_batch(task, batch_size=2, seq_len=8, vocab_size=16)
        assert batch.input_ids.shape == (2, 8)
        assert batch.targets.shape == (2, 8)


def test_previous_task_is_causal_valid():
    batch = make_synthetic_batch("previous", batch_size=2, seq_len=8, vocab_size=16)
    assert torch.equal(batch.targets[:, 0], torch.zeros(2, dtype=batch.targets.dtype))
    assert torch.equal(batch.targets[:, 1:], batch.input_ids[:, :-1])


def test_multi_relation_targets_apply_selected_relation():
    seq_len = 6
    vocab_size = 16
    for relation_id in range(4):
        relation_ids = torch.full((2,), relation_id)
        batch = make_synthetic_batch(
            "multi_relation",
            batch_size=2,
            seq_len=seq_len,
            vocab_size=vocab_size,
            relation_ids=relation_ids,
        )
        data = batch.input_ids[:, 1:]
        target_body = batch.targets[:, 1:]
        assert torch.equal(batch.input_ids[:, 0], relation_ids)
        assert torch.equal(batch.targets[:, 0], relation_ids)
        if relation_id == 0:
            expected = data
        elif relation_id == 1:
            expected = torch.flip(data, dims=[1])
        elif relation_id == 2:
            expected = torch.cat([data[:, :1], data[:, :-1]], dim=1)
        else:
            expected = torch.cat([data[:, 1:], data[:, -1:]], dim=1)
        assert torch.equal(target_body, expected)


def test_tiny_lm_forward_loss_for_attention_variants():
    torch.manual_seed(0)
    for attention_type in (
        "mha",
        "lgma",
        "lgma_v2",
        "lgma_residual",
        "lgma_quad",
        "lgma_unconstrained",
        "lgma_multibase",
    ):
        model = TinyTransformerLM(
            vocab_size=16,
            d_model=32,
            num_layers=1,
            num_heads=4,
            head_dim=8,
            attention_type=attention_type,
            num_generators=2 if attention_type.startswith("lgma") else 0,
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
        "lgma_v2",
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


def test_train_tinystories_smoke_runs_for_one_step(tmp_path):
    import sys

    data_path = tmp_path / "tiny.txt"
    data_path.write_text(
        "once upon a time there was a small model. " * 20,
        encoding="utf-8",
    )

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    old_argv = sys.argv
    sys.argv = [
        "train_tinystories.py",
        "--data_path",
        str(data_path),
        "--attention",
        "lgma",
        "--steps",
        "1",
        "--batch_size",
        "2",
        "--eval_batches",
        "1",
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
        "--num_generators",
        "2",
        "--diagnostic_every",
        "1",
    ]
    try:
        train_tinystories.main()
    finally:
        sys.argv = old_argv


def test_tinystories_eval_and_generation_helpers_load_checkpoint(tmp_path):
    import sys

    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT / "experiments"))
    from tinystories_runtime import (
        build_tokenizer,
        evaluate_loss,
        generate_text,
        load_tinystories_checkpoint,
    )

    data_path = tmp_path / "tiny.txt"
    text = "once upon a time there was a small model. " * 20
    data_path.write_text(text, encoding="utf-8")
    tokenizer = build_tokenizer(text, None)
    config = {
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 8,
        "attention_type": "lgma",
        "num_generators": 2,
        "generator_type": "full",
        "context_length": 8,
        "dropout": 0.0,
        "num_kv_heads": None,
        "causal": True,
        "theta_init_scale": 0.02,
        "generator_init_scale": 0.02,
        "base_dim": None,
        "value_dim": None,
        "num_base_heads": 1,
        "metric_mode": "exp",
        "metric_beta": 1.0,
        "theta_init": "random_sphere",
        "logit_scale_mode": "sqrt_dim",
        "learn_head_temperature": False,
        "value_transform": "none",
    }
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 3,
            "model_state": model.state_dict(),
            "model_config": config,
            "args": {"data_path": str(data_path), "val_data_path": None},
        },
        checkpoint_path,
    )

    loaded_model, loaded_tokenizer, train_encoded, val_encoded, loaded_config, step = (
        load_tinystories_checkpoint(checkpoint_path, torch.device("cpu"))
    )
    loss, perplexity = evaluate_loss(
        loaded_model,
        val_encoded,
        batch_size=2,
        seq_len=int(loaded_config["context_length"]),
        device=torch.device("cpu"),
        eval_batches=1,
    )
    generated = generate_text(
        loaded_model,
        loaded_tokenizer,
        prompt="once",
        max_new_tokens=4,
        temperature=1.0,
        top_k=5,
        device=torch.device("cpu"),
    )

    assert step == 3
    assert train_encoded.numel() == val_encoded.numel()
    assert loss > 0
    assert perplexity > 1
    assert 0 < len(generated) <= 4
