import math
from pathlib import Path

import torch

from lgma.synthetic import make_synthetic_batch
from lgma.transformer import TinyTransformerLM, build_attention, tiny_lm_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_lgma_presets_default_to_random_sphere_and_respect_explicit_circle():
    for attention_type in (
        "lgma_v2",
        "lgma_residual",
        "lgma_quad",
        "lgma_value_diag",
        "lgma_multibase",
        "lgma_multibase_value_diag",
    ):
        layer = build_attention(
            attention_type=attention_type,
            d_model=32,
            num_heads=4,
            head_dim=8,
            num_generators=4,
        )
        assert layer.theta_init == "random_sphere"

    circle_layer = build_attention(
        attention_type="lgma_residual",
        d_model=32,
        num_heads=4,
        head_dim=8,
        num_generators=4,
        theta_init="circle",
    )
    assert circle_layer.theta_init == "circle"


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


def test_tinystories_learning_rate_schedule():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    kwargs = {
        "base_lr": 1e-4,
        "min_lr": 1e-5,
        "total_steps": 10,
        "warmup_steps": 2,
        "hold_steps": 3,
        "schedule": "cosine",
    }
    assert train_tinystories.learning_rate_for_step(1, **kwargs) == 5e-5
    assert train_tinystories.learning_rate_for_step(2, **kwargs) == 1e-4
    assert train_tinystories.learning_rate_for_step(5, **kwargs) == 1e-4
    assert math.isclose(
        train_tinystories.learning_rate_for_step(10, **kwargs),
        1e-5,
        rel_tol=1e-6,
    )


def test_tinystories_generator_normalization_is_opt_in():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    old_argv = sys.argv
    try:
        sys.argv = ["train_tinystories.py", "--data_path", "unused.txt"]
        default_config = train_tinystories.model_config_from_args(train_tinystories.parse_args())

        sys.argv = [
            "train_tinystories.py",
            "--data_path",
            "unused.txt",
            "--normalize_generators",
        ]
        normalized_config = train_tinystories.model_config_from_args(
            train_tinystories.parse_args()
        )
    finally:
        sys.argv = old_argv

    assert default_config["normalize_generators"] is False
    assert normalized_config["normalize_generators"] is True


def test_tinystories_generator_mixing_defaults_to_softmax_and_supports_none():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    old_argv = sys.argv
    try:
        sys.argv = ["train_tinystories.py", "--data_path", "unused.txt"]
        default_config = train_tinystories.model_config_from_args(
            train_tinystories.parse_args()
        )

        sys.argv = [
            "train_tinystories.py",
            "--data_path",
            "unused.txt",
            "--generator_mixing",
            "none",
        ]
        raw_config = train_tinystories.model_config_from_args(
            train_tinystories.parse_args()
        )
    finally:
        sys.argv = old_argv

    assert default_config["generator_mixing"] == "softmax"
    assert raw_config["generator_mixing"] == "none"


def test_tinystories_compile_backend_is_configurable():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    old_argv = sys.argv
    try:
        sys.argv = ["train_tinystories.py", "--data_path", "unused.txt"]
        default_args = train_tinystories.parse_args()

        sys.argv = [
            "train_tinystories.py",
            "--data_path",
            "unused.txt",
            "--compile",
            "--compile_backend",
            "aot_eager",
        ]
        diagnostic_args = train_tinystories.parse_args()
    finally:
        sys.argv = old_argv

    assert default_args.compile is False
    assert default_args.compile_backend == "inductor"
    assert diagnostic_args.compile is True
    assert diagnostic_args.compile_backend == "aot_eager"


def test_tinystories_generator_stability_flags_are_configurable():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    old_argv = sys.argv
    try:
        sys.argv = ["train_tinystories.py", "--data_path", "unused.txt"]
        default_config = train_tinystories.model_config_from_args(train_tinystories.parse_args())

        sys.argv = [
            "train_tinystories.py",
            "--data_path",
            "unused.txt",
            "--no-stabilize_generators",
            "--head_generator_symmetric_cap",
            str(math.log(4.0)),
        ]
        ablation_config = train_tinystories.model_config_from_args(
            train_tinystories.parse_args()
        )
    finally:
        sys.argv = old_argv

    assert default_config["stabilize_generators"] is True
    assert default_config["head_generator_symmetric_cap"] is None
    assert ablation_config["stabilize_generators"] is False
    assert math.isclose(
        ablation_config["head_generator_symmetric_cap"],
        math.log(4.0),
    )


def test_zero_ah_norm_weight_skips_lgma_generator_work(monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    model = TinyTransformerLM(
        vocab_size=16,
        d_model=32,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        attention_type="lgma",
        num_generators=2,
        context_length=8,
    )

    def unexpected_generator_work():
        raise AssertionError("zero-weight A_h regularization must not build generators")

    monkeypatch.setattr(
        model.first_attention,
        "compute_head_generators",
        unexpected_generator_work,
    )
    regularizer = train_tinystories.compute_ah_norm_regularizer(
        model,
        ah_norm_weight=0.0,
        ah_norm_max=0.0,
        device=torch.device("cpu"),
    )

    assert regularizer.item() == 0.0
    assert not regularizer.requires_grad

    def unexpected_module_scan():
        raise AssertionError("zero-weight diversity regularization must not scan the model")

    monkeypatch.setattr(model, "modules", unexpected_module_scan)
    metric_loss, induced_loss = train_tinystories.compute_diversity_regularizers(
        model,
        metric_weight=0.0,
        induced_weight=0.0,
        squared=False,
        use_delta=True,
        device=torch.device("cpu"),
    )
    assert metric_loss.item() == 0.0
    assert induced_loss.item() == 0.0


def test_tiny_lm_forward_loss_for_attention_variants():
    torch.manual_seed(0)
    for attention_type in (
        "mha",
        "collaborative",
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
    for name in (
        "tiny_mha.json",
        "tiny_collaborative.json",
        "tiny_lgma_diag.json",
        "tiny_lgma_full.json",
    ):
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


def test_train_tinystories_checkpoint_resumes_after_logging_optimizations(tmp_path):
    import sys

    data_path = tmp_path / "tiny.txt"
    data_path.write_text(
        "once upon a time there was a small model. " * 20,
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    common_args = [
        "train_tinystories.py",
        "--data_path",
        str(data_path),
        "--attention",
        "lgma",
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
        "--output_dir",
        str(output_dir),
        "--save_every",
        "1",
        "--diagnostic_batches",
        "1",
    ]

    old_argv = sys.argv
    try:
        sys.argv = [*common_args, "--steps", "1"]
        train_tinystories.main()
        checkpoint = output_dir / "checkpoint_step_1.pt"
        assert checkpoint.exists()

        sys.argv = [
            *common_args,
            "--steps",
            "2",
            "--resume_checkpoint",
            str(checkpoint),
        ]
        train_tinystories.main()
    finally:
        sys.argv = old_argv

    from lgma.checkpointing import load_full_checkpoint

    resumed = load_full_checkpoint(
        output_dir / "checkpoint_step_2.pt",
        map_location="cpu",
    )
    assert resumed["step"] == 2


def test_tinystories_checkpoint_retention_keeps_latest_steps(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    for step in [100, 20, 300, 200]:
        (tmp_path / f"checkpoint_step_{step}.pt").write_text("checkpoint")
    unrelated = tmp_path / "checkpoint_final.pt"
    unrelated.write_text("final")

    train_tinystories.prune_old_checkpoints(tmp_path, keep=3)

    assert sorted(path.name for path in tmp_path.glob("checkpoint_step_*.pt")) == [
        "checkpoint_step_100.pt",
        "checkpoint_step_200.pt",
        "checkpoint_step_300.pt",
    ]
    assert unrelated.exists()


def test_tinystories_checkpoint_retention_preserves_milestones(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import train_tinystories

    for step in [50_000, 100_000, 150_000, 150_100, 150_200, 150_300]:
        (tmp_path / f"checkpoint_step_{step}.pt").write_text("checkpoint")

    train_tinystories.prune_old_checkpoints(
        tmp_path,
        keep=3,
        milestone_every=50_000,
    )

    assert sorted(
        int(path.stem.removeprefix("checkpoint_step_"))
        for path in tmp_path.glob("checkpoint_step_*.pt")
    ) == [50_000, 100_000, 150_000, 150_100, 150_200, 150_300]


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
