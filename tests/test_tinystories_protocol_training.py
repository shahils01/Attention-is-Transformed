import argparse
import copy
import hashlib
import json

import pytest
import torch

from experiments import train_tinystories as trainer
from experiments import tinystories_protocol_training as protocol
from lgma.checkpointing import load_full_checkpoint
from lgma.transformer import TinyTransformerLM


CPU = torch.device("cpu")


def model_config(arm):
    config = dict(
        attention_type=arm, d_model=16, num_layers=2, num_heads=4,
        head_dim=4, context_length=5, dropout=0.15, causal=True,
    )
    if arm == "gt_mha_lu":
        config.update(metric_mode="lu", num_base_heads=2, fuse_lu_queries=False)
    else:
        config.update(num_kv_heads=2)
    return config


def build_model(arm, seed=100):
    return protocol.initialize_locked_model(TinyTransformerLM, 16, model_config(arm), seed)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, **protocol.ADAMW_OPTIONS)


def run_update(model, optimizer, sampler):
    batch = trainer.make_data_batch(torch.arange(200, dtype=torch.uint8) % 16, 2, 5, CPU, generator=sampler)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(batch.input_ids, batch.targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_paired_initialization_shares_actual_tensors_and_preserves_rng():
    torch.manual_seed(17)
    expected_rng = torch.random.get_rng_state()
    lu = build_model("gt_mha_lu")
    gqa = build_model("gqa")
    assert torch.equal(torch.random.get_rng_state(), expected_rng)

    lu_state, gqa_state = lu.state_dict(), gqa.state_dict()
    for name, tensor in gqa_state.items():
        if ".q_proj." not in name:
            torch.testing.assert_close(lu_state[name], tensor, rtol=0, atol=0)
    assert lu_state["blocks.0.attn.q_proj.weight"].shape != gqa_state["blocks.0.attn.q_proj.weight"].shape
    assert torch.count_nonzero(lu_state["blocks.0.attn.lu_lower"]) > 0
    identity = torch.eye(4).expand(2, 4, 4)
    for block in lu.blocks:
        torch.testing.assert_close(block.attn.compute_metrics()[::2], identity, rtol=0, atol=0)
    digest = protocol.common_initialization_sha256(lu_state)
    assert digest == protocol.common_initialization_sha256(gqa_state)
    assert digest != protocol.common_initialization_sha256(build_model("gqa", seed=200).state_dict())


def test_locked_encoding_matches_historical_character_mapping(tmp_path):
    path = tmp_path / "tiny.txt"
    payload = ("a" * (1024 * 1024 - 1) + "🐈\r\n😀\r").encode()
    path.write_bytes(payload)
    tokenizer = protocol.build_locked_character_tokenizer(sorted(set("a🐈\n😀")))
    encoded = protocol.encode_locked_character_text(path, tokenizer, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert encoded.dtype == torch.uint8
    assert torch.equal(encoded.long(), tokenizer.encode(path.read_text()))
    with pytest.raises(ValueError, match="changed"):
        protocol.encode_locked_character_text(path, tokenizer, expected_sha256="different")
    for unknown in ("x", "Ö", "💡"):
        path.write_text(unknown, encoding="utf-8")
        with pytest.raises(ValueError, match="locked vocabulary"):
            protocol.encode_locked_character_text(path, tokenizer)


def test_locked_validation_and_diagnostics_use_matching_inputs_and_preserve_rng(monkeypatch):
    samples = []
    make_batch = trainer.make_data_batch

    def recording_batch(*args, **kwargs):
        torch.rand(5)  # Even a diagnostic implementation that draws from global RNG stays isolated.
        batch = make_batch(*args, **kwargs)
        samples.append(batch.input_ids.clone())
        return batch

    monkeypatch.setattr(trainer, "make_data_batch", recording_batch)
    fake_validation = torch.arange(200, dtype=torch.uint8) % 16
    expected_rng = torch.random.get_rng_state()
    for arm in ("gt_mha_lu", "gqa"):
        model = build_model(arm)
        trainer.evaluate_text_loss(model, fake_validation, 2, 5, CPU, 2, locked_protocol=True)
        trainer.add_attention_diagnostics({}, model, fake_validation, 2, 5, CPU, 2, locked_protocol=True)
        assert model.training and model.first_attention.training
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert len(samples) == 8
    for lu, gqa in zip(samples[:4], samples[4:]):
        assert torch.equal(lu, gqa)
    assert not torch.equal(samples[0], samples[2])


@pytest.mark.parametrize("arm", ["gt_mha_lu", "gqa"])
def test_locked_resume_checks_lineage_and_replays_dropout_and_sampler(tmp_path, arm):
    model = build_model(arm)
    optimizer = optimizer_for(model)
    source = {"git_revision": "revision", "files_sha256": {"trainer.py": "trainer-content"}}
    identity = {
        "protocol_id": "fixture", "protocol_sha256": "fixture-hash", "arm": arm, "seed": 100,
        "lock": {
            "model_config": model_config(arm), "training": {"arguments": {"steps": 3, "save_every": 1}},
            "runtime": {"world_size": 1, "tokens_per_step": 10}, "source": source,
            "data": {"train": {"sha256": "train"}},
        },
    }
    args = argparse.Namespace(steps=3, save_every=1, locked_protocol=True, attention=arm, seed=100,
                              resume_checkpoint=None, output_dir=tmp_path)
    manifest = protocol.write_initialization(tmp_path, identity, model, vars(args))
    trainer.write_lu_run_manifest(tmp_path / protocol.MANIFEST_NAME, manifest, inherited=False)
    evidence_path = protocol.verify_initialization(tmp_path, manifest)
    step_zero = load_full_checkpoint(evidence_path, map_location=CPU)
    assert step_zero["run_identity_sha256"] == protocol.manifest_sha256(identity)
    assert step_zero["common_parameters_sha256"] == protocol.common_initialization_sha256(step_zero["model_state"])
    with pytest.raises(ValueError, match="already contains"):
        protocol.write_initialization(tmp_path, identity, model, vars(args))

    sampler = torch.Generator().manual_seed(100)
    protocol.seed_training_rng(100, 0, CPU)
    run_update(model, optimizer, sampler)
    data_states, protocol_states = trainer.gather_checkpoint_rng(sampler, CPU, False, 0, 1, True)
    path = tmp_path / "checkpoint_step_1.pt"
    trainer.save_checkpoint(path, model, optimizer, None, 1, model_config(arm), args,
                            data_states, manifest, protocol_states)
    continued_loss = run_update(model, optimizer, sampler)
    saved = load_full_checkpoint(path, map_location=CPU)
    resumed = build_model(arm)
    resumed_optimizer = optimizer_for(resumed)
    resumed_sampler = torch.Generator().manual_seed(1)

    def validate(payload, active_identity=identity):
        return protocol.validate_resume_checkpoint(
            payload, active_identity, resumed_optimizer, resumed_sampler, CPU, lambda step: 3e-4,
        )

    for key, other in (("arm", "other"), ("seed", 200), ("protocol_sha256", "other")):
        with pytest.raises(ValueError, match=key):
            validate(saved, {**identity, key: other})
    for key in ("data", "source", "runtime"):
        with pytest.raises(ValueError, match=key):
            validate(saved, {**identity, "lock": {**identity["lock"], key: {}}})
    for field, value, message in (
        ("step", 2, "step"), ("tokens_seen", 11, "sampled-token"),
        ("run_manifest_sha256", "other", "digest"), ("rng_states", [], "every rank"),
    ):
        with pytest.raises(ValueError, match=message):
            validate({**saved, "protocol_resume": {**saved["protocol_resume"], field: value}})
    wrong_optimizer = copy.deepcopy(saved["optimizer_state"])
    next(iter(wrong_optimizer["state"].values()))["step"] = torch.tensor(2.0)
    with pytest.raises(ValueError, match="optimizer state"):
        validate({**saved, "optimizer_state": wrong_optimizer})
    with pytest.raises(ValueError, match="--locked_protocol"):
        trainer.load_checkpoint(path, resumed, resumed_optimizer, None, CPU, resumed_sampler)
    assert not resumed_optimizer.state

    step, inherited, rngs = validate(saved)
    assert step == 1 and inherited == manifest
    new_directory = tmp_path / "resumed"
    new_directory.mkdir()
    protocol.inherit_initialization(evidence_path, new_directory, inherited, step)
    trainer.write_lu_run_manifest(new_directory / protocol.MANIFEST_NAME, inherited, inherited=True)
    assert json.loads((new_directory / protocol.MANIFEST_NAME).read_text()) == manifest
    assert protocol.verify_initialization(new_directory, inherited).is_file()

    trainer.apply_training_checkpoint(saved, resumed, resumed_optimizer, None)
    protocol.restore_training_rng(rngs[0], resumed_sampler, CPU)
    resumed_loss = run_update(resumed, resumed_optimizer, resumed_sampler)
    torch.testing.assert_close(continued_loss, resumed_loss, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA RNG state")
def test_restores_current_device_cuda_dropout_rng():
    device = torch.device("cuda", torch.cuda.current_device())
    sampler = torch.Generator().manual_seed(234)
    protocol.seed_training_rng(100, 0, device)
    rngs = protocol.capture_training_rng(sampler, device)
    expected = torch.rand(32, device=device)
    torch.rand(100, device=device)
    protocol.restore_training_rng(rngs, sampler, device)
    torch.testing.assert_close(torch.rand(32, device=device), expected, rtol=0, atol=0)
