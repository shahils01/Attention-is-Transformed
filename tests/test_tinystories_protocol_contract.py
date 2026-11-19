from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import tinystories_lu_protocol as lock
import train_tinystories as trainer
from lgma.synthetic import CharTokenizer
from lgma.transformer import TinyTransformerLM


def launch_options(tmp_path, arm):
    return argparse.Namespace(
        arm=arm, seed=100, train_path=tmp_path / "training text.txt",
        val_path=tmp_path / "validation text.txt", output_dir=tmp_path / arm,
        resume_checkpoint=None, **dict.fromkeys(lock.TRACKING_OPTIONS),
    )


@pytest.mark.parametrize("arm", ["gt_mha_lu", "gqa"])
def test_launch_command_resolves_to_frozen_trainer_configuration(tmp_path, monkeypatch, arm):
    protocol = lock.load_protocol()
    command = lock.build_training_command(protocol, launch_options(tmp_path, arm))
    assert command[:8] == [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes", "1",
        "--nproc_per_node", "4",
    ]
    monkeypatch.setattr(sys, "argv", command[8:])
    args = trainer.parse_args()
    config = trainer.model_config_from_args(args)
    assert args.locked_protocol
    lock.validate_training_args(protocol, args, config)
    args.batch_size = 128
    with pytest.raises(SystemExit, match="batch_size"):
        lock.validate_training_args(protocol, args, config)
    args.batch_size = 256
    with pytest.raises(SystemExit, match="num_heads"):
        lock.validate_training_args(protocol, args, {**config, "num_heads": 8})


def test_lock_and_source_tampering_fails_before_launch(tmp_path, monkeypatch):
    protocol_path = tmp_path / lock.PROTOCOL_PATH.name
    protocol_path.write_bytes(lock.PROTOCOL_PATH.read_bytes() + b" ")
    monkeypatch.setattr(lock, "PROTOCOL_PATH", protocol_path)
    with pytest.raises(SystemExit, match="frozen protocol SHA-256"):
        lock.load_protocol()

    source = tmp_path / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    receipt = tmp_path / "sources.json"
    receipt.write_text(json.dumps({source.name: lock.file_sha256(source)}), encoding="utf-8")
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(lock, "SOURCE_PATHS", (source.name,))
    monkeypatch.setattr(lock, "SOURCE_RECEIPT_PATH", receipt)
    assert lock.validate_source_fingerprints(lock.source_fingerprints())["files_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="source files.+example.py"):
        lock.validate_source_fingerprints(lock.source_fingerprints())


def test_print_command_is_inspection_only(tmp_path, monkeypatch, capsys):
    def unexpected(*args):
        pytest.fail("printing the command must not check CUDA or launch a process")

    monkeypatch.setattr(lock, "source_fingerprints", unexpected)
    monkeypatch.setattr(lock.os, "execv", unexpected)
    lock.main([
        "launch", "--print", "--arm", "gqa", "--seed", "0",
        "--train-path", str(tmp_path / "train file.txt"), "--val-path", str(tmp_path / "val.txt"),
        "--output-dir", str(tmp_path / "gqa0"),
    ])
    output = capsys.readouterr().out
    assert "--locked_protocol --seed 0" in output
    assert "--num_kv_heads 4" in output
    assert "train file.txt'" in output


def test_data_lock_and_ordered_tokenizer(tmp_path):
    protocol = load_small_data_lock(tmp_path)
    train_text = tmp_path / "train.txt"
    val_text = tmp_path / "val.txt"
    vocab = protocol["data"]["tokenizer"]["vocabulary"]
    tokenizer = CharTokenizer("".join(vocab))
    proof = lock.validate_training_data(protocol, train_text, val_text, tokenizer)
    assert proof["tokenizer"]["vocab_size"] == 230
    assert proof["tokenizer"]["json_vocab_sha256"] == (
        "f74733ab93cca642f83803a1f65bb73fd7acfb77f0053719f9e2614d3b215815"
    )
    tokenizer.stoi[vocab[0]] = 1
    with pytest.raises(SystemExit, match="frozen ordered character mapping"):
        lock.validate_tokenizer(protocol, tokenizer)
    train_text.write_text("changed!", encoding="utf-8")  # same byte length
    with pytest.raises(SystemExit, match="train text.+sha256"):
        lock.validate_training_data(protocol, train_text, val_text, tokenizer)


def load_small_data_lock(tmp_path):
    protocol = lock.load_protocol()
    for split, name in (("train", "train.txt"), ("validation", "val.txt")):
        path = tmp_path / name
        path.write_text("original", encoding="utf-8")
        protocol["data"][split].update(size_bytes=8, sha256=lock.file_sha256(path))
    return protocol


@pytest.mark.parametrize("arm", ["gt_mha_lu", "gqa"])
def test_operational_model_structure_and_adamw_guard(arm):
    protocol = copy.deepcopy(lock.load_protocol())
    protocol["architecture"]["model_config"].update(
        d_model=16, num_layers=2, num_heads=8, head_dim=2, context_length=12,
    )
    protocol["arms"]["gt_mha_lu"]["model_config"].update(base_dim=2, value_dim=2)
    config = lock.model_config_for(protocol, arm)
    model = TinyTransformerLM(vocab_size=230, **config)
    frozen = protocol["training"]["optimizer"]
    options = {key: value for key, value in frozen.items() if key not in {"type", "all_parameters"}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01, **options)
    report = lock.validate_training_model(protocol, arm, model, optimizer)
    assert len(report["attention_layers"]) == 2
    assert report["optimizer"] == {**frozen, "weight_decay": 0.01}
    optimizer.param_groups[0]["weight_decay"] = 0
    with pytest.raises(SystemExit, match="optimizer.+weight_decay"):
        lock.validate_training_model(protocol, arm, model, optimizer)
    model.blocks[-1].attn.dropout = 0.0
    with pytest.raises(SystemExit, match="layer 1 attention.+dropout"):
        lock.validate_model_structure(protocol, arm, model)


def test_exposure_is_derived_from_effective_batch_and_schedule():
    protocol = lock.load_protocol()
    training = protocol["training"]
    args = training["arguments"]
    global_batch = protocol["runtime"]["ranks"] * args["batch_size"] * args["grad_accum_steps"]
    updates = args["steps"]
    assert global_batch == training["effective_global_batch"] == 2048
    assert training["targets_per_update"] == global_batch * 512 == 1_048_576
    assert training["target_exposure"] == training["targets_per_update"] * updates == 262_144_000_000
    assert updates == args["lr_schedule_steps"] == protocol["official_evaluation"]["checkpoint_step"]


def test_cuda_guard_checks_actual_runtime_versions_and_tf32(monkeypatch):
    from types import SimpleNamespace

    from lgma import lu_kernel

    protocol = lock.load_protocol()
    expected = protocol["runtime"]
    hardware = SimpleNamespace(name=expected["gpu_name"], total_memory=120_000_000_000, major=9, minor=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: hardware)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)
    monkeypatch.setattr(torch, "__version__", expected["torch"])
    monkeypatch.setattr(torch.version, "cuda", expected["cuda_build"])
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: expected["fp32_matmul_precision"])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(lu_kernel, "triton", SimpleNamespace(__version__=expected["triton"]))
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    device = torch.device("cuda:0")
    settings = {"gpu_name": expected["gpu_name"], "device_count": 4}
    assert lock.validate_cuda_device(device, **settings)["compute_capability"] == [9, 0]
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    with pytest.raises(SystemExit, match="CUDA runtime.+tf32"):
        lock.validate_cuda_device(device, **settings)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(SystemExit, match="CUDA runtime.+torch"):
        lock.validate_cuda_device(device, **settings)
