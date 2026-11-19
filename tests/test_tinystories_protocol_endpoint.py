import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import evaluate_lu_protocol as endpoint
import tinystories_lu_protocol as contract
import tinystories_protocol_training as training
from lgma.synthetic import CharTokenizer
from lgma.transformer import TinyTransformerLM


def test_historical_scoring_preserves_blank_segments_and_window_boundaries(monkeypatch):
    tokenizer = CharTokenizer("\nabcdefxy")
    separator = "<|endoftext|>"
    spec = {
        "separator": separator, "context_length": 2, "stories": 3, "evaluated_tokens": 7,
        "story_ids_sha256": endpoint.ordered_json_sha256([0, 2, 3]),
    }
    stories = endpoint.select_stories(separator.join(["abcdef", "x", "\n\n", "xy"]), spec)
    assert stories == [(0, "abcdef"), (2, "\n\n"), (3, "xy")]

    seen_inputs, seen_targets = [], []
    cross_entropy = torch.nn.functional.cross_entropy

    def model(input_ids):
        assert input_ids.shape[0] == 1
        seen_inputs.append(tokenizer.decode(input_ids[0]))
        return torch.zeros(*input_ids.shape, tokenizer.vocab_size, dtype=torch.bfloat16)

    def record_cross_entropy(logits, targets, *, reduction):
        assert logits.dtype == torch.float32 and reduction == "sum"
        seen_targets.append(tokenizer.decode(targets))
        return cross_entropy(logits, targets, reduction=reduction)

    monkeypatch.setattr(endpoint.F, "cross_entropy", record_cross_entropy)
    records = endpoint.score_stories(model, tokenizer, stories, spec["context_length"], torch.device("cpu"))
    summary = endpoint.summarize_stories(records, spec)
    assert seen_inputs == ["ab", "cd", "e", "\n", "x"]
    assert seen_targets == ["bc", "de", "f", "\n", "y"]
    assert summary["evaluated_tokens"] == 7
    assert summary["windows"] == 5
    assert summary["nll"] == pytest.approx(math.log(tokenizer.vocab_size))
    assert summary["perplexity"] == pytest.approx(tokenizer.vocab_size)
    with pytest.raises(ValueError, match="incomplete"):
        endpoint.summarize_stories(records[:-1], spec)
    with pytest.raises(ValueError, match="order"):
        endpoint.summarize_stories(records[::-1], spec)


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.parametrize("arm", endpoint.ARMS)
def test_checkpoint_gate_binds_trainer_artifacts(tmp_path, monkeypatch, arm):
    protocol = contract.load_protocol()
    protocol["architecture"]["model_config"].update(d_model=16, num_layers=1, num_heads=4, head_dim=4)
    protocol["arms"]["gqa"]["model_config"]["num_kv_heads"] = 2
    protocol["arms"]["gt_mha_lu"]["model_config"].update(num_base_heads=2, base_dim=4, value_dim=4)
    monkeypatch.setattr(endpoint, "validate_source_fingerprints", lambda source: source)
    seed = 100
    tokenizer = training.build_locked_character_tokenizer(protocol["data"]["tokenizer"]["vocabulary"])
    config = contract.model_config_for(protocol, arm)
    model = training.initialize_locked_model(TinyTransformerLM, tokenizer.vocab_size, config, seed)
    frozen = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=frozen["arguments"]["lr"],
        weight_decay=frozen["arguments"]["weight_decay"], **training.ADAMW_OPTIONS,
    )
    data = {split: protocol["data"][split] for split in ("train", "validation")}
    data.update(
        tokenizer=contract.validate_tokenizer(protocol, tokenizer),
        train_tokens=data["train"]["decoded_characters"],
        validation_tokens=data["validation"]["decoded_characters"],
    )
    world_size = protocol["runtime"]["ranks"]
    identity = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol["protocol_sha256"],
        "arm": arm, "seed": seed, "lock": {
            "model_config": config, "effective_model": contract.validate_training_model(protocol, arm, model, optimizer),
            "training": frozen, "official_evaluation": protocol["official_evaluation"], "data": data, "source": {},
            "runtime": {
                "world_size": world_size, "effective_global_batch": frozen["effective_global_batch"],
                "tokens_per_step": frozen["targets_per_update"], "target_sampled_tokens": frozen["target_exposure"],
                "effective_precision": frozen["arguments"]["precision"], "parameter_dtype": "torch.float32",
                "rank_data_seeds": list(range(seed, seed + world_size)),
            },
        },
    }
    args = {
        **frozen["arguments"], "attention": arm, "seed": seed, "locked_protocol": True,
        "resume_checkpoint": None, "output_dir": tmp_path,
    }
    manifest = training.write_initialization(tmp_path, identity, model, args)
    (tmp_path / training.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    step = frozen["arguments"]["steps"]
    checkpoint = {
        "step": step, "run_manifest": manifest, "model_config": config, "model_state": model.state_dict(),
        "args": args, "protocol_resume": training.checkpoint_binding(step, manifest, []),
    }
    torch.save(checkpoint, tmp_path / "checkpoint_final.pt")
    _, provenance = endpoint.load_endpoint(protocol, arm, seed, tmp_path, tokenizer)
    assert provenance["step"] == step
    assert provenance["training_targets"] == frozen["target_exposure"]
    with pytest.raises(ValueError, match="pair slot"):
        endpoint.load_endpoint(protocol, arm, 0, tmp_path, tokenizer)
    checkpoint["protocol_resume"]["tokens_seen"] -= frozen["targets_per_update"]
    with pytest.raises(ValueError, match="incomplete training exposure"):
        endpoint.verify_training_endpoint(protocol, arm, seed, checkpoint, manifest)


def test_aggregation_requires_six_distinct_matched_endpoints(tmp_path, monkeypatch):
    protocol = contract.load_protocol()
    spec = protocol["official_evaluation"]
    spec.update(stories=1, evaluated_tokens=3, story_ids_sha256=endpoint.ordered_json_sha256([0]))
    monkeypatch.setattr(endpoint, "validate_source_fingerprints", lambda source: source)
    validation = protocol["data"]["validation"]
    inputs = {
        **{f"validation_{key}": validation[key] for key in ("size_bytes", "decoded_characters", "sha256")},
        "tokenizer": {key: protocol["data"]["tokenizer"][key] for key in endpoint.TOKENIZER_PROOF_KEYS},
    }
    runtime = {
        **protocol["runtime"], "device_count": 1, "cross_entropy_dtype": "float32", "deterministic_algorithms": True,
    }
    for index, seed in enumerate(endpoint.SEEDS):
        for arm in endpoint.ARMS:
            tag = f"{arm}{seed}"
            nll = 1 + (index - 1) / 10 if arm == "gt_mha_lu" else 1
            rows = [{"story_id": 0, "tokens": 3, "windows": 1, "nll_sum": nll * 3}]
            record = {
                "schema_version": 1, "status": "complete", "arm": arm, "seed": seed, "inputs": inputs,
                "source": {}, "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol["protocol_sha256"],
                "evaluation": spec, "runtime": runtime, "per_story": rows,
                "metrics": endpoint.summarize_stories(rows, spec),
                "checkpoint": {
                    "source": {}, "step": spec["checkpoint_step"], "sha256": digest(f"checkpoint{tag}"),
                    "run_manifest_sha256": digest(f"manifest{tag}"), "run_manifest_file_sha256": digest(f"file{tag}"),
                    "training_targets": protocol["training"]["target_exposure"],
                    "model_config": contract.model_config_for(protocol, arm),
                    "initialization": {
                        "step": 0, "initialization_kind": "fresh", "sha256": digest(f"init{tag}"),
                        "common_parameters_sha256": digest(f"common{seed}"),
                    },
                },
            }
            output = endpoint.result_path(tmp_path, arm, seed)
            endpoint.publish_json(output, record)
    args = SimpleNamespace(results_dir=tmp_path)
    path = endpoint.result_path(tmp_path, "gqa", 200)
    original = endpoint.read_json(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        endpoint.aggregate(protocol, args)
    assert not (tmp_path / "paired_report.json").exists()

    swapped = {**original, "seed": 100}
    endpoint.publish_json(path, swapped)
    with pytest.raises(ValueError, match="pair slot"):
        endpoint.aggregate(protocol, args)
    path.unlink()
    duplicate = endpoint.read_json(endpoint.result_path(tmp_path, "gqa", 100))
    original_sha = original["checkpoint"]["sha256"]
    original["checkpoint"]["sha256"] = duplicate["checkpoint"]["sha256"]
    endpoint.publish_json(path, original)
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        endpoint.aggregate(protocol, args)
    path.unlink()
    original["checkpoint"]["sha256"] = original_sha
    endpoint.publish_json(path, original)
    endpoint.aggregate(protocol, args)
    result = endpoint.read_json(tmp_path / "paired_report.json")
    assert len(result["per_endpoint"]) == 6
    contrast = result["paired_lu_minus_gqa4"]
    nll = contrast["statistics"]["nll"]
    assert contrast["interval"]["degrees_of_freedom"] == 2
    assert nll["mean"] == pytest.approx(0, abs=1e-15)
    assert nll["sample_std"] == pytest.approx(0.1)
    assert nll["ci95"] == pytest.approx([-0.24841377, 0.24841377])
