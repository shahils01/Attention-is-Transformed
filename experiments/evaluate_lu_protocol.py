"""Evaluate and summarize the six locked TinyStories LU versus GQA-4 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from pathlib import Path, PosixPath

import torch
import torch.nn.functional as F

from tinystories_runtime import precision_context
from tinystories_lu_protocol import (
    file_sha256,
    load_protocol,
    require_values,
    source_fingerprints,
    validate_cuda_device,
    validate_model_config,
    validate_model_structure,
    validate_source_fingerprints,
    validate_tokenizer,
)
from tinystories_protocol_training import (
    IDENTITY_KEYS,
    INITIALIZATION_NAME,
    MANIFEST_NAME,
    build_locked_character_tokenizer,
    common_initialization_sha256,
    manifest_sha256,
)
from lgma.transformer import TinyTransformerLM


ARMS = ("gt_mha_lu", "gqa")
SEEDS = (0, 100, 200)
METRICS = ("nll", "perplexity")
TOKENIZER_PROOF_KEYS = ("vocab_size", "vocab_sha256", "algorithm", "json_vocab_sha256")
T_975_DF2 = 4.302652729696142


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def publish_json(path, payload):
    """Publish a complete record atomically and preserve any prior result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True, allow_nan=False)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        os.link(tmp.name, path)


def result_path(results_dir, arm, seed):
    return results_dir / f"{arm}_seed{seed}.json"


def ordered_json_sha256(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def select_stories(text, spec):
    # Keep all original story characters, including surrounding newlines.
    stories = [(index, story) for index, story in enumerate(text.split(spec["separator"]))
               if len(story) > 1]
    require(len(stories) == spec["stories"], "validation story count differs from the protocol")
    require(sum(len(story) - 1 for _, story in stories) == spec["evaluated_tokens"],
            "validation next-character target count differs from the protocol")
    require(ordered_json_sha256([index for index, _ in stories]) == spec["story_ids_sha256"],
            "validation story order differs from the protocol")
    return stories


def validation_inputs(protocol, data_dir):
    locked = protocol["data"]["validation"]
    path = data_dir / locked["filename"]
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    require(actual_sha == locked["sha256"] and len(raw) == locked["size_bytes"],
            f"{path} differs from the locked validation file")
    # Match TextIO's universal newline handling in the November evaluation.
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    require(len(text) == locked["decoded_characters"], "validation decoded character count differs")
    tokenizer_lock = protocol["data"]["tokenizer"]
    tokenizer = build_locked_character_tokenizer(tokenizer_lock["vocabulary"])
    proof = validate_tokenizer(protocol, tokenizer)
    stories = select_stories(text, protocol["official_evaluation"])
    return tokenizer, stories, {
        "validation_sha256": actual_sha,
        "validation_size_bytes": len(raw),
        "validation_decoded_characters": len(text),
        "whitespace_only_segments": sum(not story.strip() for _, story in stories),
        "tokenizer": proof,
    }


@torch.inference_mode()
def score_stories(model, tokenizer, stories, context_length, device, *, progress=None):
    rows = []
    for index, story in stories:
        encoded = tokenizer.encode(story).to(device)
        targets = encoded.numel() - 1
        losses = []
        for start in range(0, targets, context_length):
            end = min(start + context_length, targets)
            with precision_context(device, "bf16"):
                logits = model(encoded[start:end].unsqueeze(0))
            loss = F.cross_entropy(
                logits.float().reshape(-1, tokenizer.vocab_size),
                encoded[start + 1:end + 1],
                reduction="sum",
            )
            losses.append(loss.item())
        nll_sum = math.fsum(losses)
        require(math.isfinite(nll_sum) and nll_sum >= 0, f"nonfinite loss at validation story {index}")
        rows.append({"story_id": index, "nll_sum": nll_sum, "tokens": targets, "windows": len(losses)})
        if progress is not None and len(rows) % 1000 == 0:
            progress(len(rows))
    return rows


def summarize_stories(rows, spec):
    require(len(rows) == spec["stories"], "incomplete per-story evaluation")
    digest = ordered_json_sha256([row["story_id"] for row in rows])
    require(digest == spec["story_ids_sha256"], "per-story order or identities differ from the protocol")
    context = spec["context_length"]
    for row in rows:
        count = row["tokens"]
        require(type(count) is int and count > 0 and row["windows"] == (count + context - 1) // context
                and math.isfinite(row["nll_sum"]) and row["nll_sum"] >= 0,
                f"invalid loss or target coverage for story {row['story_id']}")
    targets = sum(row["tokens"] for row in rows)
    require(targets == spec["evaluated_tokens"], "per-story target count differs from the protocol")
    nll_sum = math.fsum(row["nll_sum"] for row in rows)
    nll = nll_sum / targets
    return {
        "stories": len(rows), "evaluated_tokens": targets,
        "windows": sum(row["windows"] for row in rows), "story_ids_sha256": digest,
        "target_counts_sha256": ordered_json_sha256([row["tokens"] for row in rows]),
        "nll_sum": nll_sum, "nll": nll, "perplexity": math.exp(nll),
    }


def validate_locked_data(protocol, recorded):
    for split in ("train", "validation"):
        for key in ("sha256", "size_bytes"):
            require(recorded[split][key] == protocol["data"][split][key],
                    f"run manifest {split} {key} differs from the protocol")
        require(recorded[f"{split}_tokens"] == protocol["data"][split]["decoded_characters"],
                f"run manifest {split} character count differs from the protocol")
    for key in TOKENIZER_PROOF_KEYS:
        require(recorded["tokenizer"][key] == protocol["data"]["tokenizer"][key],
                f"run manifest tokenizer {key} differs from the protocol")


def verify_training_endpoint(protocol, arm, seed, checkpoint, manifest):
    spec = protocol["official_evaluation"]
    training = protocol["training"]
    step, last = checkpoint["step"], checkpoint["protocol_resume"]
    require(step == last["step"] == spec["checkpoint_step"], "checkpoint has an incorrect endpoint")
    require(last["tokens_seen"] == training["target_exposure"], "checkpoint has incomplete training exposure")
    require(manifest == checkpoint["run_manifest"] and manifest["schema_version"] == 2,
            "checkpoint and archived run manifest differ")
    require(last["run_manifest_sha256"] == manifest_sha256(manifest),
            "checkpoint manifest digest differs")
    for key in ("protocol_id", "protocol_sha256"):
        require(manifest[key] == protocol[key], f"checkpoint {key} differs from the protocol")
    launch = manifest["launch_args"]
    saved = checkpoint["args"]
    require(manifest["arm"] == launch["attention"] == saved["attention"] == arm
            and manifest["seed"] == launch["seed"] == saved["seed"] == seed,
            "checkpoint arm or seed differs from the requested pair slot")
    for args in (launch, saved):
        require(args["locked_protocol"] is True, "checkpoint lacks a locked training launch")
        require_values("checkpoint training arguments", training["arguments"], args)
    init = manifest["initialization"]
    require(init["step"] == 0 and init["initialization_kind"] == "fresh"
            and launch["resume_checkpoint"] is None and init["filename"] == INITIALIZATION_NAME,
            "checkpoint does not descend from a locked fresh step-zero launch")
    require(init["common_parameters_algorithm"] == training["paired_initialization"],
            "checkpoint does not use the locked paired initialization algorithm")
    lock = manifest["lock"]
    require(lock["training"] == training and lock["official_evaluation"] == spec,
            "checkpoint training or evaluation lock differs from the protocol")
    require_values("checkpoint training runtime", {
        "world_size": protocol["runtime"]["ranks"],
        "effective_global_batch": training["effective_global_batch"],
        "tokens_per_step": training["targets_per_update"],
        "target_sampled_tokens": training["target_exposure"],
        "effective_precision": training["arguments"]["precision"],
        "parameter_dtype": "torch.float32",
        "rank_data_seeds": [seed + rank for rank in range(protocol["runtime"]["ranks"])],
    }, lock["runtime"])
    validate_locked_data(protocol, lock["data"])
    validate_source_fingerprints(lock["source"])
    config = checkpoint["model_config"]
    require(config == lock["model_config"], "checkpoint model configuration differs from the run manifest")
    validate_model_config(protocol, arm, config)


def load_endpoint(protocol, arm, seed, run_dir, tokenizer):
    path = run_dir / "checkpoint_final.pt"
    checkpoint_sha = file_sha256(path)
    # Trainer arguments include pathlib objects for checkpoint/output paths.
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    manifest_path = run_dir / MANIFEST_NAME
    manifest_contents = manifest_path.read_bytes()
    manifest = json.loads(manifest_contents)
    verify_training_endpoint(protocol, arm, seed, checkpoint, manifest)

    initialization = manifest["initialization"]
    init_path = run_dir / initialization["filename"]
    require(file_sha256(init_path) == initialization["sha256"], "step-zero artifact digest differs")
    step_zero = torch.load(init_path, map_location="cpu", mmap=True, weights_only=True)
    for key in ("arm", "seed", "protocol_id", "protocol_sha256"):
        require(step_zero[key] == manifest[key], f"step-zero artifact has a different {key}")
    require(step_zero["schema_version"] == 1 and step_zero["step"] == 0
            and step_zero["model_config"] == checkpoint["model_config"]
            and step_zero["run_identity_sha256"] == manifest_sha256({key: manifest[key] for key in IDENTITY_KEYS})
            and step_zero["common_parameters_sha256"] == initialization["common_parameters_sha256"]
            == common_initialization_sha256(step_zero["model_state"]),
            "step-zero artifact configuration differs from the endpoint run manifest")
    del step_zero

    config = checkpoint["model_config"]
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, **config)
    effective_model = validate_model_structure(protocol, arm, model)
    training = protocol["training"]
    expected_optimizer = {**training["optimizer"], "weight_decay": training["arguments"]["weight_decay"]}
    require({**effective_model, "optimizer": expected_optimizer} == manifest["lock"]["effective_model"],
            "instantiated evaluation model differs from the locked training model")
    state = checkpoint["model_state"]
    require(all(state[name].dtype == parameter.dtype for name, parameter in model.named_parameters()),
            "checkpoint parameter dtypes differ from the locked FP32 model")
    model.load_state_dict(state, strict=True)
    require(file_sha256(path) == checkpoint_sha, "checkpoint changed while it was being loaded")
    return model, {
        "path": str(path.resolve()), "sha256": checkpoint_sha, "step": checkpoint["step"],
        "training_targets": checkpoint["protocol_resume"]["tokens_seen"],
        "model_config": config, "effective_model": effective_model,
        "run_manifest_sha256": manifest_sha256(manifest),
        "run_manifest_file_sha256": hashlib.sha256(manifest_contents).hexdigest(),
        "source": manifest["lock"]["source"], "initialization": initialization,
        "training_runtime": manifest["lock"]["runtime"],
    }


def evaluation_device(spec):
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "official evaluation requires one process")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(spec["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    runtime = validate_cuda_device(device, gpu_name=spec["hardware"], device_count=spec["device_count"])
    return device, {
        **runtime, "device_count": torch.cuda.device_count(), "autocast": spec["autocast"],
        "cross_entropy_dtype": spec["cross_entropy_dtype"], "tf32": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def evaluate(protocol, args):
    output = result_path(args.results_dir, args.arm, args.seed)
    require(not output.exists(), f"an evaluation already exists at {output}")
    source = source_fingerprints()
    validate_source_fingerprints(source)
    spec = protocol["official_evaluation"]
    device, runtime = evaluation_device(spec)
    tokenizer, stories, inputs = validation_inputs(protocol, args.data_dir)
    model, checkpoint = load_endpoint(protocol, args.arm, args.seed, args.run_dir, tokenizer)
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    started = time.monotonic()

    def progress(done):
        print(json.dumps({"arm": args.arm, "seed": args.seed, "stories_done": done,
                          "total_stories": len(stories)}), flush=True)

    rows = score_stories(model, tokenizer, stories, spec["context_length"], device, progress=progress)
    torch.cuda.synchronize(device)
    metrics = summarize_stories(rows, spec)
    record = {
        "schema_version": 1, "status": "complete", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"], "arm": args.arm, "seed": args.seed,
        "inputs": inputs, "source": source, "checkpoint": checkpoint, "evaluation": spec,
        "runtime": runtime, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_seconds": time.monotonic() - started, "metrics": metrics, "per_story": rows,
    }
    publish_json(output, record)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "result": str(output), **metrics}), flush=True)


def require_sha256(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"missing or invalid {name} digest")


def verify_evaluation(protocol, arm, seed, result):
    require(result["schema_version"] == 1 and result["status"] == "complete",
            f"evaluation for {arm}/{seed} is incomplete")
    require(result["arm"] == arm and result["seed"] == seed,
            f"evaluation record differs from its pair slot {arm}/{seed}")
    spec = protocol["official_evaluation"]
    inputs = result["inputs"]
    require(result["protocol_id"] == protocol["protocol_id"]
            and result["protocol_sha256"] == protocol["protocol_sha256"]
            and result["evaluation"] == spec, "evaluation protocol differs from the lock")
    for key in ("sha256", "size_bytes", "decoded_characters"):
        require(inputs[f"validation_{key}"] == protocol["data"]["validation"][key],
                f"evaluation validation {key} differs from the lock")
    for key in TOKENIZER_PROOF_KEYS:
        require(inputs["tokenizer"][key] == protocol["data"]["tokenizer"][key],
                f"evaluation tokenizer {key} differs from the lock")
    runtime = result["runtime"]
    require(runtime["autocast"] == spec["autocast"] and runtime["tf32"] is False
            and runtime["deterministic_algorithms"] is True
            and runtime["cross_entropy_dtype"] == spec["cross_entropy_dtype"]
            and runtime["device_count"] == spec["device_count"]
            and runtime["gpu_name"] == spec["hardware"], "evaluation runtime differs from the lock")
    require_values("evaluation CUDA runtime", {
        key: protocol["runtime"][key]
        for key in ("torch", "cuda_build", "triton", "compute_capability", "fp32_matmul_precision")
    }, runtime)
    checkpoint = result["checkpoint"]
    require(checkpoint["step"] == spec["checkpoint_step"]
            and checkpoint["training_targets"] == protocol["training"]["target_exposure"],
            "evaluation checkpoint has incomplete training exposure")
    validate_model_config(protocol, arm, checkpoint["model_config"])
    for key in ("sha256", "run_manifest_sha256", "run_manifest_file_sha256"):
        require_sha256(checkpoint[key], f"checkpoint {key}")
    init = checkpoint["initialization"]
    require(init["initialization_kind"] == "fresh" and init["step"] == 0,
            "evaluation checkpoint has no fresh step-zero provenance")
    require_sha256(init["sha256"], "step-zero identity")
    require_sha256(init["common_parameters_sha256"], "paired initialization")
    validate_source_fingerprints(result["source"])
    validate_source_fingerprints(checkpoint["source"])
    metrics = summarize_stories(result["per_story"], spec)
    for key, value in metrics.items():
        expected = result["metrics"][key]
        agrees = math.isclose(value, expected, rel_tol=1e-13) if isinstance(value, float) else value == expected
        require(agrees, f"summary {key} differs from the archived per-story scores")
    return metrics


def sample_stats(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def paired_interval(values):
    require(len(values) == len(SEEDS), "the paired 95% interval requires all three seeds")
    mean, std = statistics.mean(values), statistics.stdev(values)
    half_width = T_975_DF2 * std / math.sqrt(len(values))
    return {"mean": mean, "sample_std": std, "ci95": [mean - half_width, mean + half_width]}


def paired_statistics(records):
    require(set(records) == {(arm, seed) for arm in ARMS for seed in SEEDS},
            "paired analysis requires exactly the six locked arm/seed slots")
    for key in ("sha256", "run_manifest_sha256", "run_manifest_file_sha256"):
        require(len({record["checkpoint"][key] for record in records.values()}) == len(records),
                f"duplicate checkpoint or run manifest {key} in paired inputs")
    require(len({record["checkpoint"]["initialization"]["sha256"] for record in records.values()})
            == len(records), "duplicate step-zero artifact in paired inputs")
    require(all(record["runtime"] == records[ARMS[0], SEEDS[0]]["runtime"] for record in records.values()),
            "evaluation runtime differs across pair slots")
    require(len({record["metrics"]["target_counts_sha256"] for record in records.values()}) == 1,
            "per-story target coverage differs across pair slots")
    common_hashes = set()
    pairs = []
    for seed in SEEDS:
        lu, gqa = (records[arm, seed] for arm in ARMS)
        lu_init, gqa_init = (record["checkpoint"]["initialization"] for record in (lu, gqa))
        common = lu_init["common_parameters_sha256"]
        require(common == gqa_init["common_parameters_sha256"] and common not in common_hashes,
                f"shared step-zero parameters do not match the distinct seed pair {seed}")
        common_hashes.add(common)
        pairs.append({"seed": seed, "common_parameters_sha256": common, **{
            metric: lu["metrics"][metric] - gqa["metrics"][metric] for metric in METRICS
        }})
    return {
        "across_seeds": {arm: {
            metric: sample_stats([records[arm, seed]["metrics"][metric] for seed in SEEDS])
            for metric in METRICS
        } for arm in ARMS},
        "paired_lu_minus_gqa4": {
            "per_seed": pairs,
            "statistics": {metric: paired_interval([pair[metric] for pair in pairs]) for metric in METRICS},
            "interval": {"confidence": 0.95, "distribution": "paired Student t", "degrees_of_freedom": 2},
        },
    }


def aggregate(protocol, args):
    output = args.results_dir / "paired_report.json"
    require(not output.exists(), f"a paired report already exists at {output}")
    records = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = result_path(args.results_dir, arm, seed)
            record = read_json(path)
            metrics = verify_evaluation(protocol, arm, seed, record)
            record.pop("per_story")
            record["metrics"] = metrics
            record["result_sha256"] = file_sha256(path)
            records[arm, seed] = record
    report = {
        "schema_version": 1, "status": "complete", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"], "seeds": list(SEEDS),
        "primary_metric": "token-weighted validation NLL", "sample_std_ddof": 1,
        "per_endpoint": list(records.values()), **paired_statistics(records),
        "interpretation": "Three seed pairs give two degrees of freedom and limited precision. "
                          "The paired intervals describe uncertainty across training seeds.",
    }
    publish_json(output, report)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("evaluate", help="evaluate one verified endpoint checkpoint")
    score.add_argument("--arm", required=True, choices=ARMS)
    score.add_argument("--seed", type=int, required=True, choices=SEEDS)
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--data-dir", type=Path, required=True)
    paired = subparsers.add_parser("aggregate", help="aggregate the six completed evaluations")
    for subparser in (score, paired):
        subparser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    require(set(protocol["arms"]) == set(ARMS) and tuple(protocol["seeds"]) == SEEDS,
            "the locked arms or seeds differ from the evaluator")
    try:
        if args.command == "evaluate":
            evaluate(protocol, args)
        else:
            aggregate(protocol, args)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
