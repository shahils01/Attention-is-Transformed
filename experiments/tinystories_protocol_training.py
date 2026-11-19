"""Training state and immutable artifacts for the locked TinyStories comparison."""

from __future__ import annotations

import codecs
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from lgma.synthetic import CharTokenizer


MANIFEST_NAME = "run_manifest.json"
INITIALIZATION_NAME = "initialization_step_0.pt"
COMMON_INITIALIZATION = "sha256_qualified_module_seed_v1"
COMMON_ATTENTION_MODULES = {"k_proj", "v_proj", "out_proj"}
IDENTITY_KEYS = ("protocol_id", "protocol_sha256", "arm", "seed", "lock")
ADAMW_OPTIONS = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "amsgrad": False,
    "foreach": False,
    "maximize": False,
    "capturable": False,
    "differentiable": False,
    "fused": False,
}


def build_locked_character_tokenizer(characters: list[str]) -> CharTokenizer:
    tokenizer = CharTokenizer("".join(characters))
    if tokenizer.itos != characters or tokenizer.vocab_size > 256:
        raise ValueError("the locked character vocabulary must be sorted, unique, and byte-sized")
    return tokenizer


def encode_locked_character_text(
    path: Path, tokenizer: CharTokenizer, *, expected_sha256: str | None = None,
) -> torch.Tensor:
    """Match read_text UTF-8/newline semantics with about one byte per character."""
    path = Path(path)
    encoded = torch.empty(path.stat().st_size, dtype=torch.uint8)
    translation = {ord(char): token for char, token in tokenizer.stoi.items()}
    # Missing characters must error even when their code points happen to fit Latin-1.
    translation.update({codepoint: None for codepoint in range(256) if codepoint not in translation})
    decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")(), translate=True)
    digest = hashlib.sha256()
    cursor = 0
    with path.open("rb") as source:
        while True:
            raw = source.read(1024 * 1024)
            if expected_sha256 is not None:
                digest.update(raw)
            chunk = decoder.decode(raw, final=not raw)
            try:
                data = bytearray(chunk.translate(translation), "latin-1")
            except UnicodeEncodeError as error:
                raise ValueError(f"{path} contains characters outside the locked vocabulary") from error
            if len(data) != len(chunk):
                raise ValueError(f"{path} contains characters outside the locked vocabulary")
            if data:
                end = cursor + len(data)
                encoded[cursor:end].copy_(torch.frombuffer(data, dtype=torch.uint8))
                cursor = end
            if not raw:
                break
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError(f"{path} changed before or during locked training data encoding")
    return encoded[:cursor]


def qualified_seed(seed: int, role: str) -> int:
    digest = hashlib.sha256(f"tinystories-lu-pair-v1:{seed}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little")


def initialize_locked_model(model_type, vocab_size: int, config: dict, seed: int):
    """Keep native query/LU initialization and share qualified backbone and K/V/O draws."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(qualified_seed(seed, f"model:{config['attention_type']}"))
        model = model_type(vocab_size=vocab_size, **config)
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Embedding, nn.Linear, nn.LayerNorm)):
                continue
            attention_projection = ".attn." in name
            if attention_projection and name.rsplit(".", 1)[1] not in COMMON_ATTENTION_MODULES:
                continue
            torch.random.default_generator.manual_seed(qualified_seed(seed, name))
            if attention_projection:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                module.reset_parameters()
    return model


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_initialization_sha256(model_state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model_state.items()):
        if ".attn." in name and name.rsplit(".", 2)[1] not in COMMON_ATTENTION_MODULES:
            continue
        tensor = tensor.detach().cpu().contiguous()
        digest.update((canonical_json([name, list(tensor.shape), str(tensor.dtype)]) + "\n").encode())
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def seed_training_rng(seed: int, rank: int, device: torch.device) -> None:
    dropout_seed = qualified_seed(seed, f"training_dropout:{rank}")
    torch.random.default_generator.manual_seed(dropout_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(dropout_seed)


def capture_training_rng(data_generator: torch.Generator, device: torch.device) -> dict:
    return {
        "sampler": data_generator.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def gather_training_rng(
    data_generator: torch.Generator, device: torch.device, rank: int, world_size: int,
) -> list[dict] | None:
    local = capture_training_rng(data_generator, device)
    if world_size == 1:
        return [local]
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


def restore_training_rng(states: dict, data_generator: torch.Generator, device: torch.device) -> None:
    data_generator.set_state(states["sampler"])
    torch.random.set_rng_state(states["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(states["cuda"], device=device)


def _validate_rng_states(states: object, current: dict, world_size: int) -> None:
    if not isinstance(states, list) or len(states) != world_size:
        raise ValueError("locked checkpoint must carry RNG state for every rank")
    for rank, state in enumerate(states):
        if not isinstance(state, dict) or state.keys() != current.keys():
            raise ValueError(f"locked checkpoint RNG state is incomplete for rank {rank}")
        for key, expected in current.items():
            observed = state[key]
            if expected is None and observed is None:
                continue
            if not (
                expected is not None and torch.is_tensor(observed)
                and observed.device.type == "cpu" and observed.dtype == torch.uint8
                and observed.shape == expected.shape
            ):
                raise ValueError(f"locked checkpoint {key} RNG state differs for rank {rank}")
        verifier = torch.Generator()
        for key in ("sampler", "cpu"):
            verifier.set_state(state[key])


def write_initialization(
    directory: Path, identity: dict, model: nn.Module, launch_args: dict,
) -> dict:
    manifest_path = directory / MANIFEST_NAME
    artifact = directory / INITIALIZATION_NAME
    if (
        manifest_path.exists() or artifact.exists() or (directory / "metrics.jsonl").exists()
        or (directory / "checkpoint_final.pt").exists() or any(directory.glob("checkpoint_step_*.pt"))
    ):
        raise ValueError(f"{directory} already contains a training run. Use a new --output_dir.")
    state = model.state_dict()
    common_sha256 = common_initialization_sha256(state)
    with tempfile.NamedTemporaryFile(dir=directory) as temporary:
        torch.save({
            "schema_version": 1,
            "step": 0,
            **{key: identity[key] for key in IDENTITY_KEYS if key != "lock"},
            "model_config": identity["lock"]["model_config"],
            "model_state": state,
            "common_parameters_sha256": common_sha256,
            "run_identity_sha256": manifest_sha256(identity),
        }, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, artifact)
    initialization = {
        "step": 0,
        "initialization_kind": "fresh",
        "recipe": (
            "fixed_base_identity_near_identity_lu" if identity["arm"] == "gt_mha_lu"
            else "grouped_query_xavier_uniform"
        ),
        "filename": INITIALIZATION_NAME,
        "sha256": file_sha256(artifact),
        "common_parameters_algorithm": COMMON_INITIALIZATION,
        "common_parameters_sha256": common_sha256,
    }
    return json.loads(canonical_json({
        "schema_version": 2,
        **identity,
        "initialization": initialization,
        "launch_args": launch_args,
    }))


def verify_initialization(directory: Path, manifest: dict) -> Path:
    saved_manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if saved_manifest != manifest:
        raise ValueError("locked checkpoint disagrees with its sibling run manifest")
    artifact = directory / INITIALIZATION_NAME
    if file_sha256(artifact) != manifest["initialization"]["sha256"]:
        raise ValueError("locked checkpoint step-zero artifact hash differs from the run manifest")
    return artifact


def inherit_initialization(source: Path, directory: Path, manifest: dict, step: int) -> None:
    checkpoints = sorted(directory.glob("checkpoint_step_*.pt"))
    if (directory / "checkpoint_final.pt").exists() or any(
        int(path.stem.removeprefix("checkpoint_step_")) > step
        for path in checkpoints if path.stem.removeprefix("checkpoint_step_").isdigit()
    ):
        raise ValueError("the output directory already contains a later checkpoint; use a new output directory")
    if source.parent.resolve() != directory.resolve():
        target = directory / INITIALIZATION_NAME
        destination_manifest = directory / MANIFEST_NAME
        if destination_manifest.exists():
            if json.loads(destination_manifest.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"{directory} already contains a different run manifest")
        elif checkpoints or (directory / "metrics.jsonl").exists() or (directory / "final_report.json").exists():
            raise ValueError(f"{directory} already contains training output without matching lineage")
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=directory) as temporary:
                with source.open("rb") as source_file:
                    shutil.copyfileobj(source_file, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.link(temporary.name, target)
        if file_sha256(target) != manifest["initialization"]["sha256"]:
            raise ValueError(f"{directory} already contains another step-zero artifact")


def checkpoint_binding(step: int, manifest: dict, states: list[dict]) -> dict:
    return {
        "step": step,
        "run_manifest_sha256": manifest_sha256(manifest),
        "tokens_seen": step * manifest["lock"]["runtime"]["tokens_per_step"],
        "rng_states": states,
    }


def is_locked_checkpoint(checkpoint: dict) -> bool:
    manifest = checkpoint.get("run_manifest")
    saved_args = checkpoint.get("args")
    return (
        "protocol_resume" in checkpoint
        or (isinstance(manifest, dict) and "protocol_id" in manifest)
        or (isinstance(saved_args, dict) and saved_args.get("locked_protocol") is True)
    )


def _validate_adamw_resume_state(saved_optimizer: object, optimizer, step: int, expected_lr: float) -> None:
    if not isinstance(saved_optimizer, dict):
        raise ValueError("locked checkpoint is missing AdamW state")
    groups = saved_optimizer.get("param_groups", [])
    expected_groups = optimizer.state_dict()["param_groups"]
    if len(groups) != len(expected_groups) or any(
        saved != {**expected, "lr": expected_lr} for saved, expected in zip(groups, expected_groups)
    ):
        raise ValueError("locked checkpoint optimizer parameter groups differ from the protocol")
    parameters = {
        identifier: parameter
        for expected, active in zip(expected_groups, optimizer.param_groups)
        for identifier, parameter in zip(expected["params"], active["params"])
    }
    state = saved_optimizer.get("state", {})
    if not isinstance(state, dict) or state.keys() != parameters.keys():
        raise ValueError("locked checkpoint optimizer state is incomplete")
    for identifier, parameter in parameters.items():
        value = state[identifier]
        if not isinstance(value, dict) or value.keys() != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"locked checkpoint optimizer state for parameter {identifier} is incomplete")
        saved_step = value["step"]
        if not torch.is_tensor(saved_step) or saved_step.ndim != 0 or saved_step.item() != step:
            raise ValueError("locked checkpoint optimizer state does not match its recorded step")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = value[name]
            if not torch.is_tensor(moment) or moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                raise ValueError(f"locked checkpoint optimizer moment {name} differs for parameter {identifier}")


def validate_resume_checkpoint(
    checkpoint: dict, identity: dict, optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator, device: torch.device, learning_rate_for_saved_step,
) -> tuple[int, dict, list[dict]]:
    manifest = checkpoint.get("run_manifest")
    binding = checkpoint.get("protocol_resume")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2 or not isinstance(binding, dict):
        raise ValueError("locked resume requires a checkpoint with an immutable protocol manifest and RNG state")
    for key in IDENTITY_KEYS:
        if canonical_json(manifest.get(key)) != canonical_json(identity[key]):
            if key == "lock" and isinstance(manifest.get("lock"), dict):
                differences = [section for section, value in identity[key].items()
                               if canonical_json(manifest[key].get(section)) != canonical_json(value)]
                raise ValueError("locked checkpoint differs from the requested lock: " + ", ".join(differences))
            raise ValueError(f"locked checkpoint {key} differs from the requested run")
    if binding.get("run_manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("locked checkpoint's run manifest digest does not match")
    lock = identity["lock"]
    training = lock["training"]["arguments"]
    step = checkpoint.get("step")
    if (
        type(step) is not int or not 0 < step < training["steps"]
        or step % training["save_every"] or binding.get("step") != step
        or binding.get("tokens_seen") != step * lock["runtime"]["tokens_per_step"]
    ):
        raise ValueError("locked checkpoint step or sampled-token count is invalid for resume")
    if canonical_json(checkpoint.get("model_config")) != canonical_json(lock["model_config"]):
        raise ValueError("locked checkpoint model configuration differs from the requested model")
    saved_args = checkpoint.get("args")
    launch_args = manifest.get("launch_args")
    required_args = {**training, "locked_protocol": True, "seed": identity["seed"]}
    for args in (saved_args, launch_args):
        if (
            not isinstance(args, dict) or args.get("attention") != identity["arm"]
            or any(key not in args or args[key] != value for key, value in required_args.items())
        ):
            raise ValueError("locked checkpoint arguments do not match the frozen training settings")
    initialization = manifest.get("initialization")
    if (
        launch_args.get("resume_checkpoint") is not None or not isinstance(initialization, dict)
        or initialization.get("step") != 0 or initialization.get("initialization_kind") != "fresh"
        or initialization.get("filename") != INITIALIZATION_NAME
        or initialization.get("common_parameters_algorithm") != COMMON_INITIALIZATION
    ):
        raise ValueError("locked checkpoint is missing fresh step-zero initialization provenance")
    _validate_adamw_resume_state(checkpoint.get("optimizer_state"), optimizer, step, learning_rate_for_saved_step(step))
    rng_states = binding.get("rng_states")
    _validate_rng_states(rng_states, capture_training_rng(data_generator, device), lock["runtime"]["world_size"])
    return step, manifest, rng_states
