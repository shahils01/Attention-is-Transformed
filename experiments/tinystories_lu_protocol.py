"""Frozen launch and preflight checks for the TinyStories LU vs GQA4 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "tinystories_lu_gqa4_v1"
PROTOCOL_PATH = ROOT / "experiments" / "protocols" / f"{PROTOCOL_ID}.json"
PROTOCOL_SHA256 = "543154cf73832ca6bd223052889de0a5a6b9d7dfef467a7f97aa651670b2d78b"
SOURCE_RECEIPT_PATH = PROTOCOL_PATH.with_name(f"{PROTOCOL_ID}_sources.json")
SOURCE_PATHS = (
    "experiments/evaluate_lu_protocol.py",
    "experiments/tinystories_lu_protocol.py",
    "experiments/tinystories_protocol_training.py",
    "experiments/tinystories_runtime.py",
    "experiments/train_tinystories.py",
    f"experiments/protocols/{PROTOCOL_ID}.json",
    "src/lgma/__init__.py",
    "src/lgma/accounting.py",
    "src/lgma/attention.py",
    "src/lgma/baselines.py",
    "src/lgma/checkpointing.py",
    "src/lgma/diagnostics.py",
    "src/lgma/kv_cache.py",
    "src/lgma/lu_kernel.py",
    "src/lgma/packed_data.py",
    "src/lgma/synthetic.py",
    "src/lgma/tracking.py",
    "src/lgma/transformer.py",
    "src/lgma/vision.py",
)
TRACKING_OPTIONS = (
    "wandb_project", "wandb_entity", "wandb_run_name", "wandb_group",
    "wandb_tags", "wandb_mode", "wandb_dir",
)
# Effective model configuration keys mapped to argparse destinations. Shape
# values that are unused for an arm remain at their ordinary parser defaults.
MODEL_OPTIONS = {
    "attention_type": "attention",
    "d_model": "d_model",
    "num_layers": "num_layers",
    "num_heads": "num_heads",
    "head_dim": "head_dim",
    "context_length": "context_length",
    "dropout": "dropout",
    "num_generators": "num_generators",
    "logit_scale_mode": "logit_scale_mode",
    "learn_head_temperature": "learn_head_temperature",
    "value_transform": "value_transform",
    "fuse_base_qkv": "fuse_base_qkv",
    "fold_value_transform_into_output": "fold_value_transform_into_output",
    "num_base_heads": "num_base_heads",
    "num_kv_heads": "num_kv_heads",
    "base_dim": "base_dim",
    "value_dim": "value_dim",
    "metric_mode": "metric_mode",
    "lu_init_scale": "lu_init_scale",
    "fuse_lu_queries": "fuse_lu_queries",
    "sdpa_gqa_mode": "sdpa_gqa_mode",
}
BOOLEAN_MODEL_OPTIONS = {
    "learn_head_temperature", "fuse_base_qkv", "fold_value_transform_into_output", "fuse_lu_queries",
}
UNUSED_MODEL_OPTIONS = (
    "generator_type", "generator_mixing", "theta_init_scale", "generator_init_scale",
    "stabilize_generators", "normalize_generators", "head_generator_symmetric_cap",
    "metric_beta", "metric_clip_min", "metric_clip_max", "value_beta", "theta_init",
)


def fail(reason: str) -> None:
    raise SystemExit(f"{PROTOCOL_ID}: {reason}")


def file_sha256(path: str | Path) -> str:
    try:
        with Path(path).open("rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    return digest.hexdigest()


def load_protocol() -> dict:
    try:
        raw = PROTOCOL_PATH.read_bytes()
    except OSError as error:
        fail(f"cannot read frozen protocol: {error}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROTOCOL_SHA256:
        fail(f"frozen protocol SHA-256 is {digest}, expected {PROTOCOL_SHA256}")
    protocol = json.loads(raw)
    return {**protocol, "protocol_sha256": digest}


def require_values(scope: str, expected: dict, actual: dict, *, exact: bool = False) -> None:
    mismatches = [
        f"{key}: expected {value!r}, got {actual[key]!r}" if key in actual else f"{key}: missing"
        for key, value in expected.items()
        if key not in actual or value != actual[key]
    ]
    if exact:
        mismatches.extend(f"{key}: unexpected" for key in sorted(actual.keys() - expected.keys()))
    if mismatches:
        fail(f"{scope} differs from the frozen protocol ({'; '.join(mismatches)})")


def source_fingerprints() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "git_revision": revision,
        "files_sha256": {relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS},
    }


def validate_source_fingerprints(fingerprints: dict) -> dict:
    try:
        receipt = json.loads(SOURCE_RECEIPT_PATH.read_bytes())
    except (OSError, ValueError) as error:
        fail(f"cannot read frozen source receipt: {error}")
    if not isinstance(receipt, dict) or set(receipt) != set(SOURCE_PATHS):
        fail("frozen source receipt does not cover the required source paths")
    actual = fingerprints.get("files_sha256")
    if not isinstance(actual, dict):
        fail("source provenance is missing file hashes")
    require_values("source files", receipt, actual, exact=True)
    return fingerprints


def validate_tokenizer(protocol: dict, tokenizer) -> dict:
    expected = protocol["data"]["tokenizer"]
    observed = list(tokenizer.itos)
    if observed != expected["vocabulary"] or tokenizer.stoi != {
        char: idx for idx, char in enumerate(observed)
    }:
        fail("tokenizer must use the exact frozen ordered character mapping")
    digest = hashlib.sha256("".join(observed).encode("utf-8")).hexdigest()
    proof = {
        "vocab_size": tokenizer.vocab_size,
        "vocab_sha256": digest,
        "algorithm": expected["algorithm"],
        "json_vocab_sha256": hashlib.sha256(
            json.dumps(observed, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    require_values("tokenizer", {key: expected[key] for key in proof}, proof)
    return proof


def validate_training_data(protocol: dict, train_path, val_path, tokenizer) -> dict:
    if not train_path or not val_path:
        fail("training and validation data paths are both required")
    paths = {"train": Path(train_path).resolve(), "validation": Path(val_path).resolve()}
    proof = {}
    for split, path in paths.items():
        expected = protocol["data"][split]
        try:
            size = path.stat().st_size
        except OSError as error:
            fail(f"cannot inspect {split} text {path}: {error}")
        require_values(f"{split} text {path}", {"size_bytes": expected["size_bytes"]}, {
            "size_bytes": size,
        })
        actual = {"size_bytes": size, "sha256": file_sha256(path)}
        require_values(f"{split} text {path}", {"sha256": expected["sha256"]}, actual)
        proof[split] = {"path": str(path), **actual}
    proof["tokenizer"] = validate_tokenizer(protocol, tokenizer)
    return proof


def model_config_for(protocol: dict, arm: str) -> dict:
    if arm not in protocol["arms"]:
        fail(f"unknown arm {arm!r}")
    return {**protocol["architecture"]["model_config"], **protocol["arms"][arm]["model_config"]}


def validate_model_config(protocol: dict, arm: str, config: dict) -> None:
    require_values("merged model configuration", model_config_for(protocol, arm), config, exact=True)


def validate_training_args(protocol: dict, args: argparse.Namespace, config: dict) -> None:
    if type(args.seed) is not int or args.seed not in protocol["seeds"]:
        fail(f"seed must be one of {protocol['seeds']}, got {args.seed!r}")
    if not args.data_path or not args.val_data_path or args.output_dir is None:
        fail("training, validation and output paths are required")
    require_values("training arguments", protocol["training"]["arguments"], vars(args))
    if args.num_generators not in (None, 0):
        fail("number of training generators must be zero")
    require_values("generator and score override arguments", {
        key: None for key in UNUSED_MODEL_OPTIONS
    }, vars(args))
    if args.attention == "gt_mha_lu" and args.fold_value_transform_into_output:
        fail("the LU arm has no value transform to fold into its output projection")
    if args.attention == "gqa" and (args.lu_init_scale is not None or args.fuse_lu_queries is not None):
        fail("LU constructor arguments apply to the LU arm only")
    validate_model_config(protocol, args.attention, config)


def validate_cuda_device(device, *, gpu_name: str, device_count: int) -> dict:
    import torch

    protocol_runtime = load_protocol()["runtime"]
    if device.type != "cuda" or not torch.cuda.is_available():
        fail("a CUDA GH200 device is required")
    if torch.cuda.device_count() != device_count:
        fail(f"expected {device_count} visible CUDA devices, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(device)
    if properties.name != gpu_name:
        fail(f"expected GPU {gpu_name!r}, got {properties.name!r}")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        fail("native BF16 support is required")
    from lgma.lu_kernel import triton

    if triton is None or os.environ.get("TRITON_INTERPRET") == "1":
        fail("native compiled Triton kernels are required")
    actual = {
        "compute_capability": [properties.major, properties.minor],
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "triton": str(triton.__version__),
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1"
        ),
    }
    require_values("CUDA runtime", {key: protocol_runtime[key] for key in actual}, actual)
    return {
        **actual,
        "gpu_name": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "device_count": device_count,
    }


def validate_training_runtime(
    protocol: dict, device, *, rank: int, local_rank: int, world_size: int,
) -> dict:
    import torch.distributed as dist

    expected = protocol["runtime"]
    if (
        world_size != expected["ranks"]
        or rank != local_rank
        or local_rank not in range(world_size)
        or device.index != local_rank
        or os.environ.get("LOCAL_WORLD_SIZE") != str(expected["devices_per_node"])
        or os.environ.get("GROUP_RANK") != "0"
        or not dist.is_initialized()
        or dist.get_backend() != expected["distributed_backend"]
    ):
        fail("training requires four CUDA ranks on one node via torchrun and NCCL")
    cuda = validate_cuda_device(
        device, gpu_name=expected["gpu_name"], device_count=expected["devices_per_node"],
    )
    return {
        **cuda, "ranks": world_size, "nodes": 1, "distributed_backend": dist.get_backend(),
        "autocast": expected["autocast"],
    }


def validate_model_structure(protocol: dict, arm: str, model) -> dict:
    import torch

    from lgma.attention import LieGeneratedMetricAttention
    from lgma.baselines import GroupedQueryAttention

    config = model_config_for(protocol, arm)
    width = config["d_model"]
    vocab_size = protocol["data"]["tokenizer"]["vocab_size"]
    structure = {
        "d_model": model.d_model,
        "vocab_size": model.vocab_size,
        "num_layers": len(model.blocks),
        "context_length": model.context_length,
        "causal": model.causal,
        "embedding_dropout": model.drop.p,
        "token_embedding_shape": list(model.token_embedding.weight.shape),
        "position_embedding_shape": list(model.position_embedding.weight.shape),
        "lm_head_shape": list(model.lm_head.weight.shape),
    }
    require_values("model backbone", {
        "d_model": width, "vocab_size": vocab_size, "num_layers": config["num_layers"],
        "context_length": config["context_length"], "causal": True,
        "embedding_dropout": config["dropout"],
        "token_embedding_shape": [vocab_size, width],
        "position_embedding_shape": [config["context_length"], width],
        "lm_head_shape": [vocab_size, width],
    }, structure)
    head_dim = config["head_dim"]
    query_heads = config["num_heads"]
    is_lu = arm == "gt_mha_lu"
    kv_heads = config["num_base_heads"] if is_lu else config["num_kv_heads"]
    layer_values = {
        "num_heads": query_heads, "head_dim": head_dim, "d_model": width,
        "dropout": config["dropout"], "causal": True,
    }
    if is_lu:
        layer_values.update({
            name: config[name] for name in (
                "num_base_heads", "base_dim", "value_dim", "num_generators", "metric_mode",
                "logit_scale_mode", "learn_head_temperature", "value_transform", "fuse_base_qkv",
                "lu_init_scale", "fuse_lu_queries", "sdpa_gqa_mode",
            )
        })
        layer_values["num_value_base_heads"] = kv_heads
    else:
        layer_values["num_kv_heads"] = kv_heads

    layer_proofs = []
    for index, block in enumerate(model.blocks):
        attn = block.attn
        expected_type = LieGeneratedMetricAttention if is_lu else GroupedQueryAttention
        if type(attn) is not expected_type:
            fail(f"layer {index} must instantiate {expected_type.__name__}")
        actual = {key: getattr(attn, key, None) for key in layer_values}
        require_values(f"layer {index} attention", layer_values, actual)
        require_values(f"layer {index} feedforward", {
            "hidden_width": width * protocol["architecture"]["feedforward_expansion"],
            "dropout": config["dropout"],
        }, {"hidden_width": block.mlp[0].out_features, "dropout": block.mlp[-1].p})
        projections = {
            "q_proj": [(kv_heads if is_lu else query_heads) * head_dim, width],
            "k_proj": [kv_heads * head_dim, width],
            "v_proj": [kv_heads * head_dim, width],
            "out_proj": [width, query_heads * head_dim],
        }
        for name, shape in projections.items():
            projection = getattr(attn, name)
            if list(projection.weight.shape) != shape or projection.bias is not None:
                fail(f"layer {index} {name} must have shape {shape} with bias disabled")
        if hasattr(attn, "head_logit_scale"):
            fail(f"layer {index} has an active learned temperature")
        extra_shapes = {}
        if is_lu:
            factors = {
                "lu_lower": head_dim * (head_dim - 1) // 2,
                "lu_upper": head_dim * (head_dim - 1) // 2,
                "lu_log_diag": head_dim,
            }
            expected_factors = {
                name: [kv_heads, query_heads // kv_heads - 1, entries]
                for name, entries in factors.items()
            }
            extra_shapes = {
                name: list(parameter.shape)
                for name, parameter in attn.named_parameters(recurse=False)
            }
            require_values(f"layer {index} LU parameter shapes", expected_factors, extra_shapes, exact=True)
        layer_proofs.append({
            **actual, "projection_weight_shapes": projections, "extra_parameter_shapes": extra_shapes,
        })
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        fail("all master parameters must be FP32")
    return {**structure, "attention_layers": layer_proofs, "master_parameter_dtype": "float32"}


def validate_training_model(protocol: dict, arm: str, model, optimizer) -> dict:
    import torch

    structure = validate_model_structure(protocol, arm, model)
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        fail("optimizer must be AdamW with one group for all parameters")
    group = optimizer.param_groups[0]
    parameters = list(model.parameters())
    if (
        not all(parameter.requires_grad for parameter in parameters)
        or len(group["params"]) != len(parameters)
        or {id(parameter) for parameter in parameters} != {id(parameter) for parameter in group["params"]}
    ):
        fail("the AdamW group must contain all trainable model parameters exactly once")
    expected_lr = protocol["training"]["arguments"]["lr"]
    require_values("initial optimizer learning rate", {"lr": expected_lr}, group)
    frozen = protocol["training"]["optimizer"]
    settings = {key: group.get(key) for key in frozen.keys() - {"type", "all_parameters"}}
    settings["betas"] = list(settings["betas"])
    settings.update(type="AdamW", all_parameters=True, weight_decay=group["weight_decay"])
    require_values("optimizer", {
        **frozen, "weight_decay": protocol["training"]["arguments"]["weight_decay"],
    }, settings)
    return {**structure, "optimizer": settings}


def build_training_command(protocol: dict, options: argparse.Namespace) -> list[str]:
    config = model_config_for(protocol, options.arm)
    runtime = protocol["runtime"]
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes", str(runtime["nodes"]), "--nproc_per_node", str(runtime["devices_per_node"]),
        str(ROOT / "experiments" / "train_tinystories.py"), "--locked_protocol",
        "--seed", str(options.seed), "--data_path", str(options.train_path.resolve()),
        "--val_data_path", str(options.val_path.resolve()), "--output_dir", str(options.output_dir.resolve()),
    ]
    for field, option in MODEL_OPTIONS.items():
        value = config.get(field)
        if field in BOOLEAN_MODEL_OPTIONS and value is not None:
            flag = f"--{option}" if value else f"--no-{option}"
            command.append(flag)
        elif value is not None:
            command.extend([f"--{option}", str(value)])
    for field, value in protocol["training"]["arguments"].items():
        if value is not None and value is not False:
            command.extend([f"--{field}", str(value)])
    if options.resume_checkpoint is not None:
        command.extend(["--resume_checkpoint", str(options.resume_checkpoint.resolve())])
    for field in TRACKING_OPTIONS:
        value = getattr(options, field)
        if value is not None:
            command.extend([f"--{field}", str(value)])
    return command


def main(argv: list[str] | None = None) -> None:
    protocol = load_protocol()
    parser = argparse.ArgumentParser(
        description="Inspect or launch the frozen TinyStories LU vs GQA4 training protocol.",
        allow_abbrev=False,
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("show", help="Print the frozen protocol and its digest.", allow_abbrev=False)
    for action, description in (
        ("print-command", "Print the frozen torchrun command without launching it."),
        ("launch", "Verify source and launch on an allocated four-GH200 node."),
    ):
        run_parser = actions.add_parser(action, help=description, allow_abbrev=False)
        run_parser.add_argument("--arm", choices=protocol["arms"], required=True)
        run_parser.add_argument("--seed", type=int, choices=protocol["seeds"], required=True)
        run_parser.add_argument("--train-path", type=Path, required=True)
        run_parser.add_argument("--val-path", type=Path, required=True)
        run_parser.add_argument("--output-dir", type=Path, required=True)
        run_parser.add_argument("--resume-checkpoint", type=Path)
        run_parser.add_argument("--print", action="store_true", help="Display the command only.")
        for option in TRACKING_OPTIONS:
            kwargs = {"choices": ["online", "offline", "disabled"]} if option == "wandb_mode" else {}
            run_parser.add_argument("--" + option.replace("_", "-"), **kwargs)
    options = parser.parse_args(argv)
    if options.action == "show":
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    command = build_training_command(protocol, options)
    if options.action == "print-command" or options.print:
        print(shlex.join(command))
        return
    validate_source_fingerprints(source_fingerprints())
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
