from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.checkpointing import load_full_checkpoint
from lgma.lu_export import export_lu_model
from lgma.transformer import TinyTransformerLM


def export_checkpoint(source_path: Path, output_path: Path) -> None:
    """Write a standard TinyStories inference checkpoint to a new path."""
    if os.path.lexists(output_path):
        raise FileExistsError(f"output already exists: {output_path}")

    checkpoint = load_full_checkpoint(source_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a TinyStories trainer checkpoint")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain model_config and model_state")
    embedding = state.get("token_embedding.weight")
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ValueError("checkpoint is missing a valid token_embedding.weight")
    if not embedding.is_floating_point():
        raise ValueError("checkpoint parameters must be floating point")
    vocab_size = embedding.shape[0]
    saved_args = checkpoint.get("args")
    data_args = {}
    if isinstance(saved_args, dict):
        for key in ("data_path", "val_data_path", "packed_data_dir"):
            if saved_args.get(key) is not None:
                data_args[key] = str(saved_args[key])
    deployment = {
        "format": "fixed_base_lu_to_gqa",
        "source_checkpoint": str(source_path.resolve()),
        "source_model_config": config,
    }
    if "run_manifest" in checkpoint:
        deployment["source_run_manifest"] = checkpoint["run_manifest"]
    payload = {
        "step": int(checkpoint.get("step", 0)),
        "vocab_size": vocab_size,
        "args": data_args,
        "deployment": deployment,
        "inference_only": True,
    }
    for key in ("tokenizer", "tokenizer_config", "tokenizer_state"):
        if key in checkpoint:
            payload[key] = checkpoint[key]
    del checkpoint

    # Training checkpoints store a single parameter dtype even when the run
    # used autocast. Loading in that dtype preserves the serialized weights.
    if any(
        isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        and tensor.dtype != embedding.dtype
        for tensor in state.values()
    ):
        raise ValueError("checkpoint parameters must have a common dtype")
    with torch.random.fork_rng(devices=[]):
        source = TinyTransformerLM(vocab_size=vocab_size, **config).to(dtype=embedding.dtype)
    source.load_state_dict(state)
    del state, embedding
    model = export_lu_model(source)
    del source
    attention = model.first_attention
    payload["model_config"] = {
        "attention_type": "gqa",
        "d_model": model.d_model,
        "num_layers": len(model.blocks),
        "num_heads": attention.num_heads,
        "head_dim": attention.head_dim,
        "num_kv_heads": attention.num_kv_heads,
        "context_length": model.context_length,
        "dropout": model.drop.p,
        "causal": model.causal,
    }
    payload["model_state"] = model.state_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish after serialization, with a no-clobber operation so concurrent
    # exports cannot overwrite either an existing export or a source checkpoint.
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    ) as temporary:
        torch.save(payload, temporary)
        temporary.flush()
        os.link(temporary.name, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trusted fixed-base LU TinyStories checkpoint as standard GQA."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="source trainer checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new inference checkpoint path")
    args = parser.parse_args()
    try:
        export_checkpoint(args.checkpoint, args.output)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(args.output)


if __name__ == "__main__":
    main()
