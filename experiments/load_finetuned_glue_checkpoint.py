from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lgma.bert import replace_bert_self_attention


def load_finetuned_glue_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trust_remote_code: bool = False,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    """Reconstruct and strictly load an LGMA BERT GLUE classifier checkpoint."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification

    path = Path(checkpoint_dir)
    manifest_path = path / "bert_glue_manifest.json"
    state_path = path / "model.safetensors"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing fine-tuning manifest: {manifest_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"missing model state: {state_path}")

    manifest = json.loads(manifest_path.read_text())
    saved_args = manifest.get("arguments", {})
    attention_type = manifest["attention_type"]
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    # The checkpoint will replace every parameter, so avoid spending minutes
    # randomly initializing a full BERT on CPU before strict state assignment.
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=trust_remote_code
        )
        audit = replace_bert_self_attention(
            model,
            attention_type=attention_type,
            num_kv_heads=int(manifest.get("num_kv_heads", 4)),
            num_base_heads=int(manifest.get("num_base_heads", 4)),
            num_generators=int(manifest.get("num_generators", 8)),
            qk_base_dim=manifest.get("qk_base_dim"),
            value_head_dim=manifest.get("value_head_dim"),
            generator_mixing=manifest.get("generator_mixing", "softmax"),
            use_sdpa=bool(saved_args.get("use_sdpa", False)),
            fuse_base_qkv=bool(saved_args.get("fuse_base_qkv", False)),
            sdpa_gqa_mode=saved_args.get("sdpa_gqa_mode", "auto"),
            initialize_from_mha=False,
            enforce_paper_gt_mha=bool(saved_args.get("enforce_paper_gt_mha", True)),
        )
    state = load_file(str(state_path), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()
    model, audit = load_finetuned_glue_checkpoint(args.checkpoint_dir)
    print(
        json.dumps(
            {
                "model_class": type(model).__name__,
                "attention_layers_replaced": len(audit),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
