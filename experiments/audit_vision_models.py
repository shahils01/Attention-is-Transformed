from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgma.vision import deit_base_patch16_224, vision_parameter_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-base-heads", type=int, default=4)
    args = parser.parse_args()
    variants = (
        "mha",
        "reduced_mha",
        "gqa",
        "collaborative",
        "shared_identity",
        "gt_mha_residual",
    )
    rows = []
    baseline = None
    for attention_type in variants:
        model = deit_base_patch16_224(
            attention_type=attention_type,
            num_base_heads=args.num_base_heads,
        )
        counts = vision_parameter_counts(model)
        if baseline is None:
            baseline = counts
        rows.append({
            "attention_type": attention_type,
            **counts,
            "total_reduction_vs_mha": 1.0 - counts["total_parameters"] / baseline["total_parameters"],
            "attention_reduction_vs_mha": 1.0 - counts["attention_parameters"] / baseline["attention_parameters"],
        })
        del model
    payload = {"num_base_heads": args.num_base_heads, "models": rows}
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
