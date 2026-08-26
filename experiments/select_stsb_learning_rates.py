from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one STS-B learning rate per attention type by mean dev score."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", default="stsb_papersweep_v2")
    parser.add_argument(
        "--methods", nargs="+", default=("mha", "gqa", "collaborative", "gt_mha_residual")
    )
    parser.add_argument("--learning-rates", nargs="+", default=("2e-5", "3e-5", "5e-5"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def lr_tag(learning_rate: str) -> str:
    return learning_rate.replace("e-", "em")


def main() -> None:
    args = parse_args()
    summary: dict[str, dict[str, object]] = {}
    for method in args.methods:
        candidates: dict[str, dict[str, object]] = {}
        for learning_rate in args.learning_rates:
            scores = []
            for seed in args.seeds:
                run_dir = args.root / (
                    f"{args.prefix}_{method}_lr{lr_tag(learning_rate)}_seed{seed}_grid"
                )
                metrics_path = run_dir / "eval_results.json"
                if not metrics_path.is_file():
                    raise FileNotFoundError(f"missing completed metrics: {metrics_path}")
                metrics = json.loads(metrics_path.read_text())
                scores.append(float(metrics["eval_combined_score"]))
            candidates[learning_rate] = {
                "scores": scores,
                "mean": float(np.mean(scores)),
                "sample_std": float(np.std(scores, ddof=1)),
            }
        selected = max(
            args.learning_rates,
            key=lambda learning_rate: (
                float(candidates[learning_rate]["mean"]),
                -float(learning_rate),
            ),
        )
        summary[method] = {
            "selected_learning_rate": selected,
            "selection_seeds": args.seeds,
            "candidates": candidates,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    with args.output_tsv.open("w") as handle:
        for method in args.methods:
            learning_rate = str(summary[method]["selected_learning_rate"])
            mean = float(summary[method]["candidates"][learning_rate]["mean"])
            handle.write(f"{method}\t{learning_rate}\t{mean:.10f}\n")

    for method in args.methods:
        selected = str(summary[method]["selected_learning_rate"])
        details = summary[method]["candidates"][selected]
        print(
            f"{method}: selected {selected}; "
            f"mean={100 * float(details['mean']):.2f}, "
            f"sd={100 * float(details['sample_std']):.2f}"
        )


if __name__ == "__main__":
    main()
