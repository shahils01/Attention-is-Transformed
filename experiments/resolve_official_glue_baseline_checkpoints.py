from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("gqa", "collaborative")
METRICS = {
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
    "sst2": "eval_accuracy",
    "cola": "eval_matthews_correlation",
    "mrpc": "eval_combined_score",
    "stsb": "eval_combined_score",
    "qqp": "eval_combined_score",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
}
SWEEPS = {
    "mnli": "glue_stable_papersweep_v1",
    "qnli": "glue_stable_papersweep_v1",
    "sst2": "glue_stable_papersweep_v1",
    "qqp": "glue_stable_papersweep_v1",
    "cola": "glue_hivar_papersweep_v1",
    "mrpc": "glue_hivar_papersweep_v1",
    "rte": "glue_hivar_papersweep_v1",
}


def lr_tag(value: str) -> str:
    return value.replace("e-", "em")


def best_sweep_run(root: Path, task: str, method: str) -> dict[str, object]:
    prefix = SWEEPS[task]
    selection = json.loads(
        (root / f"{prefix}_{task}_selected_learning_rates.json").read_text()
    )
    learning_rate = str(selection[method]["selected_learning_rate"])
    candidates = []
    for seed in range(42, 47):
        stage = "grid" if seed <= 44 else "final"
        run_dir = root / (
            f"{prefix}_{task}_{method}_lr{lr_tag(learning_rate)}_seed{seed}_{stage}"
        )
        metrics = json.loads((run_dir / "eval_results.json").read_text())
        candidates.append(
            {
                "seed": seed,
                "learning_rate": learning_rate,
                "stage": stage,
                "validation_metric": METRICS[task],
                "validation_score": float(metrics[METRICS[task]]),
                "validation_loss": float(metrics["eval_loss"]),
                "checkpoint_dir": str(run_dir),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            float(row["validation_score"]),
            -float(row["validation_loss"]),
        ),
    )


def fixed_run(path: Path, task: str, seed: int, learning_rate: str) -> dict[str, object]:
    metrics = json.loads((path / "eval_results.json").read_text())
    return {
        "seed": seed,
        "learning_rate": learning_rate,
        "stage": "fixed",
        "validation_metric": METRICS[task],
        "validation_score": float(metrics[METRICS[task]]),
        "validation_loss": float(metrics["eval_loss"]),
        "checkpoint_dir": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stsb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, dict[str, dict[str, object]]] = {method: {} for method in METHODS}
    for method in METHODS:
        for task in SWEEPS:
            result[method][task] = best_sweep_run(args.run_root, task, method)

    stsb_paths = {
        "gqa": args.stsb_root
        / "Fine-tuned GLUE Checkpoints/stsb/gqa/stsb_papersweep_v2_gqa_lr3em5_seed44_grid",
        "collaborative": args.stsb_root
        / "Fine-tuned GLUE Checkpoints/stsb/collaborative/stsb_papersweep_v2_collaborative_lr5em5_seed45_final",
    }
    for method, path in stsb_paths.items():
        seed = 44 if method == "gqa" else 45
        learning_rate = "3e-5" if method == "gqa" else "5e-5"
        result[method]["stsb"] = fixed_run(path, "stsb", seed, learning_rate)

    for method in METHODS:
        path = args.run_root / f"glue_official_wnli_fixed_{method}_lr2em5_seed42"
        result[method]["wnli"] = fixed_run(path, "wnli", 42, "2e-5")

    for method in METHODS:
        for task, row in result[method].items():
            checkpoint_dir = Path(str(row["checkpoint_dir"]))
            required = (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "bert_glue_manifest.json",
            )
            missing = [name for name in required if not (checkpoint_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{method}/{task}: missing {missing} in {checkpoint_dir}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    for method in METHODS:
        for task in METRICS:
            row = result[method][task]
            print(
                f"{method:16s} {task:5s} seed={row['seed']} "
                f"lr={row['learning_rate']} "
                f"dev={100 * float(row['validation_score']):.4f} "
                f"path={row['checkpoint_dir']}"
            )


if __name__ == "__main__":
    main()
