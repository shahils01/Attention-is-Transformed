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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.package_root / "selected_checkpoints.json").read_text())
    assert set(manifest) == set(METHODS)

    for method in METHODS:
        assert set(manifest[method]) == set(METRICS)
        for task, prefix in SWEEPS.items():
            selection = json.loads(
                (args.run_root / f"{prefix}_{task}_selected_learning_rates.json").read_text()
            )
            learning_rate = str(selection[method]["selected_learning_rate"])
            selected = manifest[method][task]
            assert selected["learning_rate"] == learning_rate

            candidates = []
            for seed in range(42, 47):
                stage = "grid" if seed <= 44 else "final"
                run_dir = args.run_root / (
                    f"{prefix}_{task}_{method}_lr{lr_tag(learning_rate)}_"
                    f"seed{seed}_{stage}"
                )
                metrics = json.loads((run_dir / "eval_results.json").read_text())
                candidates.append(
                    (
                        float(metrics[METRICS[task]]),
                        -float(metrics["eval_loss"]),
                        seed,
                        str(run_dir),
                    )
                )
            best = max(candidates)
            assert selected["seed"] == best[2]
            assert selected["checkpoint_dir"] == best[3]
            assert selected["validation_score"] == best[0]
            assert selected["validation_loss"] == -best[1]

        assert manifest[method]["wnli"]["learning_rate"] == "2e-5"
        assert manifest[method]["wnli"]["seed"] == 42
        assert manifest[method]["stsb"]["learning_rate"] == (
            "3e-5" if method == "gqa" else "5e-5"
        )
        assert manifest[method]["stsb"]["seed"] == (44 if method == "gqa" else 45)

        provenance = json.loads(
            (args.package_root / f"GT-MHA_GLUE_official_{method}_provenance.json").read_text()
        )
        assert provenance["method"] == method
        assert provenance["official_test_labels_accessed"] is False
        assert len(provenance["files"]) == 11
        for record in provenance["files"]:
            selected = manifest[method][record["checkpoint_task"]]
            assert record["checkpoint_seed"] == selected["seed"]
            assert record["checkpoint_learning_rate"] == selected["learning_rate"]

        print(f"{method}: validation-only selection and provenance verified")


if __name__ == "__main__":
    main()
