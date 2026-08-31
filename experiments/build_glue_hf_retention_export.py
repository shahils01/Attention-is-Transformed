from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


REQUIRED_CHECKPOINT_FILES = {
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "bert_glue_manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selections(paths: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for method, tasks in payload.items():
            for task, record in tasks.items():
                key = (method, task)
                if key in selected:
                    raise ValueError(f"duplicate selection: {method}/{task}")
                selected[key] = record
    if len(selected) != 36:
        raise ValueError(f"expected 36 selected checkpoints, found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, action="append", required=True)
    parser.add_argument("--provenance", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    selected = load_selections(args.selection_manifest)
    checkpoint_root = (
        args.output_root / "Fine-tuned GLUE Checkpoints" / "official_glue_submission"
    )
    selection_root = checkpoint_root / "selection_metadata"
    selection_root.mkdir(parents=True)

    exported = []
    for (method, task), record in sorted(selected.items()):
        source = Path(str(record["checkpoint_dir"]))
        if not source.is_dir():
            raise FileNotFoundError(source)
        source_files = {path.name for path in source.iterdir() if path.is_file()}
        missing = REQUIRED_CHECKPOINT_FILES - source_files
        if missing:
            raise FileNotFoundError(f"{method}/{task} missing {sorted(missing)}")

        destination = checkpoint_root / method / task
        destination.mkdir(parents=True)
        for source_file in sorted(source.iterdir()):
            if not source_file.is_file():
                continue
            os.link(source_file, destination / source_file.name)

        public_record = {
            key: value for key, value in record.items() if key != "checkpoint_dir"
        }
        public_record.update(
            {
                "architecture": method,
                "task": task,
                "selection_policy": (
                    "best validation seed at the learning rate selected by the "
                    "seeds 42-44 validation mean"
                    if task != "wnli"
                    else "fixed common WNLI setting; no architecture-specific sweep"
                ),
            }
        )
        (destination / "selection_record.json").write_text(
            json.dumps(public_record, indent=2, sort_keys=True) + "\n"
        )
        exported.append(public_record)

    sanitized_selections: dict[str, dict[str, dict[str, object]]] = {}
    for record in exported:
        method = str(record["architecture"])
        task = str(record["task"])
        sanitized_selections.setdefault(method, {})[task] = record
    (selection_root / "selected_checkpoints.json").write_text(
        json.dumps(sanitized_selections, indent=2, sort_keys=True) + "\n"
    )

    for provenance_path in args.provenance:
        payload = json.loads(provenance_path.read_text())
        for file_record in payload.get("files", []):
            file_record.pop("checkpoint_dir", None)
        payload.pop("zip", None)
        (selection_root / provenance_path.name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    learning_rate_root = selection_root / "learning_rate_selections"
    learning_rate_root.mkdir()
    for path in sorted(args.run_root.glob("glue_*_papersweep_v1_*_selected_learning_rates.*")):
        if path.is_file():
            shutil.copy2(path, learning_rate_root / path.name)

    sweep_records = []
    patterns = (
        "glue_stable_papersweep_v1_*",
        "glue_hivar_papersweep_v1_*",
        "glue_official_wnli_fixed_*",
    )
    for pattern in patterns:
        for run_dir in sorted(args.run_root.glob(pattern)):
            if not run_dir.is_dir():
                continue
            record: dict[str, object] = {"run_name": run_dir.name}
            for filename in (
                "eval_results.json",
                "train_results.json",
                "all_results.json",
                "bert_glue_manifest.json",
            ):
                path = run_dir / filename
                if path.is_file():
                    record[filename.removesuffix(".json")] = json.loads(path.read_text())
            sweep_records.append(record)
    (selection_root / "paper_sweep_run_records.json").write_text(
        json.dumps(sweep_records, indent=2, sort_keys=True) + "\n"
    )

    (checkpoint_root / "README.md").write_text(
        "# Official GLUE submission checkpoints\n\n"
        "These are the 36 lean task checkpoints used to generate the official "
        "GLUE test submissions for MHA, GQA, Collaborative MHA, and GT-MHA "
        "residual. Checkpoints were selected using validation results only; "
        "official test labels were never accessed. Each directory contains the "
        "inference-ready model, tokenizer, configuration, training arguments, "
        "evaluation metrics, and its validation-selection record.\n"
    )

    files = []
    for path in sorted(args.output_root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(args.output_root)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "checkpoint_count": 36,
        "selection_basis": "validation only",
        "official_test_labels_accessed": False,
        "files": files,
        "total_bytes": sum(record["size"] for record in files),
    }
    (checkpoint_root / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"files": len(files), "bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
