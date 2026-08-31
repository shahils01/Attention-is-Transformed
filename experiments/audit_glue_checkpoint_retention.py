from __future__ import annotations

import argparse
import json
from pathlib import Path


LARGE_STATE_NAMES = {
    "model.safetensors",
    "gt_mha_state_dict.pt",
    "optimizer.pt",
    "scheduler.pt",
}


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()

    selected: dict[tuple[str, str], Path] = {}
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text())
        for method, tasks in manifest.items():
            for task, record in tasks.items():
                key = (method, task)
                assert key not in selected
                selected[key] = Path(record["checkpoint_dir"]).resolve()

    assert len(selected) == 36, len(selected)
    missing = [str(path) for path in selected.values() if not path.is_dir()]
    assert not missing, missing

    upload_bytes = 0
    upload_files = 0
    selected_weight_bytes = 0
    for path in selected.values():
        for item in path.iterdir():
            if not item.is_file():
                continue
            size = item.stat().st_size
            upload_bytes += size
            upload_files += 1
            if item.name in LARGE_STATE_NAMES or item.name.startswith("rng_state"):
                selected_weight_bytes += size

    if args.selected_only:
        print(
            json.dumps(
                {
                    "selected_checkpoints": len(selected),
                    "selected_upload_files": upload_files,
                    "selected_upload_gib": upload_bytes / 2**30,
                    "selected_large_state_gib": selected_weight_bytes / 2**30,
                },
                indent=2,
            )
        )
        return

    reclaim_bytes = 0
    reclaim_files = 0
    reclaim_checkpoint_bytes = 0
    reclaim_root_state_bytes = 0
    for item in args.run_root.rglob("*"):
        if not item.is_file() or "glue" not in str(item).lower():
            continue
        if any(item.parent == path for path in selected.values()):
            continue
        relative = item.relative_to(args.run_root)
        in_checkpoint_dir = any(part.startswith("checkpoint-") for part in relative.parts)
        is_large_state = (
            item.name in LARGE_STATE_NAMES or item.name.startswith("rng_state")
        )
        if not in_checkpoint_dir and not is_large_state:
            continue
        size = item.stat().st_size
        reclaim_bytes += size
        reclaim_files += 1
        if in_checkpoint_dir:
            reclaim_checkpoint_bytes += size
        else:
            reclaim_root_state_bytes += size

    report = {
        "selected_checkpoints": len(selected),
        "selected_upload_files": upload_files,
        "selected_upload_gib": upload_bytes / 2**30,
        "selected_large_state_gib": selected_weight_bytes / 2**30,
        "reclaim_candidate_files": reclaim_files,
        "reclaim_candidate_gib": reclaim_bytes / 2**30,
        "reclaim_intermediate_checkpoint_gib": reclaim_checkpoint_bytes / 2**30,
        "reclaim_unselected_root_state_gib": reclaim_root_state_bytes / 2**30,
        "selected": {
            f"{method}/{task}": str(path)
            for (method, task), path in sorted(selected.items())
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
