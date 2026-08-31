from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


STATE_NAMES = {"model.safetensors", "gt_mha_state_dict.pt", "optimizer.pt", "scheduler.pt"}


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def tree_files(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--retention-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    retention_root = args.retention_root.resolve()
    selected: set[Path] = set()
    for manifest_path in args.manifest:
        data = json.loads(manifest_path.read_text())
        for tasks in data.values():
            for record in tasks.values():
                selected.add(Path(record["checkpoint_dir"]).resolve())
    assert len(selected) == 36, len(selected)
    assert all(path.is_dir() for path in selected)
    assert retention_root.is_dir()

    glue_roots = sorted(
        path for path in run_root.iterdir() if "glue" in path.name.lower()
    )
    assert glue_roots

    checkpoint_dirs: list[Path] = []
    cache_dirs: list[Path] = []
    for base in glue_roots:
        for dirpath, dirnames, _ in os.walk(base):
            current = Path(dirpath)
            if inside(current, retention_root):
                if current == retention_root:
                    dirnames[:] = [name for name in dirnames if name != ".cache"]
                else:
                    dirnames[:] = []
                continue
            lower = str(current).lower()
            for name in list(dirnames):
                child = current / name
                if name.startswith("checkpoint-"):
                    checkpoint_dirs.append(child)
                    dirnames.remove(name)
                elif name == ".cache" and (child / "huggingface").is_dir() and "glue" in lower:
                    cache_dirs.append(child / "huggingface")
                    dirnames.remove(name)

    staging_cache = retention_root / ".cache" / "huggingface"
    if staging_cache.is_dir():
        cache_dirs.append(staging_cache)

    standalone_files: list[Path] = []
    covered_dirs = {path.resolve() for path in checkpoint_dirs + cache_dirs}
    for base in glue_roots:
        for dirpath, dirnames, filenames in os.walk(base):
            current = Path(dirpath).resolve()
            if inside(current, retention_root) or current in covered_dirs:
                dirnames[:] = []
                continue
            dirnames[:] = [
                name for name in dirnames if (current / name).resolve() not in covered_dirs
            ]
            if current in selected:
                continue
            for name in filenames:
                if name in STATE_NAMES or name.startswith("rng_state"):
                    standalone_files.append(current / name)

    targets = [("intermediate_checkpoint_directory", path) for path in checkpoint_dirs]
    targets += [("temporary_hf_cache", path) for path in cache_dirs]
    targets += [("unselected_or_redundant_state_file", path) for path in standalone_files]
    targets.sort(key=lambda pair: str(pair[1]))

    inode_sizes: dict[tuple[int, int], int] = {}
    inode_nlinks: dict[tuple[int, int], int] = {}
    target_link_counts: dict[tuple[int, int], int] = defaultdict(int)
    logical_bytes = 0
    target_records = []
    for category, target in targets:
        target_bytes = 0
        target_files = 0
        for item in tree_files(target):
            stat = item.stat()
            target_bytes += stat.st_size
            target_files += 1
            key = (stat.st_dev, stat.st_ino)
            inode_sizes[key] = stat.st_size
            inode_nlinks[key] = stat.st_nlink
            target_link_counts[key] += 1
        logical_bytes += target_bytes
        target_records.append({
            "category": category,
            "path": str(target),
            "files": target_files,
            "logical_bytes": target_bytes,
        })

    # A targeted inode is physically reclaimable only if every hard link is targeted.
    physical_bytes = 0
    retained_by_external_hardlink_bytes = 0
    for key, size in inode_sizes.items():
        if target_link_counts[key] >= inode_nlinks[key]:
            physical_bytes += size
        else:
            retained_by_external_hardlink_bytes += size

    output = {
        "schema_version": 1,
        "action": "delete_only_after_explicit_user_approval",
        "run_root": str(run_root),
        "scanned_glue_roots": list(map(str, glue_roots)),
        "retention_root": str(retention_root),
        "protected_selected_source_directories": sorted(map(str, selected)),
        "summary": {
            "selected_checkpoints_protected": len(selected),
            "targets": len(target_records),
            "checkpoint_directories": len(checkpoint_dirs),
            "temporary_hf_caches": len(cache_dirs),
            "standalone_state_files": len(standalone_files),
            "logical_bytes": logical_bytes,
            "logical_gib": logical_bytes / 2**30,
            "estimated_physically_reclaimable_bytes": physical_bytes,
            "estimated_physically_reclaimable_gib": physical_bytes / 2**30,
            "retained_by_external_hardlink_bytes": retained_by_external_hardlink_bytes,
        },
        "targets": target_records,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
