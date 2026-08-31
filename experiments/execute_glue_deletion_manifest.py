from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


APPROVAL = "I approve the deletion scope"


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--execution-log", type=Path, required=True)
    args = parser.parse_args()

    if args.approval != APPROVAL:
        raise SystemExit("Exact approval phrase was not supplied")

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("action") != "delete_only_after_explicit_user_approval":
        raise SystemExit("Unexpected manifest action")

    run_root = Path(manifest["run_root"]).resolve()
    retention_root = Path(manifest["retention_root"]).resolve()
    selected = [Path(path).resolve() for path in manifest["protected_selected_source_directories"]]
    targets = manifest["targets"]
    if len(targets) != manifest["summary"]["targets"]:
        raise SystemExit("Target count no longer matches manifest summary")

    resolved: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for record in targets:
        category = record["category"]
        path = Path(record["path"])
        absolute = path.absolute()
        if not inside(absolute, run_root):
            raise SystemExit(f"Target escapes run root: {path}")
        if absolute in seen:
            raise SystemExit(f"Duplicate target: {path}")
        seen.add(absolute)

        # The only allowed target inside the protected export is its resumable upload cache.
        if inside(absolute, retention_root):
            allowed_cache = retention_root / ".cache" / "huggingface"
            if category != "temporary_hf_cache" or absolute != allowed_cache:
                raise SystemExit(f"Target overlaps retention export: {path}")

        for protected in selected:
            if absolute == protected or inside(protected, absolute):
                raise SystemExit(f"Target contains a protected selected checkpoint: {path}")
        resolved.append((category, absolute))

    started = time.time()
    deleted = []
    missing = []
    errors = []
    for category, path in resolved:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
                deleted.append({"category": category, "path": str(path)})
            elif path.is_dir():
                shutil.rmtree(path)
                deleted.append({"category": category, "path": str(path)})
            else:
                missing.append({"category": category, "path": str(path)})
        except Exception as exc:  # retain an exact audit log before failing
            errors.append({"category": category, "path": str(path), "error": repr(exc)})
            break

    still_present = [str(path) for _, path in resolved if path.exists() or path.is_symlink()]
    protected_missing = [str(path) for path in selected if not path.is_dir()]
    retention_weights = list(retention_root.rglob("model.safetensors"))
    result = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "started_unix": started,
        "finished_unix": time.time(),
        "manifest_targets": len(resolved),
        "deleted_targets": len(deleted),
        "already_missing_targets": len(missing),
        "errors": errors,
        "still_present_targets": still_present,
        "protected_selected_directories_missing": protected_missing,
        "retention_model_safetensors": len(retention_weights),
        "deleted": deleted,
        "already_missing": missing,
    }
    args.execution_log.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "manifest_targets",
        "deleted_targets",
        "already_missing_targets",
        "errors",
        "still_present_targets",
        "protected_selected_directories_missing",
        "retention_model_safetensors",
    )}, indent=2))

    if errors or still_present or protected_missing or len(retention_weights) != 36:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
