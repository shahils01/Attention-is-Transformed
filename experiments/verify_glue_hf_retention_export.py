from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = (
        args.staging_root
        / "Fine-tuned GLUE Checkpoints"
        / "official_glue_submission"
        / "UPLOAD_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text())
    info = HfApi().model_info(args.repo_id, files_metadata=True)
    remote = {sibling.rfilename: sibling for sibling in info.siblings}

    for record in manifest["files"]:
        name = record["path"]
        if name not in remote:
            raise FileNotFoundError(f"missing from HF: {name}")
        sibling = remote[name]
        if sibling.size != record["size"]:
            raise ValueError(
                f"size mismatch for {name}: {sibling.size} != {record['size']}"
            )
        if name.endswith("model.safetensors"):
            if sibling.lfs is None:
                raise ValueError(f"weight is not stored with LFS/Xet metadata: {name}")
            if sibling.lfs.sha256 != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for weight: {name}")

    remote_manifest = Path(
        hf_hub_download(
            args.repo_id,
            filename=str(manifest_path.relative_to(args.staging_root)),
            force_download=True,
        )
    )
    if sha256(remote_manifest) != sha256(manifest_path):
        raise ValueError("remote UPLOAD_MANIFEST.json does not match local manifest")

    print(
        json.dumps(
            {
                "verified_files": len(manifest["files"]),
                "verified_model_weights": sum(
                    record["path"].endswith("model.safetensors")
                    for record in manifest["files"]
                ),
                "total_bytes": manifest["total_bytes"],
                "repo_commit": info.sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
