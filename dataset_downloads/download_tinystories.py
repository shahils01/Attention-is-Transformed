from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
V2_GPT4_FILES = [
    "TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt",
]


def default_data_dir() -> Path:
    if "DATA_DIR" in os.environ:
        return Path(os.environ["DATA_DIR"])
    if "SCRATCH" in os.environ:
        return Path(os.environ["SCRATCH"]) / "lgma_data"
    return Path("data")


def download_file(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"exists: {destination}")
        return
    print(f"downloading: {url}")
    print(f"        to: {destination}")
    urlretrieve(url, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TinyStories text files.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_data_dir() / "tinystories",
        help="Destination directory. Defaults to $DATA_DIR/tinystories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename in V2_GPT4_FILES:
        download_file(
            f"{BASE_URL}/{filename}",
            args.output_dir / filename,
            force=args.force,
        )
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
