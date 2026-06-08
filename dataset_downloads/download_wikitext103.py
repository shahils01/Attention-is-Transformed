from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_data_dir() -> Path:
    if "DATA_DIR" in os.environ:
        return Path(os.environ["DATA_DIR"])
    if "SCRATCH" in os.environ:
        return Path(os.environ["SCRATCH"]) / "lgma_data"
    return Path("data")


def configure_hf_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install -e \".[data]\"`."
        ) from exc
    return load_dataset


def write_split(dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row["text"].strip()
            if text:
                handle.write(text + "\n")
    print(f"wrote: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WikiText-103 as plain text.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_data_dir() / "wikitext103",
        help="Destination directory. Defaults to $DATA_DIR/wikitext103.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(os.environ["HF_HOME"]) if "HF_HOME" in os.environ else None,
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.cache_dir)
    load_dataset = require_datasets()
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    for split in args.splits:
        write_split(dataset[split], args.output_dir / f"{split}.txt")
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
