from __future__ import annotations

import argparse
import itertools
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a streaming C4 English subset.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_data_dir() / "c4_subset",
        help="Destination directory. Defaults to $DATA_DIR/c4_subset.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(os.environ["HF_HOME"]) if "HF_HOME" in os.environ else None,
        help="Hugging Face cache directory.",
    )
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")

    configure_hf_cache(args.cache_dir)
    load_dataset = require_datasets()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"train_{args.rows}.txt"
    output_path = args.output_dir / output_name

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in itertools.islice(dataset, args.rows):
            text = row["text"].strip()
            if not text:
                continue
            handle.write(text.replace("\n", " ") + "\n")
            written += 1

    print(f"wrote {written} rows: {output_path}")


if __name__ == "__main__":
    main()
