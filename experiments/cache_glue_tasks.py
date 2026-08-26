from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate Hugging Face GLUE task caches.")
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    for task in args.tasks:
        dataset = load_dataset("nyu-mll/glue", task)
        sizes = {split: len(rows) for split, rows in dataset.items()}
        print(f"Cached nyu-mll/glue/{task}: {sizes}", flush=True)


if __name__ == "__main__":
    main()
