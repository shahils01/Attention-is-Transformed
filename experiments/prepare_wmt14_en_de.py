#!/usr/bin/env python3
"""Create shared, reproducible WMT14 EN→DE assets for every attention method.

Input is the official ``wmt/wmt14`` ``de-en`` DatasetDict saved by
``palmetto/download_wmt14_en_de.slurm``.  This program does no model-specific
processing: all baselines and GT-MHA must use the resulting tokenizer and
indexed token sequences verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentencepiece as spm
from datasets import load_from_disk


WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def keep_pair(en: str, de: str, max_words: int, max_ratio: float) -> bool:
    en_words, de_words = len(en.split()), len(de.split())
    if not en_words or not de_words or max(en_words, de_words) > max_words:
        return False
    return max(en_words / de_words, de_words / en_words) <= max_ratio


def write_parallel_text(dataset, split: str, root: Path, *, filter_train: bool, args) -> int:
    en_path, de_path = root / f"{split}.en", root / f"{split}.de"
    if en_path.exists() and de_path.exists():
        return sum(1 for _ in en_path.open(encoding="utf-8"))
    kept = 0
    with en_path.open("w", encoding="utf-8") as en_out, de_path.open("w", encoding="utf-8") as de_out:
        for row in dataset[split]:
            pair = row["translation"]
            en, de = normalize(pair["en"]), normalize(pair["de"])
            if not en or not de or (filter_train and not keep_pair(en, de, args.max_words, args.max_ratio)):
                continue
            en_out.write(en + "\n")
            de_out.write(de + "\n")
            kept += 1
    return kept


def train_tokenizer(text_root: Path, output_root: Path, vocab_size: int) -> Path:
    model_prefix = output_root / "wmt14_en_de_spm"
    model_path = model_prefix.with_suffix(".model")
    if model_path.exists():
        return model_path
    # Train only on WMT14 train text; valid/test are never used for vocabulary fitting.
    spm.SentencePieceTrainer.train(
        input=f"{text_root / 'train.en'},{text_root / 'train.de'}",
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        normalization_rule_name="nmt_nfkc",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        hard_vocab_limit=False,
        shuffle_input_sentence=True,
    )
    return model_path


def encode_file(text_path: Path, processor: spm.SentencePieceProcessor, output_prefix: Path) -> dict:
    # Keep the language extension: ``train.en`` must not collide with
    # ``train.de`` as ``Path.with_suffix`` would turn both into ``train.bin``.
    token_path = Path(f"{output_prefix}.bin")
    index_path = Path(f"{output_prefix}.idx.npy")
    if token_path.exists() and index_path.exists():
        offsets = np.load(index_path, mmap_mode="r")
        return {"sequences": int(len(offsets) - 1), "tokens": int(offsets[-1])}
    offsets = [0]
    total = 0
    with token_path.open("wb") as token_out, text_path.open(encoding="utf-8") as text_in:
        for line in text_in:
            ids = processor.encode(line.rstrip("\n"), out_type=int) + [processor.eos_id()]
            np.asarray(ids, dtype=np.uint32).tofile(token_out)
            total += len(ids)
            offsets.append(total)
    np.save(index_path, np.asarray(offsets, dtype=np.uint64))
    return {"sequences": len(offsets) - 1, "tokens": total}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/scratch/shahils/lgma_data/wmt14_en_de"))
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--max-ratio", type=float, default=3.0)
    args = parser.parse_args()
    if args.vocab_size < 256 or args.max_words < 1 or args.max_ratio < 1:
        raise SystemExit("invalid tokenizer or filtering argument")

    raw = args.data_root / "raw_hf"
    if not raw.exists():
        raise SystemExit(f"missing raw dataset: {raw}")
    text_root, output_root = args.data_root / "clean_text", args.data_root / "processed_spm32k"
    text_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(str(raw))
    text_counts = {
        "train": write_parallel_text(dataset, "train", text_root, filter_train=True, args=args),
        "validation": write_parallel_text(dataset, "validation", text_root, filter_train=False, args=args),
        "test": write_parallel_text(dataset, "test", text_root, filter_train=False, args=args),
    }
    model_path = train_tokenizer(text_root, output_root, args.vocab_size)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    encoded = {}
    for split in ("train", "validation", "test"):
        encoded[split] = {
            "en": encode_file(text_root / f"{split}.en", processor, output_root / f"{split}.en"),
            "de": encode_file(text_root / f"{split}.de", processor, output_root / f"{split}.de"),
        }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "wmt/wmt14 de-en; direction EN→DE",
        "tokenizer": {"model": model_path.name, "vocab_size": processor.vocab_size(), "sha256": sha256(model_path),
                      "special_ids": {"unk": processor.unk_id(), "bos": processor.bos_id(), "eos": processor.eos_id(), "pad": processor.pad_id()}},
        "normalization": "Unicode NFKC plus whitespace canonicalization",
        "training_filter": {"max_whitespace_words": args.max_words, "max_length_ratio": args.max_ratio},
        "text_pairs": text_counts,
        "encoded": encoded,
        "storage": "uint32 .bin token streams + uint64 .idx.npy sequence offsets; EOS appended to each sequence",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
