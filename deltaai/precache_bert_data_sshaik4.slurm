#!/usr/bin/env bash
# Pre-cache the official BERT checkpoint, Wikitext-103, GLUE data, and GLUE
# evaluation metrics on sshaik4 shared storage before allocating training GPUs.
#SBATCH --job-name=bert-precache
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/biad/sshaik4
VENV_DIR=${PROJECT_DIR}/venvs/lgma
CACHE_DIR=${PROJECT_DIR}/lgma_data/.hf_cache
MODEL_NAME=${MODEL_NAME:-google-bert/bert-base-uncased}
export MODEL_NAME

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${PROJECT_DIR}/logs"

python - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import evaluate
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

model_name = os.environ.get("MODEL_NAME", "google-bert/bert-base-uncased")
cache_dir = Path(os.environ["HF_HOME"])
glue_tasks = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte")

print(f"Caching model and tokenizer: {model_name}", flush=True)
AutoConfig.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name, use_fast=True)
AutoModelForMaskedLM.from_pretrained(model_name)

print("Caching Wikitext-103", flush=True)
wikitext = load_dataset("wikitext", "wikitext-103-raw-v1")
print({split: len(dataset) for split, dataset in wikitext.items()}, flush=True)

glue_sizes = {}
for task in glue_tasks:
    print(f"Caching GLUE/{task} and its evaluation metric", flush=True)
    dataset = load_dataset("glue", task)
    evaluate.load("glue", task)
    glue_sizes[task] = {split: len(data) for split, data in dataset.items()}

manifest = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "model": model_name,
    "wikitext": {split: len(dataset) for split, dataset in wikitext.items()},
    "glue": glue_sizes,
}
(cache_dir / "bert_precache_complete.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
PY

du -sh "${CACHE_DIR}"
