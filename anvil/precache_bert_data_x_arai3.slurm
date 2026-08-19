#!/usr/bin/env bash
# Pre-cache BERT assets, WikiText-103, GLUE datasets, and GLUE metrics on Anvil.
# The cis261553-ai association requires the ai partition and at least one GPU,
# although this staging workload itself is CPU/I/O bound.
#SBATCH --job-name=bert-precache
#SBATCH --account=cis261553-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/anvil/projects/x-cis261553/logs/%x_%j.out
#SBATCH --error=/anvil/projects/x-cis261553/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/anvil/projects/x-cis261553
REPO_DIR=/home/x-arai3/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-bert
CACHE_DIR=${PROJECT_DIR}/lgma_data/.hf_cache
MODEL_NAME=google-bert/bert-base-uncased

module purge >/dev/null 2>&1 || true
module load anaconda/2024.02-py311
source activate "${VENV_DIR}"

export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"

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

print("Caching WikiText-103", flush=True)
wikitext = load_dataset("wikitext", "wikitext-103-raw-v1")
print({split: len(dataset) for split, dataset in wikitext.items()}, flush=True)

glue_sizes = {}
for task in glue_tasks:
    print(f"Caching GLUE/{task} and its metric", flush=True)
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
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
PY

du -sh "${CACHE_DIR}"
