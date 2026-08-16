#!/usr/bin/env bash
# Download the official Hugging Face mirror of WMT14 EN-DE once, to shared scratch.
# This is intentionally a CPU-only data-preparation job; no model-specific data is made here.
#SBATCH --job-name=wmt14-en-de-download
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shahils/lgma_data/wmt14_en_de/logs/download_%j.out
#SBATCH --error=/scratch/shahils/lgma_data/wmt14_en_de/logs/download_%j.err

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -euo pipefail

DATA_ROOT=/scratch/shahils/lgma_data/wmt14_en_de
export HF_HOME=/scratch/shahils/hf_cache
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${DATA_ROOT}/logs" "${HF_HOME}"

python - <<'PY'
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

root = Path('/scratch/shahils/lgma_data/wmt14_en_de')
destination = root / 'raw_hf'
staging = root / 'raw_hf.staging'

if destination.exists():
    print(f'{destination} already exists; leaving the existing download unchanged.')
else:
    if staging.exists():
        shutil.rmtree(staging)
    # Official WMT14 dataset mirror; config de-en exposes translation {de, en} pairs.
    dataset = load_dataset('wmt/wmt14', 'de-en', cache_dir=os.environ['HF_DATASETS_CACHE'])
    print(dataset)
    dataset.save_to_disk(str(staging))
    staging.rename(destination)

from datasets import load_from_disk
dataset = load_from_disk(str(destination))
manifest = {
    'source': 'wmt/wmt14',
    'config': 'de-en',
    'language_direction': 'en-de',
    'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
    'splits': {name: len(split) for name, split in dataset.items()},
    'format': 'Hugging Face DatasetDict saved_to_disk',
    'next_step': 'shared filtering, SentencePiece tokenizer training, and fixed tokenized train/valid/test files',
}
(root / 'raw_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
