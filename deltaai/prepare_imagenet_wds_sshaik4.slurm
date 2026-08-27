#!/usr/bin/env bash
# Convert the authorized official ImageNet-1K competition archive into
# deterministically class-mixed WebDataset shards.
#
#   sbatch deltaai/prepare_imagenet_wds_sshaik4.slurm
#SBATCH --job-name=imagenet-wds
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/sshaik4/Attention-is-Transformed-vision}
VENV_DIR=/work/hdd/biad/sshaik4/venvs/lgma-clean
ARCHIVE=${ARCHIVE:-/work/hdd/biad/sshaik4/lgma_data/imagenet/archive/imagenet-object-localization-challenge.zip}
OUTPUT_DIR=${OUTPUT_DIR:-/work/nvme/biad/sshaik4/lgma_data/imagenet_wds_official}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1024}
TRAIN_SHUFFLE_SEED=${TRAIN_SHUFFLE_SEED:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
mkdir -p "${OUTPUT_DIR}" /work/hdd/biad/sshaik4/logs
cd "${REPO_DIR}"

LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  LIMIT_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  LIMIT_ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi

python -m experiments.prepare_imagenet_kaggle \
  --archive "${ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --samples-per-shard "${SAMPLES_PER_SHARD}" \
  --train-shuffle-seed "${TRAIN_SHUFFLE_SEED}" \
  "${LIMIT_ARGS[@]}"
