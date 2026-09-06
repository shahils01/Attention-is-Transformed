#!/usr/bin/env bash
# Convert the authorized ImageNet-1K Kaggle archive into the same deterministic,
# class-mixed WebDataset protocol used by the DeltaAI experiments.
#SBATCH --job-name=imagenet-wds
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shahils/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/scratch/shahils/Attention-is-Transformed-vision}
ARCHIVE=${ARCHIVE:-/scratch/shahils/imagenet/archive/imagenet-object-localization-challenge.zip}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/shahils/imagenet_wds_official}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1024}
TRAIN_SHUFFLE_SEED=${TRAIN_SHUFFLE_SEED:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-}

module load anaconda3/2023.09-0
export PYTHONNOUSERSITE=1

mkdir -p "${OUTPUT_DIR}" /scratch/shahils/logs
cd "${REPO_DIR}"

test -f "${ARCHIVE}"

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
