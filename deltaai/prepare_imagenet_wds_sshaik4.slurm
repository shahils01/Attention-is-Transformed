#!/usr/bin/env bash
# Convert the authorized Kaggle ImageNet archive directly to WebDataset shards.
# This avoids materializing the annotation and test trees and is safe to rerun.
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

REPO_DIR=/u/sshaik4/Attention-is-Transformed
ARCHIVE=${ARCHIVE:-/work/hdd/biad/sshaik4/lgma_data/imagenet/archive/imagenet-object-localization-challenge.zip}
OUTPUT_DIR=${OUTPUT_DIR:-/work/nvme/biad/sshaik4/lgma_data/imagenet_wds}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1024}

mkdir -p "${OUTPUT_DIR}" /work/hdd/biad/sshaik4/logs
cd "${REPO_DIR}"

python experiments/prepare_imagenet_kaggle.py \
  --archive "${ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --samples-per-shard "${SAMPLES_PER_SHARD}"

