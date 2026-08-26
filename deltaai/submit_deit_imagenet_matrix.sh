#!/usr/bin/env bash
# Submit the five controlled primary comparisons. Set SEED or other sbatch
# exports before invoking this script when reproducing additional seeds.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SEED=${SEED:-42}

for ATTENTION_TYPE in mha reduced_mha gqa collaborative gt_mha_residual; do
  sbatch \
    --export=ALL,ATTENTION_TYPE="${ATTENTION_TYPE}",SEED="${SEED}" \
    "${SCRIPT_DIR}/train_deit_imagenet_gh200x4.slurm"
done

