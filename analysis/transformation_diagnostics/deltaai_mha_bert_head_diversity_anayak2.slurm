#!/usr/bin/env bash
#SBATCH --job-name=mha-bert-div
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/hdd/bicn/anayak2/lgma_jobs/mha_head_diversity_20260924
#SBATCH --output=/work/hdd/bicn/anayak2/lgma_jobs/mha_head_diversity_20260924/%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/lgma_jobs/mha_head_diversity_20260924/%j.err

set -euo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bicn/anayak2/venvs/lgma-clean/bin/activate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/work/hdd/bicn/anayak2/lgma_data/.hf_cache
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

PROJECT=/work/hdd/bicn/anayak2
SOURCE_ROOT=${PROJECT}/repos/Attention-is-Transformed
VALIDATION=${PROJECT}/lgma_data/.hf_cache/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow
BERT_CHECKPOINT=${PROJECT}/lgma_runs/bert_base_mha_random_seed42_v1/checkpoint-100000

test -f "${BERT_CHECKPOINT}/model.safetensors"
test -f "${VALIDATION}"
test "$(sha256sum "${VALIDATION}" | awk '{print $1}')" = 8e136ab7130c5fca602eb406ce2557967a974d901b53c4f2992f87b873e0fe5b
BERT_SHA=$(sha256sum "${BERT_CHECKPOINT}/model.safetensors" | awk '{print $1}')
python analyze_mha_head_diversity.py \
  --task bert \
  --checkpoint "${BERT_CHECKPOINT}" \
  --checkpoint-sha256 "${BERT_SHA}" \
  --source-root "${SOURCE_ROOT}" \
  --bert-validation-file "${VALIDATION}" \
  --bert-seed 42 \
  --bert-examples 1000 \
  --ablation-examples 128 \
  --inference-precision fp32 \
  --batch-size 8 \
  --output-dir results/bert_seed42_fp32
