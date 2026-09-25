#!/usr/bin/env bash
#SBATCH --job-name=bert-head-div-s44
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/hdd/bicn/anayak2/lgma_jobs/bert_head_diversity_seed44_20260923
#SBATCH --output=/work/hdd/bicn/anayak2/lgma_jobs/bert_head_diversity_seed44_20260923/%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/lgma_jobs/bert_head_diversity_seed44_20260923/%j.err

set -eo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bicn/anayak2/venvs/lgma-clean/bin/activate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/work/hdd/bicn/anayak2/lgma_data/.hf_cache
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python analyze_head_diversity.py \
  --task bert \
  --bert-seed 44 \
  --bert-examples 1000 \
  --batch-size 8 \
  --bert-checkpoint-dir /work/hdd/bicn/anayak2/lgma_runs/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000 \
  --bert-checkpoint-sha256 f37a0546162df0a386c445050ba9f9fa664f65eacf0a1fa9bd5523fd45d9b643 \
  --bert-source-root /work/hdd/bicn/anayak2/repos/Attention-is-Transformed \
  --bert-validation-file /work/hdd/bicn/anayak2/lgma_data/.hf_cache/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow \
  --output-dir results/bert_seed44
