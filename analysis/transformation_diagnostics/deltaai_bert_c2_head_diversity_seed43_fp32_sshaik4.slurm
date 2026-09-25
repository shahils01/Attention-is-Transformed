#!/usr/bin/env bash
#SBATCH --job-name=bert-c2-div-s43
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/hdd/biad/sshaik4/lgma_jobs/bert_c2_head_diversity_seed43_fp32_20260924
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/bert_c2_head_diversity_seed43_fp32_20260924/%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/bert_c2_head_diversity_seed43_fp32_20260924/%j.err

set -eo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-bert-no-te/bin/activate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/work/hdd/biad/sshaik4/lgma_data/.hf_cache
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python analyze_head_diversity.py \
  --task bert \
  --bert-seed 43 \
  --bert-examples 1000 \
  --expected-num-bases 2 \
  --batch-size 8 \
  --inference-precision fp32 \
  --bert-checkpoint-dir /work/hdd/biad/sshaik4/lgma_runs/bert_base_gt_mha_residual_b2g8h12_random_seed43_explicit_20260911/checkpoint-100000 \
  --bert-checkpoint-sha256 71387b86a5d7d07f6ac9821eb87e541c74c062cd44dc66d728b342ac031b4f30 \
  --bert-source-root /work/hdd/biad/sshaik4/lgma_jobs/bert_b2g8h12_seed43_20260911/repo \
  --bert-validation-file /work/hdd/biad/sshaik4/lgma_data/.hf_cache/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow \
  --output-dir results/bert_c2_seed43_fp32
