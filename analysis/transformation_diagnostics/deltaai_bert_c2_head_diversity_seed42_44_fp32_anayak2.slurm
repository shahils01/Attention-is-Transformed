#!/usr/bin/env bash
#SBATCH --job-name=bert-c2-div-s42-44
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/hdd/bicn/anayak2/lgma_jobs/bert_c2_head_diversity_seed42_44_fp32_20260924
#SBATCH --output=/work/hdd/bicn/anayak2/lgma_jobs/bert_c2_head_diversity_seed42_44_fp32_20260924/%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/lgma_jobs/bert_c2_head_diversity_seed42_44_fp32_20260924/%j.err

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
  --bert-seed 42 \
  --bert-examples 1000 \
  --expected-num-bases 2 \
  --batch-size 8 \
  --inference-precision fp32 \
  --bert-checkpoint-dir /work/hdd/bicn/anayak2/lgma_runs/bert_base_gt_mha_residual_b2g8h12_random_seed42_explicit_20260911/checkpoint-100000 \
  --bert-checkpoint-sha256 c6c5b2757bda16eeb8ee7749f0224b20213b2693347253f3e871137975550a3b \
  --bert-source-root /work/hdd/bicn/anayak2/lgma_jobs/bert_b2g8h12_20260911/repo \
  --bert-validation-file /work/hdd/bicn/anayak2/lgma_data/.hf_cache/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow \
  --output-dir results/bert_c2_seed42_fp32

python analyze_head_diversity.py \
  --task bert \
  --bert-seed 44 \
  --bert-examples 1000 \
  --expected-num-bases 2 \
  --batch-size 8 \
  --inference-precision fp32 \
  --bert-checkpoint-dir /work/hdd/bicn/anayak2/lgma_runs/bert_base_gt_mha_residual_b2g8h12_random_seed44_explicit_20260912/checkpoint-100000 \
  --bert-checkpoint-sha256 affdb6afb7a722682e53530c47647fc2b59297a0a9d4e7622f3d9e5bdb9e1575 \
  --bert-source-root /work/hdd/bicn/anayak2/lgma_jobs/bert_b2g8h12_seed44_20260912/repo \
  --bert-validation-file /work/hdd/bicn/anayak2/lgma_data/.hf_cache/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow \
  --output-dir results/bert_c2_seed44_fp32
