#!/usr/bin/env bash
#SBATCH --job-name=gt-head-diversity
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/hdd/bifg/arai3/lgma_jobs/gt_head_diversity_20260923
#SBATCH --output=/work/hdd/bifg/arai3/lgma_jobs/gt_head_diversity_20260923/%j.out
#SBATCH --error=/work/hdd/bifg/arai3/lgma_jobs/gt_head_diversity_20260923/%j.err

set -eo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bifg/arai3/venvs/lgma-clean/bin/activate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/work/hdd/bifg/arai3/lgma_data/.hf_cache
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python analyze_head_diversity.py \
  --task tinystories \
  --batch-size 8 \
  --causal-query-start 16 \
  --output-dir results/tinystories

python analyze_head_diversity.py \
  --task bert \
  --bert-seed 43 \
  --bert-examples 1000 \
  --batch-size 8 \
  --output-dir results/bert_seed43
