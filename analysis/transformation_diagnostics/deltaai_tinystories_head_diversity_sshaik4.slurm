#!/usr/bin/env bash
#SBATCH --job-name=tiny-head-div
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/hdd/biad/sshaik4/lgma_jobs/tinystories_head_diversity_20260923
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/tinystories_head_diversity_20260923/%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/tinystories_head_diversity_20260923/%j.err

set -eo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-bert-no-te/bin/activate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/work/hdd/biad/sshaik4/lgma_data/.hf_cache
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python analyze_head_diversity.py \
  --task tinystories \
  --batch-size 8 \
  --causal-query-start 16 \
  --tinystories-checkpoint "/work/hdd/biad/sshaik4/lgma_checkpoints/paper_gt_mha/TinyStories-v2 Checkpoints/GT-MHA residual/checkpoint_step_250000.pt" \
  --tinystories-checkpoint-sha256 a3b36ada91bb42b22b6df4da2b36be5cd3333f5a473e857c326aa06323ae06f2 \
  --tinystories-source-root /work/hdd/biad/sshaik4/repos/Attention-is-Transformed \
  --tinystories-data-dir /work/hdd/biad/sshaik4/lgma_data/tinystories \
  --output-dir results/tinystories
