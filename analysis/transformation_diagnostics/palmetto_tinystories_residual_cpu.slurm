#!/usr/bin/env bash
#SBATCH --job-name=gt-transform-diag
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shahils/tinystories_transform_analysis/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/tinystories_transform_analysis/logs/%x_%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u

ANALYSIS_ROOT=/scratch/shahils/tinystories_transform_analysis
SOURCE_ROOT=/scratch/shahils/tinystories_eval_20260919
CHECKPOINT_SHA=a3b36ada91bb42b22b6df4da2b36be5cd3333f5a473e857c326aa06323ae06f2

mkdir -p "${ANALYSIS_ROOT}/logs" "${ANALYSIS_ROOT}/results/residual_step250000"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

python -u "${ANALYSIS_ROOT}/analyze_checkpoint.py" \
  --checkpoint "${SOURCE_ROOT}/checkpoints/${CHECKPOINT_SHA}.pt" \
  --expected-sha256 "${CHECKPOINT_SHA}" \
  --tokenizer "${SOURCE_ROOT}/tokenizer.json" \
  --repo-root "${SOURCE_ROOT}/repo" \
  --model-name "GT-MHA residual TinyStories step 250000" \
  --output-dir "${ANALYSIS_ROOT}/results/residual_step250000"
