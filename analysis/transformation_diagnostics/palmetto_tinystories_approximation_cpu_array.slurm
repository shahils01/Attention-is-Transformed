#!/usr/bin/env bash
#SBATCH --job-name=gt-map-approx
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/shahils/tinystories_transform_analysis/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shahils/tinystories_transform_analysis/logs/%x_%A_%a.err

set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u

ANALYSIS_ROOT=/scratch/shahils/tinystories_transform_analysis
SOURCE_ROOT=/scratch/shahils/tinystories_eval_20260919

NAMES=(
  residual_step250000
  quadratic_step250000
  exact_step250000
)
MODEL_NAMES=(
  "GT-MHA first-order residual TinyStories step 250000"
  "GT-MHA second-order quadratic TinyStories step 250000"
  "GT-MHA exact exponential TinyStories step 250000"
)
CHECKPOINTS=(
  /scratch/shahils/lgma_runs/tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100/checkpoint_step_250000.pt
  /scratch/shahils/lgma_runs/tinystories_lgma_quad_b4_h16_h200_4/checkpoint_step_250000.pt
  /scratch/shahils/lgma_runs/tinystories_lgma_b4_h16_g8_mixnone_theta035_ddp4_gb2048/checkpoint_step_250000.pt
)

INDEX=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$INDEX]}
MODEL_NAME=${MODEL_NAMES[$INDEX]}
CHECKPOINT=${CHECKPOINTS[$INDEX]}
OUTPUT_DIR=${ANALYSIS_ROOT}/results/approximation/${NAME}

test -s "${CHECKPOINT}"
test -s "${SOURCE_ROOT}/tokenizer.json"
test -d "${SOURCE_ROOT}/repo/src/lgma"
mkdir -p "${ANALYSIS_ROOT}/logs" "${OUTPUT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python -u "${ANALYSIS_ROOT}/analyze_approximation.py" \
  --checkpoint "${CHECKPOINT}" \
  --tokenizer "${SOURCE_ROOT}/tokenizer.json" \
  --repo-root "${SOURCE_ROOT}/repo" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"
