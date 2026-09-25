#!/usr/bin/env bash
# Matched 1,000-step continued-MLM control and transform-diversity pilot for C=2.
# One GPU preserves the original global batch of 256 through gradient accumulation.
#SBATCH --job-name=bert-c2-div-int
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/hdd/bicn/anayak2/lgma_jobs/bert_transform_diversity_pilot_20260924
#SBATCH --output=/work/hdd/bicn/anayak2/lgma_jobs/bert_transform_diversity_pilot_20260924/%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/lgma_jobs/bert_transform_diversity_pilot_20260924/%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bicn/anayak2
STAGE_DIR=${PROJECT_DIR}/lgma_jobs/bert_transform_diversity_pilot_20260924
OUTPUT_ROOT=${PROJECT_DIR}/lgma_runs/bert_transform_diversity_pilot_20260924
C2_CHECKPOINT=${PROJECT_DIR}/lgma_runs/bert_base_gt_mha_residual_b2g8h12_random_seed42_explicit_20260911
C2_SOURCE=${PROJECT_DIR}/lgma_jobs/bert_b2g8h12_20260911/repo

module load python/miniforge3_pytorch/2.12.0
source "${PROJECT_DIR}/venvs/lgma-clean/bin/activate"
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/lgma_jobs/bert_b1g8h12_seed43_20260919/compat/no_transformer_engine:${PYTHONPATH:-}

mkdir -p "${OUTPUT_ROOT}"

run_branch() {
  local variant=$1
  local weight=$2
  local output_dir=${OUTPUT_ROOT}/c2_${variant}
  test ! -e "${output_dir}" || {
    echo "Refusing to overwrite ${output_dir}" >&2
    exit 1
  }
  echo "Starting C=2 ${variant} at $(date -Is)"
  python "${STAGE_DIR}/finetune_bert_transform_diversity.py" \
    --source-root "${C2_SOURCE}" \
    --checkpoint-root "${C2_CHECKPOINT}" \
    --output-dir "${output_dir}" \
    --expected-num-bases 2 \
    --diversity-weight "${weight}" \
    --margin-multiplier 1.25 \
    --max-steps 1000 \
    --learning-rate 1e-5 \
    --warmup-steps 100 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 8 \
    --seed 91337
  echo "Completed C=2 ${variant} at $(date -Is)"
}

run_branch control 0.0
run_branch diversity 0.001

echo "C=2 matched pilot completed at $(date -Is)"
