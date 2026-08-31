#!/usr/bin/env bash
# Evaluate the completed DeiT-B/16 MHA and GT-MHA checkpoints sequentially on
# the exact 50,000-image ImageNet-1K validation WebDataset cache.
#
# Smoke test:
#   sbatch --export=ALL,MAX_VAL_BATCHES=2 deltaai/eval_deit_imagenet_anayak2.slurm
# Full evaluation:
#   sbatch deltaai/eval_deit_imagenet_anayak2.slurm

#SBATCH --job-name=deit-eval
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/anayak2/Attention-is-Transformed-vision}
VENV_DIR=${VENV_DIR:-/work/hdd/bicn/anayak2/venvs/lgma-vision}
DATA_DIR=${DATA_DIR:-/work/nvme/bicn/anayak2/lgma_data/imagenet_wds_val}
EVAL_ROOT=${EVAL_ROOT:-/work/hdd/bicn/anayak2/lgma_runs/imagenet_eval}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${EVAL_ROOT}/checkpoints}
RESULTS_DIR=${RESULTS_DIR:-${EVAL_ROOT}/results}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-}

MHA_CHECKPOINT=${MHA_CHECKPOINT:-${CHECKPOINT_DIR}/deit_b16_mha_seed42_checkpoint_best.pt}
GT_CHECKPOINT=${GT_CHECKPOINT:-${CHECKPOINT_DIR}/deit_b16_gt_mha_residual_seed42_b4g8_checkpoint_best.pt}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS=4
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${RESULTS_DIR}" /work/hdd/bicn/anayak2/logs
cd "${REPO_DIR}"

test -f "${DATA_DIR}/manifest.json"
test -f "${MHA_CHECKPOINT}"
test -f "${GT_CHECKPOINT}"

python -c 'import timm, torch, webdataset; print(torch.__version__, timm.__version__, webdataset.__version__)'

EXTRA_ARGS=()
RUN_LABEL=full
if [[ -n "${MAX_VAL_BATCHES}" ]]; then
  EXTRA_ARGS=(--max-val-batches "${MAX_VAL_BATCHES}")
  RUN_LABEL=smoke_${MAX_VAL_BATCHES}_batches
fi

evaluate_checkpoint() {
  local attention_type=$1
  local checkpoint=$2
  local output_name=$3
  shift 3

  local output_dir="${RESULTS_DIR}/${output_name}_${RUN_LABEL}"
  mkdir -p "${output_dir}"

  python experiments/train_imagenet_deit.py \
    --data-dir "${DATA_DIR}" \
    --dataset-format wds \
    --output-dir "${output_dir}" \
    --attention-type "${attention_type}" \
    --batch-size 256 \
    --workers 12 \
    --precision bf16 \
    --seed 42 \
    --eval-only \
    --resume "${checkpoint}" \
    --wandb-mode disabled \
    "${EXTRA_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${output_dir}/evaluation.log"
}

evaluate_checkpoint \
  mha \
  "${MHA_CHECKPOINT}" \
  deit_b16_mha_seed42

evaluate_checkpoint \
  gt_mha_residual \
  "${GT_CHECKPOINT}" \
  deit_b16_gt_mha_residual_seed42_b4g8 \
  --num-base-heads 4 \
  --num-generators 8 \
  --generator-mixing softmax

echo "Both evaluations completed successfully. Results: ${RESULTS_DIR}"
