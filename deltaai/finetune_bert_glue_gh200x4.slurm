#!/usr/bin/env bash
# Fine-tune a completed BERT MLM checkpoint on one GLUE task.
#
# On sshaik4:
#   sbatch --export=ALL,TASK=mnli,ATTENTION_TYPE=gt_mha_residual,\
# CHECKPOINT_DIR=/work/hdd/biad/sshaik4/lgma_runs/<mlm-run>,SEED=42 \
#     deltaai/finetune_bert_glue_gh200x4.slurm
#
# For another account, override the Slurm account/log paths on the sbatch
# command line and export its PROJECT_DIR, REPO_DIR, and VENV_DIR.
#SBATCH --job-name=bert-glue
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/work/hdd/biad/sshaik4}
REPO_DIR=${REPO_DIR:-/u/sshaik4/Attention-is-Transformed}
VENV_DIR=${VENV_DIR:-${PROJECT_DIR}/venvs/lgma-clean}
TASK=${TASK:?Set a GLUE task such as mnli, qnli, sst2, or cola}
ATTENTION_TYPE=${ATTENTION_TYPE:?Set ATTENTION_TYPE to the checkpoint attention type}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
SEED=${SEED:-42}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a completed MLM output directory}
WANDB_PROJECT=${WANDB_PROJECT:-bert-gt-mha-glue}

case "${ATTENTION_TYPE}" in
  mha)
    ATTENTION_ARGS=(--attention-type mha)
    ;;
  gqa)
    ATTENTION_ARGS=(--attention-type gqa --num-kv-heads "${NUM_KV_HEADS}")
    ;;
  mqa)
    ATTENTION_ARGS=(--attention-type mqa)
    ;;
  collaborative)
    ATTENTION_ARGS=(--attention-type collaborative)
    ;;
  gt_mha_exact|gt_mha_residual|gt_mha_quadratic)
    ATTENTION_ARGS=(
      --attention-type "${ATTENTION_TYPE}"
    )
    ;;
  *)
    echo "Unsupported ATTENTION_TYPE=${ATTENTION_TYPE}" >&2
    exit 2
    ;;
esac

CHECKPOINT_NAME=$(basename "${CHECKPOINT_DIR}")
RUN_NAME=${CHECKPOINT_NAME}_glue_${TASK}_seed${SEED}_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4
export WANDB_PROJECT
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
  --model-name-or-path "${CHECKPOINT_DIR}" \
  --task "${TASK}" \
  "${ATTENTION_ARGS[@]}" \
  --max-sequence-length 128 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --num-train-epochs 3 \
  --seed "${SEED}" \
  --bf16 \
  --output-dir "${OUTPUT_DIR}" \
  --report-to wandb \
  --run-name "${RUN_NAME}"
