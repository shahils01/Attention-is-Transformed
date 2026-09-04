#!/usr/bin/env bash
# Controlled DeiT-B/16 ImageNet-1K training. Examples:
#
#   sbatch --export=ALL,ATTENTION_TYPE=mha deltaai/train_deit_imagenet_gh200x4.slurm
#   sbatch --export=ALL,ATTENTION_TYPE=gt_mha_residual,NUM_BASE_HEADS=4 deltaai/train_deit_imagenet_gh200x4.slurm
#SBATCH --job-name=deit-attn
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/sshaik4/Attention-is-Transformed-vision}
VENV_DIR=/work/hdd/biad/sshaik4/venvs/lgma-vision
DATA_DIR=${DATA_DIR:-/work/nvme/biad/sshaik4/lgma_data/imagenet_wds_official}
RUNS_DIR=${RUNS_DIR:-/work/hdd/biad/sshaik4/lgma_runs/imagenet}
ATTENTION_TYPE=${ATTENTION_TYPE:-gt_mha_residual}
NUM_BASE_HEADS=${NUM_BASE_HEADS:-4}
NUM_VALUE_BASE_HEADS=${NUM_VALUE_BASE_HEADS:-${NUM_BASE_HEADS}}
NUM_GENERATORS=${NUM_GENERATORS:-8}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
REDUCED_QK_DIM=${REDUCED_QK_DIM:-384}
COLLABORATIVE_QK_DIM=${COLLABORATIVE_QK_DIM:-384}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-256}
WANDB_MODE=${WANDB_MODE:-online}
MAX_GRADIENT_NORM=${MAX_GRADIENT_NORM:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
MINIMUM_FINAL_TOP1=${MINIMUM_FINAL_TOP1:-}
NO_DECAY_GT_GENERATORS=${NO_DECAY_GT_GENERATORS:-0}
GT_GENERATOR_WEIGHT_DECAY=${GT_GENERATOR_WEIGHT_DECAY:-}

RUN_NAME=deit_b16_${ATTENTION_TYPE}_seed${SEED}
if [[ "${ATTENTION_TYPE}" == "gt_mha_"* ]]; then
  RUN_NAME=${RUN_NAME}_qkb${NUM_BASE_HEADS}vb${NUM_VALUE_BASE_HEADS}g${NUM_GENERATORS}
fi
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_NAME=${RUN_NAME}_${RUN_SUFFIX}
fi
OUTPUT_DIR=${RUNS_DIR}/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4
export WANDB_ENTITY=${WANDB_ENTITY:-i2rLAB}
export WANDB_PROJECT=${WANDB_PROJECT:-gt-mha-imagenet}

mkdir -p "${OUTPUT_DIR}" /work/hdd/biad/sshaik4/logs
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python -c 'import timm, webdataset'
test -f "${DATA_DIR}/manifest.json"

RESUME_ARGS=()
if [[ -f "${OUTPUT_DIR}/checkpoint_last.pt" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoint_last.pt")
fi

GRADIENT_ARGS=()
if [[ -n "${MAX_GRADIENT_NORM}" ]]; then
  GRADIENT_ARGS=(--max-gradient-norm "${MAX_GRADIENT_NORM}")
fi

SMOKE_GUARD_ARGS=()
if [[ -n "${MINIMUM_FINAL_TOP1}" ]]; then
  SMOKE_GUARD_ARGS=(--minimum-final-top1 "${MINIMUM_FINAL_TOP1}")
fi

GENERATOR_DECAY_ARGS=()
if [[ "${NO_DECAY_GT_GENERATORS}" == "1" ]]; then
  GENERATOR_DECAY_ARGS=(--no-decay-gt-generators)
elif [[ -n "${GT_GENERATOR_WEIGHT_DECAY}" ]]; then
  GENERATOR_DECAY_ARGS=(--gt-generator-weight-decay "${GT_GENERATOR_WEIGHT_DECAY}")
fi

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/train_imagenet_deit.py \
  --data-dir "${DATA_DIR}" \
  --dataset-format wds \
  --output-dir "${OUTPUT_DIR}" \
  --attention-type "${ATTENTION_TYPE}" \
  --reduced-qk-dim "${REDUCED_QK_DIM}" \
  --collaborative-qk-dim "${COLLABORATIVE_QK_DIM}" \
  --num-kv-heads "${NUM_KV_HEADS}" \
  --num-base-heads "${NUM_BASE_HEADS}" \
  --num-value-base-heads "${NUM_VALUE_BASE_HEADS}" \
  --num-generators "${NUM_GENERATORS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers 12 \
  --precision bf16 \
  --seed "${SEED}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-run-name "${RUN_NAME}" \
  "${GENERATOR_DECAY_ARGS[@]}" \
  "${GRADIENT_ARGS[@]}" \
  "${SMOKE_GUARD_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
