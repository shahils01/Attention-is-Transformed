#!/usr/bin/env bash
# Controlled DeiT-B/16 ImageNet-1K training on four Palmetto H200 GPUs.
# Use MAX_TRAIN_BATCHES and MAX_VAL_BATCHES for a bounded smoke test.
#SBATCH --job-name=deit-attn
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/shahils/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/logs/%x_%j.err

source /etc/profile
set -euo pipefail

REPO_DIR=${REPO_DIR:-/scratch/shahils/Attention-is-Transformed-vision}
ENV_DIR=${ENV_DIR:-/scratch/shahils/conda_envs/lgma-vision}
DATA_DIR=${DATA_DIR:-/scratch/shahils/imagenet_wds_official}
RUNS_DIR=${RUNS_DIR:-/scratch/shahils/lgma_runs/imagenet}
ATTENTION_TYPE=${ATTENTION_TYPE:-gt_mha_residual}
NUM_BASE_HEADS=${NUM_BASE_HEADS:-4}
NUM_VALUE_BASE_HEADS=${NUM_VALUE_BASE_HEADS:-${NUM_BASE_HEADS}}
NUM_GENERATORS=${NUM_GENERATORS:-8}
GENERATOR_MIXING=${GENERATOR_MIXING:-softmax}
RAW_MIXING_INIT=${RAW_MIXING_INIT:-direct}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
REDUCED_QK_DIM=${REDUCED_QK_DIM:-384}
COLLABORATIVE_QK_DIM=${COLLABORATIVE_QK_DIM:-384}
THETA_INIT=${THETA_INIT:-balanced_simplex}
THETA_INIT_SCALE=${THETA_INIT_SCALE:-4.0}
GENERATOR_INIT_SCALE=${GENERATOR_INIT_SCALE:-0.02}
GT_GENERATOR_WEIGHT_DECAY=${GT_GENERATOR_WEIGHT_DECAY:-}
VALUE_DIVERSITY_WEIGHT=${VALUE_DIVERSITY_WEIGHT:-0}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_GPUS=${NUM_GPUS:-4}
WORKERS=${WORKERS:-12}
WANDB_MODE=${WANDB_MODE:-online}
RUN_SUFFIX=${RUN_SUFFIX:-}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-}

RUN_NAME=deit_b16_${ATTENTION_TYPE}_seed${SEED}
if [[ "${ATTENTION_TYPE}" == "gt_mha_"* ]]; then
  RUN_NAME=${RUN_NAME}_qkb${NUM_BASE_HEADS}vb${NUM_VALUE_BASE_HEADS}g${NUM_GENERATORS}
fi
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_NAME=${RUN_NAME}_${RUN_SUFFIX}
fi
OUTPUT_DIR=${RUNS_DIR}/${RUN_NAME}

module load anaconda3/2023.09-0
source "${ANACONDA3_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export WANDB_ENTITY=${WANDB_ENTITY:-i2rLAB}
export WANDB_PROJECT=${WANDB_PROJECT:-gt-mha-imagenet}

mkdir -p "${OUTPUT_DIR}" /scratch/shahils/logs
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python -c 'import timm, torch, torchvision, webdataset'
test -f "${DATA_DIR}/manifest.json"

EXTRA_ARGS=()
if [[ -n "${MAX_TRAIN_BATCHES}" ]]; then
  EXTRA_ARGS+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if [[ -n "${MAX_VAL_BATCHES}" ]]; then
  EXTRA_ARGS+=(--max-val-batches "${MAX_VAL_BATCHES}")
fi

GENERATOR_DECAY_ARGS=()
if [[ -n "${GT_GENERATOR_WEIGHT_DECAY}" ]]; then
  GENERATOR_DECAY_ARGS=(--gt-generator-weight-decay "${GT_GENERATOR_WEIGHT_DECAY}")
fi

RESUME_ARGS=()
if [[ -f "${OUTPUT_DIR}/checkpoint_last.pt" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoint_last.pt")
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
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
  --generator-mixing "${GENERATOR_MIXING}" \
  --raw-mixing-init "${RAW_MIXING_INIT}" \
  --theta-init "${THETA_INIT}" \
  --theta-init-scale "${THETA_INIT_SCALE}" \
  --generator-init-scale "${GENERATOR_INIT_SCALE}" \
  --value-diversity-weight "${VALUE_DIVERSITY_WEIGHT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --precision bf16 \
  --seed "${SEED}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-run-name "${RUN_NAME}" \
  "${GENERATOR_DECAY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
