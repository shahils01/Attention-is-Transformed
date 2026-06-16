#!/usr/bin/env bash
set -euo pipefail

TRAIN_METAS_PATH="${TRAIN_METAS_PATH:-/scratch/shahils/openpi/datasets/Libero-XVLA-format/libero_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/shahils/lgma_runs/libero_vla_lgma_residual_multibase_b2_v1}"
WANDB_PROJECT="${WANDB_PROJECT:-lgma_vla}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-libero_vla_lgma_residual_multibase_b2_v1}"
WANDB_GROUP="${WANDB_GROUP:-libero}"
WANDB_TAGS="${WANDB_TAGS:-libero,vla,lgma_residual,multibase,b2}"
WANDB_MODE="${WANDB_MODE:-online}"
VALUE_TRANSFORM="${VALUE_TRANSFORM:-none}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if (( NPROC_PER_NODE > 1 )); then
  DEVICE="${DEVICE:-cuda}"
  LAUNCHER=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" experiments/train_vla.py)
else
  DEVICE="${DEVICE:-cuda:0}"
  LAUNCHER=(python experiments/train_vla.py)
fi

"${LAUNCHER[@]}" \
  --train_metas_path "${TRAIN_METAS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_tags "${WANDB_TAGS}" \
  --wandb_mode "${WANDB_MODE}" \
  --attention lgma_residual \
  --num_base_heads 2 \
  --num_generators 16 \
  --generator_type full \
  --value_transform "${VALUE_TRANSFORM}" \
  --device "${DEVICE}" \
  --batch_size 128 \
  --num_workers 4 \
  --steps 500000 \
  --log_every 100 \
  --eval_every 1000 \
  --eval_batches 20 \
  --save_every 5000 \
  --val_fraction 0.05 \
  --image_size 128 \
  --num_views 2 \
  --text_length 32 \
  --action_horizon 10 \
  --d_model 1024 \
  --num_layers 12 \
  --num_heads 16 \
  --head_dim 64 \
  --base_dim 64 \
  --value_dim 64 \
  --dropout 0.1 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --precision bf16
