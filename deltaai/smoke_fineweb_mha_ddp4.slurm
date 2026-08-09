#!/usr/bin/env bash
#SBATCH --job-name=fw-mha-smoke
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/u/sshaik4/Attention-is-Transformed
PROJECT_DIR=/work/hdd/biad/sshaik4
PACKED_DATA=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT_packed_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/fineweb_mha_smoke_ddp4
VENV_DIR=${PROJECT_DIR}/venvs/lgma

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=8

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"

torchrun --standalone --nproc_per_node=4 experiments/train_text_lm.py \
  --packed_data_dir "${PACKED_DATA}" \
  --device cuda \
  --batch_size 64 \
  --grad_accum_steps 1 \
  --steps 20 \
  --log_every 5 \
  --eval_every 10 \
  --eval_batches 4 \
  --save_every 0 \
  --output_dir "${OUTPUT_DIR}" \
  --diagnostic_every 0 \
  --diagnostic_batches 1 \
  --wandb_mode disabled \
  --d_model 1024 \
  --num_layers 12 \
  --head_dim 64 \
  --context_length 512 \
  --dropout 0.1 \
  --lr 3e-4 \
  --lr_schedule constant \
  --weight_decay 0.01 \
  --precision bf16 \
  --attention mha \
  --num_heads 16 \
  --seed 0
