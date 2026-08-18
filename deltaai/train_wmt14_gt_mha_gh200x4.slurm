#!/usr/bin/env bash
# WMT14 EN->DE paper GT-MHA: C=4, quadratic score/value transforms,
# p=p_V=8, and full self- and cross-attention replacement.
# 54.85M parameters at maximum source/target length 256.
#SBATCH --job-name=wmt14-gt-mha-full
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/work/hdd/bicn/anayak2/repos/Attention-is-Transformed
PROJECT_DIR=/work/hdd/bicn/anayak2
DATA_DIR=${PROJECT_DIR}/lgma_data/wmt14_en_de/processed_spm32k
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/wmt14_paper_gt_mha_quadratic_full_attention_gh200x4
VENV_DIR=${PROJECT_DIR}/venvs/lgma

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=8

mkdir -p "${OUTPUT_DIR}/logs" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=4 experiments/train_wmt14.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --attention gt_mha_quadratic \
  --cross-attention gt_mha_quadratic \
  --enforce-paper-gt-mha \
  --d-model 512 \
  --ffn-dim 2048 \
  --num-encoder-layers 6 \
  --num-decoder-layers 6 \
  --num-heads 8 \
  --num-base-heads 4 \
  --num-generators 8 \
  --value-transform lie \
  --metric-beta 1.0 \
  --value-beta 1.0 \
  --generator-mixing softmax \
  --no-stabilize-generators \
  --fuse-base-qkv \
  --fold-value-transform-into-output \
  --sdpa-gqa-mode auto \
  --head-dim 64 \
  --max-source-length 256 \
  --max-target-length 256 \
  --share-all-embeddings \
  --dropout 0.1 \
  --tokens-per-batch 4096 \
  --grad-accum-steps 2 \
  --steps 100000 \
  --lr 5e-4 \
  --lr-schedule inverse_sqrt \
  --warmup-steps 4000 \
  --label-smoothing 0.1 \
  --weight-decay 0.0 \
  --precision bf16 \
  --log-every 100 \
  --eval-every 1000 \
  --eval-batches 32 \
  --save-every 5000 \
  --wandb-project wmt14 \
  --wandb-entity i2rLAB \
  --wandb-run-name wmt14-paper-gt-mha-quadratic-full-attention-gh200x4 \
  --wandb-group wmt14 \
  --wandb-tags wmt14,gt-mha,paper-config,quadratic,c4,p8,value-quadratic,transformer-base,55m,full-attention,gh200,ddp4,tokens4096,gb32768 \
  --wandb-mode online
