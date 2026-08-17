#!/usr/bin/env bash
# WMT14 EN→DE Transformer-base MHA baseline: 60.76M parameters at length 256.
#SBATCH --job-name=wmt14-mha-base
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/shahils/lgma_runs/wmt14_mha_base_h200x4/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/lgma_runs/wmt14_mha_base_h200x4/logs/%x_%j.err

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -euo pipefail

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
DATA_DIR=/scratch/shahils/lgma_data/wmt14_en_de/processed_spm32k
OUTPUT_DIR=/scratch/shahils/lgma_runs/wmt14_mha_base_h200x4
mkdir -p "${OUTPUT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=4 experiments/train_wmt14.py \
  --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" \
  --attention mha --d-model 512 --ffn-dim 2048 \
  --num-encoder-layers 6 --num-decoder-layers 6 --num-heads 8 --head-dim 64 \
  --max-source-length 256 --max-target-length 256 --share-all-embeddings --dropout 0.1 \
  --tokens-per-batch 4096 --grad-accum-steps 2 --steps 100000 \
  --lr 5e-4 --lr-schedule inverse_sqrt --warmup-steps 4000 --label-smoothing 0.1 \
  --weight-decay 0.01 --precision bf16 \
  --log-every 100 --eval-every 1000 --eval-batches 32 --save-every 5000 \
  --wandb-project wmt14 --wandb-entity i2rLAB \
  --wandb-run-name wmt14-mha-base-h200x4 --wandb-group wmt14 \
  --wandb-tags wmt14,mha,transformer-base,60m,h200,ddp4,tokens4096,gb32768 \
  --wandb-mode online
