#!/usr/bin/env bash
# End-to-end WMT14 EN→DE smoke test using the standard MHA baseline.
#SBATCH --job-name=wmt14-mha-smoke
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shahils/lgma_runs/wmt14_mha_smoke_tied/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/lgma_runs/wmt14_mha_smoke_tied/logs/%x_%j.err

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -euo pipefail

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
DATA_DIR=/scratch/shahils/lgma_data/wmt14_en_de/processed_spm32k
OUTPUT_DIR=/scratch/shahils/lgma_runs/wmt14_mha_smoke_tied
mkdir -p "${OUTPUT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=1 experiments/train_wmt14.py \
  --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" \
  --attention mha --d-model 512 --ffn-dim 2048 \
  --num-encoder-layers 6 --num-decoder-layers 6 --num-heads 8 --head-dim 64 \
  --max-source-length 128 --max-target-length 128 --share-all-embeddings --dropout 0.1 \
  --batch-size 8 --tokens-per-batch 512 --grad-accum-steps 1 --steps 20 --lr 5e-4 --precision bf16 \
  --log-every 1 --eval-every 10 --eval-batches 2 --save-every 20 \
  --wandb-project wmt14 --wandb-entity i2rLAB \
  --wandb-run-name wmt14-mha-smoke-tied --wandb-group wmt14 --wandb-tags wmt14,mha,smoke,tied-embeddings,60m \
  --wandb-mode online
