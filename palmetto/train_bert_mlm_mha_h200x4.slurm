#!/usr/bin/env bash
# BERT-base MHA MLM run from random initialization on four Palmetto H200 GPUs.
# Model architecture and comparison hyperparameters match the DeltaAI GT-MHA run.
#SBATCH --job-name=bert-mha
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/shahils/lgma_runs/bert_base_mha_random_seed42_v1/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/lgma_runs/bert_base_mha_random_seed42_v1/logs/%x_%j.err

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -euo pipefail

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
OUTPUT_DIR=/scratch/shahils/lgma_runs/bert_base_mha_random_seed42_v1
CACHE_DIR=/scratch/shahils/hf_cache
RUN_NAME=bert_base_mha_random_seed42_v1

mkdir -p "${OUTPUT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_ENTITY=i2rLAB
export WANDB_PROJECT=bert-gt-mha
export WANDB_RUN_ID="${RUN_NAME}"
export WANDB_RESUME=allow

resume_args=()
latest_checkpoint=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
  | sort -t- -k2,2n | tail -1 || true)
if [[ -n "${latest_checkpoint}" ]]; then
  resume_args=(--resume-from-checkpoint "${latest_checkpoint}")
fi

torchrun --standalone --nproc_per_node=4 experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --attention-type mha \
  --initialization random \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --max-sequence-length 128 \
  --mlm-probability 0.15 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --warmup-steps 10000 \
  --max-steps 100000 \
  --logging-steps 100 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --seed 42 \
  --num-workers 4 \
  --bf16 \
  --gradient-checkpointing \
  --output-dir "${OUTPUT_DIR}" \
  --report-to wandb \
  --run-name "${RUN_NAME}" \
  "${resume_args[@]}"
