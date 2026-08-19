#!/usr/bin/env bash
# BERT-base residual GT-MHA MLM training from random initialization on Anvil.
# Matches the controlled WikiText-103, sequence-length-128 comparison protocol.
#SBATCH --job-name=bert-gtres
#SBATCH --account=cis261553-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/anvil/projects/x-cis261553/logs/%x_%j.out
#SBATCH --error=/anvil/projects/x-cis261553/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/anvil/projects/x-cis261553
REPO_DIR=/home/x-arai3/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-bert
RUN_NAME=bert_base_gt_mha_residual_random_seed42_anvil_h100x4_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module purge >/dev/null 2>&1 || true
module load anaconda/2024.02-py311
source activate "${VENV_DIR}"

export OMP_NUM_THREADS=2
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=i2rLAB
export WANDB_PROJECT=bert-gt-mha
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${OUTPUT_DIR}

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"

resume_args=()
latest_checkpoint=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
  | sort -t- -k2,2n | tail -1 || true)
if [[ -n "${latest_checkpoint}" ]]; then
  resume_args=(--resume-from-checkpoint "${latest_checkpoint}")
fi

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --attention-type gt_mha_residual \
  --initialization random \
  --num-base-heads 4 \
  --num-generators 8 \
  --enforce-paper-gt-mha \
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
