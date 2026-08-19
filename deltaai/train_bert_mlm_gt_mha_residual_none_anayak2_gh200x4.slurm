#!/usr/bin/env bash
# GT-MHA residual experiment matched exactly to the anayak2 MHA baseline,
# except for the attention implementation and raw (unnormalized) generator mixing.
#
# Submit with:
#   sbatch deltaai/train_bert_mlm_gt_mha_residual_none_anayak2_gh200x4.slurm
#SBATCH --job-name=bert-gt-none
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

PROJECT_DIR=/work/hdd/bicn/anayak2
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
INITIALIZATION=${INITIALIZATION:-random}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-100000}
MODEL_NAME=${MODEL_NAME:-google-bert/bert-base-uncased}
WANDB_PROJECT=${WANDB_PROJECT:-bert-gt-mha}

RUN_NAME=bert_base_gt_mha_residual_${INITIALIZATION}_seed${SEED}_mixnone_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4
export WANDB_PROJECT
export WANDB_ENTITY=${WANDB_ENTITY:-i2rLAB}
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

resume_args=()
latest_checkpoint=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
  | sort -t- -k2,2n \
  | tail -1 || true)
if [[ -n "${latest_checkpoint}" ]]; then
  resume_args=(--resume-from-checkpoint "${latest_checkpoint}")
fi

python -m torch.distributed.run --standalone --nproc_per_node=4 experiments/train_bert_mlm.py \
  --model-name-or-path "${MODEL_NAME}" \
  --initialization "${INITIALIZATION}" \
  --attention-type gt_mha_residual \
  --num-base-heads 4 \
  --num-generators 8 \
  --generator-mixing none \
  --no-enforce-paper-gt-mha \
  --use-sdpa \
  --fuse-base-qkv \
  --sdpa-gqa-mode native \
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
  --max-steps "${MAX_STEPS}" \
  --logging-steps 100 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --seed "${SEED}" \
  --bf16 \
  --gradient-checkpointing \
  --output-dir "${OUTPUT_DIR}" \
  --report-to wandb \
  --run-name "${RUN_NAME}" \
  "${resume_args[@]}"
