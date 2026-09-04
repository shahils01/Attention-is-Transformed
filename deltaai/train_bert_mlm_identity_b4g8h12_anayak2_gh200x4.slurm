#!/usr/bin/env bash
# Three-seed BERT GT-MHA scoring-identity ablation. Each Slurm array task
# trains one seed on one four-GH200 node; all runs share one W&B group.
# The value pathway remains residual so only M_h is fixed to I.
#SBATCH --job-name=bert-id-b4g8
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --array=42-44
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bicn/anayak2
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed-bert-identity
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME=bert_base_gt_mha_identity_qk_b4g8h12_random_seed${SEED}
WANDB_GROUP=bert-ablation-identity-qk-b4g8h12
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4
export WANDB_PROJECT=bert-gt-mha
export WANDB_ENTITY=i2rLAB
export WANDB_RUN_GROUP=${WANDB_GROUP}
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

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
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type gt_mha_identity \
  --num-base-heads 4 \
  --num-generators 8 \
  --generator-mixing softmax \
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
  --max-steps 100000 \
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
