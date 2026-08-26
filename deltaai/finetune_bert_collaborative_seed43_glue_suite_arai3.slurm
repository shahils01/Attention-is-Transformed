#!/usr/bin/env bash
# Four-epoch GLUE fine-tuning suite using the HF-restored Collaborative-MHA checkpoint.
# Each task independently reloads the same pretrained checkpoint.
#SBATCH --job-name=collab43-glue4
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bifg/arai3/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR=/work/hdd/bifg/arai3
REPO_DIR=/u/arai3/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
HF_CHECKPOINT_ROOT="${PROJECT_DIR}/lgma_runs/hf_collaborative_mha"
CHECKPOINT_DIR="${HF_CHECKPOINT_ROOT}/BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000"
SEED=44
EPOCHS=4
WANDB_PROJECT=bert-collaborative-seed43-glue-4epoch
TASKS=(mnli qnli sst2 cola)

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000/*" \
  --local-dir "${HF_CHECKPOINT_ROOT}"
export OMP_NUM_THREADS=4 WANDB_PROJECT
export WANDB_RUN_GROUP=collaborative-seed43-glue-4epoch-suite WANDB_JOB_TYPE=glue-finetune WANDB_RESUME=never
export HF_HOME="${PROJECT_DIR}/lgma_data/.hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for task in "${TASKS[@]}"; do
  run_name=bert_base_collaborative_h12_random_seed43_glue_${task}_ftseed${SEED}_${EPOCHS}ep_v5
  output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
  test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
  export WANDB_RUN_ID=collab43_${task}_s${SEED}_${EPOCHS}ep_v5 WANDB_NAME=${run_name}
  torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
    --model-name-or-path "${CHECKPOINT_DIR}" --task "${task}" \
    --attention-type collaborative --num-base-heads 4 --num-generators 8 \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${EPOCHS}" \
    --seed "${SEED}" --bf16 --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
done
