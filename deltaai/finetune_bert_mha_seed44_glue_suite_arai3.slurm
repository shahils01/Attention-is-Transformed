#!/usr/bin/env bash
# Fine-tune the completed seed-44 MHA BERT checkpoint on four GLUE tasks.
# Each task starts independently from the same MLM checkpoint while sharing one
# four-GH200 allocation. Runs are logged to a dedicated W&B project.
#SBATCH --job-name=mha-s44-glue4
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
CHECKPOINT_DIR=${PROJECT_DIR}/lgma_runs/bert_base_mha_random_seed44_fixedval_v2
CHECKPOINT_NAME=$(basename "${CHECKPOINT_DIR}")
SEED=42
EPOCHS=4
WANDB_PROJECT=bert-mha-seed44-glue-4epoch
TASKS=(mnli qnli sst2 cola)

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS=4
export WANDB_PROJECT
export WANDB_RUN_GROUP=mha-seed44-glue-4epoch-suite
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=never
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for task in "${TASKS[@]}"; do
  run_name=${CHECKPOINT_NAME}_glue_${task}_ftseed${SEED}_${EPOCHS}ep_v1
  wandb_run_id=mha44_${task}_s${SEED}_${EPOCHS}ep_v1
  output_dir=${PROJECT_DIR}/lgma_runs/${run_name}

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${output_dir}" >&2
    exit 1
  fi

  export WANDB_RUN_ID=${wandb_run_id}
  export WANDB_NAME=${run_name}

  echo "Starting GLUE/${task} from ${CHECKPOINT_DIR} at $(date -Is)"
  torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
    --model-name-or-path "${CHECKPOINT_DIR}" \
    --task "${task}" \
    --attention-type mha \
    --max-sequence-length 128 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --num-train-epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --bf16 \
    --output-dir "${output_dir}" \
    --report-to wandb \
    --run-name "${run_name}"
  echo "Completed GLUE/${task} at $(date -Is)"
done

echo "Completed all GLUE tasks at $(date -Is)"
