#!/usr/bin/env bash
# Four-epoch GLUE fine-tuning suite for the selected GT-MHA residual checkpoint.
# Each task independently reloads the same pretrained checkpoint.
#SBATCH --job-name=gtres44-glue4
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR=/work/hdd/biad/sshaik4
REPO_DIR=/u/sshaik4/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
CHECKPOINT_DIR=${PROJECT_DIR}/lgma_runs/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000
SEED=42
EPOCHS=4
WANDB_PROJECT=bert-gt-mha-residual-seed44-glue-4epoch
TASKS=(mnli qnli sst2 cola)

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4 WANDB_PROJECT
export WANDB_RUN_GROUP=gt-mha-residual-seed44-glue-4epoch-suite WANDB_JOB_TYPE=glue-finetune WANDB_RESUME=never
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache HF_HUB_CACHE=${HF_HOME}/hub HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for task in "${TASKS[@]}"; do
  run_name=bert_base_gt_mha_residual_b4g8h12_random_seed44_glue_${task}_ftseed${SEED}_${EPOCHS}ep_v1
  output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
  test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
  export WANDB_RUN_ID=gtres44_${task}_s${SEED}_${EPOCHS}ep_v1 WANDB_NAME=${run_name}
  torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
    --model-name-or-path "${CHECKPOINT_DIR}" --task "${task}" \
    --attention-type gt_mha_residual --num-base-heads 4 --num-generators 8 --fuse-base-qkv \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${EPOCHS}" \
    --seed "${SEED}" --bf16 --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
done
