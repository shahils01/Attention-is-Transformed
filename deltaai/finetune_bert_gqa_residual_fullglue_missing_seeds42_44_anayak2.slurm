#!/usr/bin/env bash
# Complete the missing GLUE tasks for GQA and GT-MHA residual, seeds 42-44.
# All 30 fine-tuning runs execute sequentially in one four-GH200 allocation.
# Existing MNLI/QNLI/SST-2/CoLA runs are intentionally not repeated.
#SBATCH --job-name=gqa-gtres-glue9
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bicn/anayak2
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
GQA_HF_ROOT="${PROJECT_DIR}/lgma_runs/hf_gqa_seed42"
GQA_CHECKPOINT="${GQA_HF_ROOT}/BERT-Base Checkpoints/gqa/bert_base_gqa_kv4_random_seed42_v1/checkpoint-100000"
RESIDUAL_HF_ROOT="${PROJECT_DIR}/lgma_runs/hf_gt_mha_residual"
RESIDUAL_CHECKPOINT="${RESIDUAL_HF_ROOT}/BERT-Base Checkpoints/gt_mha_residual/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000"
EPOCHS=4
SEEDS=(42 43 44)
TASKS=(mrpc stsb qqp rte wnli)

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"

hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/gqa/bert_base_gqa_kv4_random_seed42_v1/checkpoint-100000/*" \
  --local-dir "${GQA_HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/gt_mha_residual/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000/*" \
  --local-dir "${RESIDUAL_HF_ROOT}"

export OMP_NUM_THREADS=4
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=never
export HF_HOME="${PROJECT_DIR}/lgma_data/.hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# datasets>=4 resolves GLUE from nyu-mll/glue. The training script retains the
# legacy `glue` name, so expose each canonical cache entry under that name.
mkdir -p "${HF_DATASETS_CACHE}/glue"
for task in "${TASKS[@]}"; do
  legacy="${HF_DATASETS_CACHE}/glue/${task}"
  canonical="${HF_DATASETS_CACHE}/nyu-mll___glue/${task}"
  test -e "${canonical}" || { echo "Missing cached GLUE/${task}: ${canonical}" >&2; exit 1; }
  test -e "${legacy}" || ln -s "../nyu-mll___glue/${task}" "${legacy}"
done

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for seed in "${SEEDS[@]}"; do
  export WANDB_PROJECT=bert-gqa-kv4-seed42-glue-4epoch
  export WANDB_RUN_GROUP=gqa-kv4-seed42-glue-full-suite
  for task in "${TASKS[@]}"; do
    run_name=bert_base_gqa_kv4_random_seed42_v1_glue_${task}_ftseed${seed}_${EPOCHS}ep_fullsuite_v1
    output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
    test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    export WANDB_RUN_ID=gqa4s42_${task}_s${seed}_full_v1
    export WANDB_NAME=${run_name}
    echo "Starting GQA GLUE/${task}, seed ${seed}, at $(date -Is)"
    torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
      --model-name-or-path "${GQA_CHECKPOINT}" --task "${task}" \
      --attention-type gqa --num-kv-heads 4 \
      --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
      --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
      --num-train-epochs "${EPOCHS}" --seed "${seed}" --bf16 \
      --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
  done

  export WANDB_PROJECT=bert-gt-mha-residual-seed44-glue-4epoch-anayak2
  export WANDB_RUN_GROUP=gt-mha-residual-seed44-glue-full-suite
  for task in "${TASKS[@]}"; do
    run_name=bert_base_gt_mha_residual_b4g8h12_random_seed44_glue_${task}_ftseed${seed}_${EPOCHS}ep_fullsuite_v1
    output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
    test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    export WANDB_RUN_ID=gtres44_${task}_s${seed}_full_v1
    export WANDB_NAME=${run_name}
    echo "Starting GT-MHA residual GLUE/${task}, seed ${seed}, at $(date -Is)"
    torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
      --model-name-or-path "${RESIDUAL_CHECKPOINT}" --task "${task}" \
      --attention-type gt_mha_residual --num-base-heads 4 --num-generators 8 --fuse-base-qkv \
      --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
      --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
      --num-train-epochs "${EPOCHS}" --seed "${seed}" --bf16 \
      --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
  done
done

echo "Completed missing GLUE tasks for GQA and GT-MHA residual at $(date -Is)"
