#!/usr/bin/env bash
# Complete the missing GLUE tasks for MHA and Collaborative MHA, seeds 42-44.
# All 30 fine-tuning runs execute sequentially in one four-GH200 allocation.
# Existing MNLI/QNLI/SST-2/CoLA runs are intentionally not repeated.
#SBATCH --job-name=mha-collab-glue9
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bifg/arai3/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bifg/arai3
REPO_DIR=/u/arai3/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
MHA_HF_ROOT="${PROJECT_DIR}/lgma_runs/hf_mha_seed44"
MHA_CHECKPOINT="${MHA_HF_ROOT}/BERT-Base Checkpoints/mha/bert_base_mha_random_seed44_fixedval_v2/checkpoint-100000"
COLLAB_HF_ROOT="${PROJECT_DIR}/lgma_runs/hf_collaborative_mha"
COLLAB_CHECKPOINT="${COLLAB_HF_ROOT}/BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000"
EPOCHS=4
SEEDS=(42 43 44)
TASKS=(mrpc stsb qqp rte wnli)

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"

hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/mha/bert_base_mha_random_seed44_fixedval_v2/checkpoint-100000/*" \
  --local-dir "${MHA_HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000/*" \
  --local-dir "${COLLAB_HF_ROOT}"

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
  export WANDB_PROJECT=bert-mha-seed44-glue-4epoch
  export WANDB_RUN_GROUP=mha-seed44-glue-full-suite
  for task in "${TASKS[@]}"; do
    run_name=bert_base_mha_random_seed44_fixedval_v2_glue_${task}_ftseed${seed}_${EPOCHS}ep_fullsuite_v1
    output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
    test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    export WANDB_RUN_ID=mha44_${task}_s${seed}_full_v1
    export WANDB_NAME=${run_name}
    echo "Starting MHA GLUE/${task}, seed ${seed}, at $(date -Is)"
    torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
      --model-name-or-path "${MHA_CHECKPOINT}" --task "${task}" \
      --attention-type mha \
      --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
      --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
      --num-train-epochs "${EPOCHS}" --seed "${seed}" --bf16 \
      --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
  done

  export WANDB_PROJECT=bert-collaborative-seed43-glue-4epoch
  export WANDB_RUN_GROUP=collaborative-seed43-glue-full-suite
  for task in "${TASKS[@]}"; do
    run_name=bert_base_collaborative_h12_random_seed43_glue_${task}_ftseed${seed}_${EPOCHS}ep_fullsuite_v1
    output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
    test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    export WANDB_RUN_ID=collab43_${task}_s${seed}_full_v1
    export WANDB_NAME=${run_name}
    echo "Starting Collaborative MHA GLUE/${task}, seed ${seed}, at $(date -Is)"
    torchrun --standalone --nproc_per_node=4 experiments/finetune_bert_glue.py \
      --model-name-or-path "${COLLAB_CHECKPOINT}" --task "${task}" \
      --attention-type collaborative --num-base-heads 4 --num-generators 8 \
      --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
      --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
      --num-train-epochs "${EPOCHS}" --seed "${seed}" --bf16 \
      --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
  done
done

echo "Completed missing GLUE tasks for MHA and Collaborative MHA at $(date -Is)"
