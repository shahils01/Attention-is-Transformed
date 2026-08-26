#!/usr/bin/env bash
# Fine-tune MHA and Collaborative MHA on GLUE for seeds 45 and 46.
# All suites run sequentially within one four-GH200 allocation.
#SBATCH --job-name=mha-collab-s45s46
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
MHA_HF_ROOT="${PROJECT_DIR}/lgma_runs/hf_mha_seed44"
MHA_CHECKPOINT="${MHA_HF_ROOT}/BERT-Base Checkpoints/mha/bert_base_mha_random_seed44_fixedval_v2/checkpoint-100000"
COLLAB_HF_ROOT="${PROJECT_DIR}/lgma_runs/hf_collaborative_mha"
COLLAB_CHECKPOINT="${COLLAB_HF_ROOT}/BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000"
EPOCHS=4
SEEDS=(45 46)
TASKS=(mnli qnli sst2 cola)

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
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for seed in "${SEEDS[@]}"; do
  export WANDB_PROJECT=bert-mha-seed44-glue-4epoch
  export WANDB_RUN_GROUP=mha-seed44-glue-4epoch-suite
  for task in "${TASKS[@]}"; do
    run_name=bert_base_mha_random_seed44_fixedval_v2_glue_${task}_ftseed${seed}_${EPOCHS}ep_v6
    output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
    test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    export WANDB_RUN_ID=mha44_${task}_s${seed}_${EPOCHS}ep_v6
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
  export WANDB_RUN_GROUP=collaborative-seed43-glue-4epoch-suite
  for task in "${TASKS[@]}"; do
    run_name=bert_base_collaborative_h12_random_seed43_glue_${task}_ftseed${seed}_${EPOCHS}ep_v6
    output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
    test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; exit 1; }
    export WANDB_RUN_ID=collab43_${task}_s${seed}_${EPOCHS}ep_v6
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

echo "Completed MHA and Collaborative MHA seeds 45 and 46 at $(date -Is)"
