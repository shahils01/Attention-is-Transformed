#!/usr/bin/env bash
# Equal-budget adaptive sweeps for high-variance GLUE tasks: CoLA, MRPC, and RTE.
# Per task, Stage 1 runs 4 methods x 3 LRs x seeds 42-44 (36 runs).
# Stage 2 selects one LR per method by mean task metric and runs seeds 45-46 (8 runs).
# Four independent one-GPU processes run concurrently with effective batch size 128.
#SBATCH --job-name=glue-hivar-sweep
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
JOB_DIR=${PROJECT_DIR}/lgma_jobs
TRAIN_SCRIPT=${JOB_DIR}/finetune_bert_glue_hivar_sweep.py
SELECT_SCRIPT=${JOB_DIR}/select_glue_learning_rates.py
CACHE_SCRIPT=${JOB_DIR}/cache_glue_tasks.py
RUN_VERSION=glue_hivar_papersweep_v1
EPOCHS=8
read -r -a TASKS <<< "${TASK_LIST:-cola mrpc rte}"
GRID_SEEDS=(42 43 44)
FINAL_SEEDS=(45 46)
LEARNING_RATES=(2e-5 3e-5 5e-5)
METHODS=(mha gqa collaborative gt_mha_residual)

MHA_HF_ROOT=${PROJECT_DIR}/lgma_runs/hf_mha_seed44
MHA_CHECKPOINT="${MHA_HF_ROOT}/BERT-Base Checkpoints/mha/bert_base_mha_random_seed44_fixedval_v2/checkpoint-100000"
GQA_HF_ROOT=${PROJECT_DIR}/lgma_runs/hf_gqa_seed42
GQA_CHECKPOINT="${GQA_HF_ROOT}/BERT-Base Checkpoints/gqa/bert_base_gqa_kv4_random_seed42_v1/checkpoint-100000"
COLLAB_HF_ROOT=${PROJECT_DIR}/lgma_runs/hf_collaborative_mha
COLLAB_CHECKPOINT="${COLLAB_HF_ROOT}/BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000"
GT_HF_ROOT=${PROJECT_DIR}/lgma_runs/hf_gt_mha_residual
GT_CHECKPOINT="${GT_HF_ROOT}/BERT-Base Checkpoints/gt_mha_residual/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000"

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
test -f "${TRAIN_SCRIPT}" || { echo "Missing trainer: ${TRAIN_SCRIPT}" >&2; exit 1; }
test -f "${SELECT_SCRIPT}" || { echo "Missing selector: ${SELECT_SCRIPT}" >&2; exit 1; }
test -f "${CACHE_SCRIPT}" || { echo "Missing cache utility: ${CACHE_SCRIPT}" >&2; exit 1; }

hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/mha/bert_base_mha_random_seed44_fixedval_v2/checkpoint-100000/*" \
  --local-dir "${MHA_HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/gqa/bert_base_gqa_kv4_random_seed42_v1/checkpoint-100000/*" \
  --local-dir "${GQA_HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000/*" \
  --local-dir "${COLLAB_HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/gt_mha_residual/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000/*" \
  --local-dir "${GT_HF_ROOT}"

export OMP_NUM_THREADS=4
export WANDB_PROJECT=bert-glue-high-variance-sweep
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=never
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/lgma_runs"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python "${CACHE_SCRIPT}" --tasks "${TASKS[@]}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "${REPO_DIR}"
export PYTHONPATH=${REPO_DIR}/src:${PYTHONPATH:-}

lr_tag() {
  printf '%s' "${1//e-/em}"
}

task_metric() {
  case "$1" in
    cola) printf '%s' matthews_correlation ;;
    mrpc) printf '%s' combined_score ;;
    rte) printf '%s' accuracy ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local gpu=$1 task=$2 method=$3 seed=$4 learning_rate=$5 stage=$6
  local tag metric_name
  tag=$(lr_tag "${learning_rate}")
  metric_name=$(task_metric "${task}")
  local prefix=${RUN_VERSION}_${task}
  local run_name=${prefix}_${method}_lr${tag}_seed${seed}_${stage}
  local output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
  if [[ -f "${output_dir}/eval_results.json" ]]; then
    echo "Skipping completed run ${run_name}"
    return 0
  fi
  test ! -e "${output_dir}" || { echo "Refusing to overwrite incomplete ${output_dir}" >&2; return 1; }

  local checkpoint attention_args
  case "${method}" in
    mha)
      checkpoint=${MHA_CHECKPOINT}
      attention_args=(--attention-type mha)
      ;;
    gqa)
      checkpoint=${GQA_CHECKPOINT}
      attention_args=(--attention-type gqa --num-kv-heads 4)
      ;;
    collaborative)
      checkpoint=${COLLAB_CHECKPOINT}
      attention_args=(--attention-type collaborative --num-base-heads 4 --num-generators 8)
      ;;
    gt_mha_residual)
      checkpoint=${GT_CHECKPOINT}
      attention_args=(--attention-type gt_mha_residual --num-base-heads 4 --num-generators 8 --fuse-base-qkv)
      ;;
    *) echo "Unknown method: ${method}" >&2; return 1 ;;
  esac

  echo "Starting ${run_name} on GPU ${gpu} at $(date -Is)"
  CUDA_VISIBLE_DEVICES=${gpu} \
  WANDB_RUN_ID=${RUN_VERSION}_${task}_${method}_${tag}_s${seed}_${stage}${WANDB_RUN_ID_SUFFIX:-} \
  WANDB_NAME=${run_name}${WANDB_NAME_SUFFIX:-} \
  WANDB_RUN_GROUP=${task}-${method} \
  WANDB_TAGS=paper-glue-hivar-v1,task-${task},attention-${method},lr-${tag},stage-${stage} \
  python "${TRAIN_SCRIPT}" \
    --model-name-or-path "${checkpoint}" --task "${task}" \
    "${attention_args[@]}" \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 4 --learning-rate "${learning_rate}" \
    --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${EPOCHS}" \
    --save-total-limit 1 \
    --metric-for-best-model "${metric_name}" --greater-is-better \
    --seed "${seed}" --bf16 --num-workers 2 \
    --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
}

run_specs() {
  local -a specs=("$@") pids=() labels=()
  local gpu=0 failed=0 spec task method seed learning_rate stage
  for spec in "${specs[@]}"; do
    IFS='|' read -r task method seed learning_rate stage <<<"${spec}"
    run_one "${gpu}" "${task}" "${method}" "${seed}" "${learning_rate}" "${stage}" &
    pids+=("$!")
    labels+=("${spec}")
    gpu=$((gpu + 1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "Failed: ${labels[$i]}" >&2; failed=1; }
  done
  (( failed == 0 ))
}

for task in "${TASKS[@]}"; do
  queue=()
  for method in "${METHODS[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      for seed in "${GRID_SEEDS[@]}"; do
        queue+=("${task}|${method}|${seed}|${learning_rate}|grid")
        if (( ${#queue[@]} == 4 )); then
          run_specs "${queue[@]}"
          queue=()
        fi
      done
    done
  done
  (( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"

  prefix=${RUN_VERSION}_${task}
  selection_json=${PROJECT_DIR}/lgma_runs/${prefix}_selected_learning_rates.json
  selection_tsv=${PROJECT_DIR}/lgma_runs/${prefix}_selected_learning_rates.tsv
  metric_name=$(task_metric "${task}")
  python "${SELECT_SCRIPT}" \
    --root "${PROJECT_DIR}/lgma_runs" --prefix "${prefix}" \
    --metric-key "eval_${metric_name}" \
    --output-json "${selection_json}" --output-tsv "${selection_tsv}"

  queue=()
  while IFS=$'\t' read -r method learning_rate mean_score; do
    echo "Selected ${task}/${method}: lr=${learning_rate}, grid mean=${mean_score}"
    for seed in "${FINAL_SEEDS[@]}"; do
      queue+=("${task}|${method}|${seed}|${learning_rate}|final")
      if (( ${#queue[@]} == 4 )); then
        run_specs "${queue[@]}"
        queue=()
      fi
    done
  done < "${selection_tsv}"
  (( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"
done

echo "Completed CoLA/MRPC/RTE adaptive sweeps at $(date -Is)"
