#!/usr/bin/env bash
# Fair STS-B paper sweep for all attention types in one allocation.
# Stage 1: 4 methods x 3 learning rates x seeds 42-44 = 36 runs.
# Stage 2: select one LR per method by Stage-1 three-seed mean, then run seeds 45-46 = 8 runs.
# Four independent one-GPU processes run concurrently. Gradient accumulation preserves the
# original effective batch size of 128 from the four-GPU GLUE experiments.
#SBATCH --job-name=stsb-paper-sweep
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bicn/anayak2
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
JOB_DIR=${PROJECT_DIR}/lgma_jobs
TRAIN_SCRIPT=${JOB_DIR}/finetune_bert_glue_stsb_sweep.py
SELECT_SCRIPT=${JOB_DIR}/select_stsb_learning_rates.py
RUN_PREFIX=stsb_papersweep_v2
EPOCHS=8
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
export WANDB_PROJECT=bert-gt-mha-residual-stsb-sweep
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=never
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_DATASETS_CACHE}/glue" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/lgma_runs"
legacy=${HF_DATASETS_CACHE}/glue/stsb
canonical=${HF_DATASETS_CACHE}/nyu-mll___glue/stsb
test -e "${canonical}" || { echo "Missing cached GLUE/STS-B: ${canonical}" >&2; exit 1; }
test -e "${legacy}" || ln -s ../nyu-mll___glue/stsb "${legacy}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "${REPO_DIR}"
export PYTHONPATH=${REPO_DIR}/src:${PYTHONPATH:-}

lr_tag() {
  printf '%s' "${1//e-/em}"
}

run_one() {
  local gpu=$1 method=$2 seed=$3 learning_rate=$4 stage=$5
  local tag
  tag=$(lr_tag "${learning_rate}")
  local run_name=${RUN_PREFIX}_${method}_lr${tag}_seed${seed}_${stage}
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
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac

  echo "Starting ${run_name} on GPU ${gpu} at $(date -Is)"
  CUDA_VISIBLE_DEVICES=${gpu} \
  WANDB_RUN_ID=${RUN_PREFIX}_${method}_${tag}_s${seed}_${stage} \
  WANDB_NAME=${run_name} \
  WANDB_RUN_GROUP=${method} \
  WANDB_TAGS=paper-stsb-sweep-v2,attention-${method},lr-${tag},stage-${stage} \
  python "${TRAIN_SCRIPT}" \
    --model-name-or-path "${checkpoint}" --task stsb \
    "${attention_args[@]}" \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 4 --learning-rate "${learning_rate}" \
    --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${EPOCHS}" \
    --metric-for-best-model combined_score --greater-is-better \
    --seed "${seed}" --bf16 --num-workers 2 \
    --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
}

run_specs() {
  local -a specs=("$@") pids=() labels=()
  local gpu=0 failed=0 spec method seed learning_rate stage
  for spec in "${specs[@]}"; do
    IFS='|' read -r method seed learning_rate stage <<<"${spec}"
    run_one "${gpu}" "${method}" "${seed}" "${learning_rate}" "${stage}" &
    pids+=("$!")
    labels+=("${spec}")
    gpu=$((gpu + 1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "Failed: ${labels[$i]}" >&2; failed=1; }
  done
  (( failed == 0 ))
}

queue=()
for method in "${METHODS[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for seed in "${GRID_SEEDS[@]}"; do
      queue+=("${method}|${seed}|${learning_rate}|grid")
      if (( ${#queue[@]} == 4 )); then
        run_specs "${queue[@]}"
        queue=()
      fi
    done
  done
done
(( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"

SELECTION_JSON=${PROJECT_DIR}/lgma_runs/${RUN_PREFIX}_selected_learning_rates.json
SELECTION_TSV=${PROJECT_DIR}/lgma_runs/${RUN_PREFIX}_selected_learning_rates.tsv
python "${SELECT_SCRIPT}" \
  --root "${PROJECT_DIR}/lgma_runs" --prefix "${RUN_PREFIX}" \
  --output-json "${SELECTION_JSON}" --output-tsv "${SELECTION_TSV}"

queue=()
while IFS=$'\t' read -r method learning_rate mean_score; do
  echo "Selected ${method}: lr=${learning_rate}, grid mean=${mean_score}"
  for seed in "${FINAL_SEEDS[@]}"; do
    queue+=("${method}|${seed}|${learning_rate}|final")
    if (( ${#queue[@]} == 4 )); then
      run_specs "${queue[@]}"
      queue=()
    fi
  done
done < "${SELECTION_TSV}"
(( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"

echo "Completed adaptive all-attention STS-B sweep at $(date -Is)"
