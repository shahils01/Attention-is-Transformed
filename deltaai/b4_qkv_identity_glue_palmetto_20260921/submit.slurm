#!/usr/bin/env bash
# One-job B4/G8/H12 GT-MHA QKV identity GLUE protocol sweep.
# Matches the archived adaptive protocol: LRs {2e-5,3e-5,5e-5}, seeds 42-44 for
# LR selection, confirmation seeds 45-46, and fixed WNLI lr=2e-5 seed42.
#SBATCH --job-name=b4-idqkv-s44-glue
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --chdir=/scratch/shahils/b4_qkv_identity_glue_palmetto_20260921/repo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --no-requeue
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/shahils/logs/%x_%j.out
#SBATCH --error=/scratch/shahils/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR=/scratch/shahils
STAGE_DIR=${PROJECT_DIR}/b4_qkv_identity_glue_palmetto_20260921
REPO_DIR=${STAGE_DIR}/repo
TRAIN_SCRIPT=${REPO_DIR}/experiments/finetune_bert_glue.py
CHECKPOINT=${STAGE_DIR}/checkpoint
METHOD=gt_mha_identity_both
NUM_BASE_HEADS=4
NUM_GENERATORS=8
GRID_SEEDS=(42 43 44)
FINAL_SEEDS=(45 46)
LEARNING_RATES=(2e-5 3e-5 5e-5)
RUN_VERSION=glue_b4_identity_qkv_preseed44_papersweep_20260921

export PYTHONNOUSERSITE=1
source "${STAGE_DIR}/venv/bin/activate"
export OMP_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export PYTHONPATH="${STAGE_DIR}/compat/no_transformer_engine:${REPO_DIR}/src"
export HF_HOME=${PROJECT_DIR}/hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=bert-gt-mha-glue
export WANDB_ENTITY=i2rLAB
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=never

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/lgma_runs"
python -u "${STAGE_DIR}/prepare.py"
python -c 'import torch; assert torch.cuda.device_count()==2; assert all("A100" in torch.cuda.get_device_name(i) and torch.cuda.get_device_properties(i).total_memory>75*1024**3 for i in range(2)); assert torch.cuda.is_bf16_supported()'
test -f "${CHECKPOINT}/model.safetensors"
test -f "${CHECKPOINT}/gt_mha_state_dict.pt"
test -f "${STAGE_DIR}/preflight.json"
python -u "${STAGE_DIR}/gpu_smoke.py"
IFS=, read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
(( ${#GPU_DEVICES[@]} == 2 ))

task_metric() {
  case "$1" in
    cola) printf "%s" matthews_correlation ;;
    mrpc|qqp|stsb) printf "%s" combined_score ;;
    mnli|qnli|rte|sst2|wnli) printf "%s" accuracy ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

task_epochs() {
  case "$1" in
    mnli|qnli|sst2|qqp|wnli) printf "%s" 4 ;;
    cola|mrpc|rte|stsb) printf "%s" 8 ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

lr_tag() { printf "%s" "${1//e-/em}"; }

run_one() {
  local gpu=$1 task=$2 seed=$3 learning_rate=$4 stage=$5
  local tag metric epochs prefix run_name output_dir accumulation=4
  [[ "$task" != wnli ]] || accumulation=1
  tag=$(lr_tag "${learning_rate}")
  metric=$(task_metric "${task}")
  epochs=$(task_epochs "${task}")
  prefix=${RUN_VERSION}_${task}
  run_name=${prefix}_${METHOD}_lr${tag}_seed${seed}_${stage}
  output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
  if [[ -f "${output_dir}/eval_results.json" ]]; then
    echo "Skipping completed ${run_name}"
    return 0
  fi
  test ! -e "${output_dir}" || { echo "Refusing to overwrite incomplete ${output_dir}" >&2; return 1; }
  echo "Starting ${run_name} on GPU ${gpu} at $(date -Is)"
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$gpu]} \
  WANDB_RUN_ID=${run_name} \
  WANDB_NAME=${run_name} \
  WANDB_RUN_GROUP=b4-idqkv-s44-glue \
  WANDB_TAGS=paper-glue-b4-idqkv-preseed44,task-${task},lr-${tag},seed-${seed},stage-${stage} \
  python "${TRAIN_SCRIPT}" \
    --model-name-or-path "${CHECKPOINT}" --task "${task}" \
    --attention-type "${METHOD}" \
    --num-base-heads "${NUM_BASE_HEADS}" \
    --num-generators "${NUM_GENERATORS}" \
    --fuse-base-qkv --no-enforce-paper-gt-mha \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps "${accumulation}" --learning-rate "${learning_rate}" \
    --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${epochs}" \
    --metric-for-best-model "${metric}" --greater-is-better \
    --seed "${seed}" --bf16 --num-workers 2 \
    --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
  echo "Completed ${run_name} at $(date -Is)"
}

run_specs() {
  local -a specs=("$@") pids=() labels=()
  local gpu=0 failed=0 spec task seed learning_rate stage
  for spec in "${specs[@]}"; do
    IFS="|" read -r task seed learning_rate stage <<< "${spec}"
    run_one "${gpu}" "${task}" "${seed}" "${learning_rate}" "${stage}" &
    pids+=("$!")
    labels+=("${spec}")
    gpu=$((gpu + 1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "Failed: ${labels[$i]}" >&2; failed=1; }
  done
  (( failed == 0 ))
}

run_fixed_wnli() {
  run_one 0 wnli 42 2e-5 fixed
}

# Sweep eight primary GLUE tasks.
for task in cola mrpc rte mnli qnli sst2 qqp stsb; do
  queue=()
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for seed in "${GRID_SEEDS[@]}"; do
      queue+=("${task}|${seed}|${learning_rate}|grid")
      if (( ${#queue[@]} == 2 )); then
        run_specs "${queue[@]}"
        queue=()
      fi
    done
  done
  (( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"

  prefix=${RUN_VERSION}_${task}
  metric=$(task_metric "${task}")
  python "${STAGE_DIR}/select_glue_learning_rates.py" \
    --root "${PROJECT_DIR}/lgma_runs" --prefix "${prefix}" \
    --metric-key "eval_${metric}" --methods "${METHOD}" \
    --output-json "${PROJECT_DIR}/lgma_runs/${prefix}_selected_learning_rates.json" \
    --output-tsv "${PROJECT_DIR}/lgma_runs/${prefix}_selected_learning_rates.tsv"

  selected=$(cut -f2 "${PROJECT_DIR}/lgma_runs/${prefix}_selected_learning_rates.tsv")
  echo "Selected ${task}: lr=${selected}"
  queue=()
  for seed in "${FINAL_SEEDS[@]}"; do
    queue+=("${task}|${seed}|${selected}|final")
    if (( ${#queue[@]} == 2 )); then
      run_specs "${queue[@]}"
      queue=()
    fi
  done
  (( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"
done

# WNLI follows the fixed historical exception.
run_fixed_wnli

python "${STAGE_DIR}/select_final_checkpoints.py"

echo "Completed B4/G8/H12 GLUE sweep at $(date -Is)"
