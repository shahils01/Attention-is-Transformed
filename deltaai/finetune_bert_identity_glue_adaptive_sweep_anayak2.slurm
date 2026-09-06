#!/usr/bin/env bash
# Equal-budget paper GLUE sweep for the score-identity GT-MHA ablation.
# Submit with FAMILY=stable, hivar, or stsb. Four independent one-GPU runs
# execute concurrently, with seeds 42-44 selecting the LR and 45-46 confirming it.
# The stable family also runs the fixed WNLI protocol after its adaptive tasks.
#SBATCH --job-name=id-glue-sweep
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bicn/anayak2
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed-bert-identity
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
TRAIN_SCRIPT=${REPO_DIR}/experiments/finetune_bert_glue.py
SELECT_SCRIPT=${REPO_DIR}/experiments/select_glue_learning_rates.py
CACHE_SCRIPT=${REPO_DIR}/experiments/cache_glue_tasks.py
CHECKPOINT_DIR=${PROJECT_DIR}/lgma_runs/bert_base_gt_mha_identity_qk_b4g8h12_random_seed42
FAMILY=${FAMILY:?Submit with FAMILY=stable, hivar, or stsb}
METHOD=gt_mha_identity
GRID_SEEDS=(42 43 44)
FINAL_SEEDS=(45 46)
LEARNING_RATES=(2e-5 3e-5 5e-5)

case "${FAMILY}" in
  stable)
    RUN_PREFIX=glue_stable_papersweep_v1
    TASKS=(mnli qnli sst2 qqp)
    EPOCHS=4
    WANDB_PROJECT=bert-glue-stable-sweep
    WANDB_TAG_PREFIX=paper-glue-stable-v1
    ;;
  hivar)
    RUN_PREFIX=glue_hivar_papersweep_v1
    TASKS=(cola mrpc rte)
    EPOCHS=8
    WANDB_PROJECT=bert-glue-high-variance-sweep
    WANDB_TAG_PREFIX=paper-glue-hivar-v1
    ;;
  stsb)
    RUN_PREFIX=stsb_papersweep_v2
    TASKS=(stsb)
    EPOCHS=8
    WANDB_PROJECT=bert-gt-mha-residual-stsb-sweep
    WANDB_TAG_PREFIX=paper-stsb-sweep-v2
    ;;
  *)
    echo "Unknown FAMILY=${FAMILY}; expected stable, hivar, or stsb" >&2
    exit 2
    ;;
esac

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
for path in "${TRAIN_SCRIPT}" "${SELECT_SCRIPT}" "${CACHE_SCRIPT}" \
            "${CHECKPOINT_DIR}/model.safetensors" \
            "${CHECKPOINT_DIR}/bert_gt_mha_manifest.json" \
            "${CHECKPOINT_DIR}/gt_mha_state_dict.pt"; do
  test -f "${path}" || { echo "Missing required file: ${path}" >&2; exit 1; }
done

export OMP_NUM_THREADS=4
export WANDB_ENTITY=i2rLAB
export WANDB_PROJECT
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=allow
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/lgma_runs"

if [[ "${FAMILY}" == stable ]]; then
  CACHE_TASKS=("${TASKS[@]}" wnli)
else
  CACHE_TASKS=("${TASKS[@]}")
fi
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python "${CACHE_SCRIPT}" --tasks "${CACHE_TASKS[@]}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/compat/no_transformer_engine:${REPO_DIR}/src:${PYTHONPATH:-}"

lr_tag() {
  printf '%s' "${1//e-/em}"
}

task_metric() {
  case "$1" in
    cola) printf '%s' matthews_correlation ;;
    mrpc|qqp|stsb) printf '%s' combined_score ;;
    mnli|qnli|rte|sst2|wnli) printf '%s' accuracy ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

latest_checkpoint() {
  local output_dir=$1 latest
  latest=$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null \
    | sort -V | tail -1)
  printf '%s' "${latest}"
}

run_one() {
  local gpu=$1 task=$2 seed=$3 learning_rate=$4 stage=$5 epochs=$6
  local tag metric_name prefix run_name output_dir checkpoint wandb_group resume_args=()
  tag=$(lr_tag "${learning_rate}")
  metric_name=$(task_metric "${task}")
  if [[ "${task}" == wnli ]]; then
    prefix=glue_official_wnli_fixed
    run_name=${prefix}_${METHOD}_lr${tag}_seed${seed}
  else
    prefix=${RUN_PREFIX}_${task}
    run_name=${prefix}_${METHOD}_lr${tag}_seed${seed}_${stage}
  fi
  if [[ "${FAMILY}" == stsb ]]; then
    wandb_group=${METHOD}
  else
    wandb_group=${task}-${METHOD}
  fi
  output_dir=${PROJECT_DIR}/lgma_runs/${run_name}

  if [[ -f "${output_dir}/eval_results.json" ]]; then
    echo "Skipping completed run ${run_name}"
    return 0
  fi
  if [[ -d "${output_dir}" ]]; then
    checkpoint=$(latest_checkpoint "${output_dir}")
    test -n "${checkpoint}" || {
      echo "Incomplete run has no resumable checkpoint: ${output_dir}" >&2
      return 1
    }
    resume_args=(--resume-from-checkpoint "${checkpoint}")
    echo "Resuming ${run_name} from ${checkpoint}"
  fi

  echo "Starting ${run_name} on GPU ${gpu} at $(date -Is)"
  CUDA_VISIBLE_DEVICES=${gpu} \
  WANDB_RUN_ID=${prefix}_${METHOD}_${tag}_s${seed}_${stage} \
  WANDB_NAME=${run_name} \
  WANDB_RUN_GROUP=${wandb_group} \
  WANDB_TAGS=${WANDB_TAG_PREFIX},task-${task},attention-${METHOD},lr-${tag},stage-${stage} \
  python "${TRAIN_SCRIPT}" \
    --model-name-or-path "${CHECKPOINT_DIR}" --task "${task}" \
    --attention-type "${METHOD}" --num-base-heads 4 --num-generators 8 \
    --no-enforce-paper-gt-mha --use-sdpa --fuse-base-qkv --sdpa-gqa-mode native \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 4 --learning-rate "${learning_rate}" \
    --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${epochs}" \
    --save-total-limit 1 --metric-for-best-model "${metric_name}" --greater-is-better \
    --seed "${seed}" --bf16 --num-workers 2 \
    --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}" \
    "${resume_args[@]}"
}

run_specs() {
  local -a specs=("$@") pids=() labels=()
  local gpu=0 failed=0 spec task seed learning_rate stage epochs
  for spec in "${specs[@]}"; do
    IFS='|' read -r task seed learning_rate stage epochs <<<"${spec}"
    run_one "${gpu}" "${task}" "${seed}" "${learning_rate}" "${stage}" "${epochs}" &
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
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for seed in "${GRID_SEEDS[@]}"; do
      queue+=("${task}|${seed}|${learning_rate}|grid|${EPOCHS}")
      if (( ${#queue[@]} == 4 )); then
        run_specs "${queue[@]}"
        queue=()
      fi
    done
  done
  (( ${#queue[@]} == 0 )) || run_specs "${queue[@]}"

  prefix=${RUN_PREFIX}_${task}
  selection_json=${PROJECT_DIR}/lgma_runs/${prefix}_${METHOD}_selected_learning_rates.json
  selection_tsv=${PROJECT_DIR}/lgma_runs/${prefix}_${METHOD}_selected_learning_rates.tsv
  python "${SELECT_SCRIPT}" \
    --root "${PROJECT_DIR}/lgma_runs" --prefix "${prefix}" \
    --metric-key "eval_$(task_metric "${task}")" --methods "${METHOD}" \
    --output-json "${selection_json}" --output-tsv "${selection_tsv}"

  queue=()
  while IFS=$'\t' read -r method learning_rate mean_score; do
    test "${method}" = "${METHOD}"
    echo "Selected ${task}/${method}: lr=${learning_rate}, grid mean=${mean_score}"
    for seed in "${FINAL_SEEDS[@]}"; do
      queue+=("${task}|${seed}|${learning_rate}|final|${EPOCHS}")
    done
  done < "${selection_tsv}"
  run_specs "${queue[@]}"
done

if [[ "${FAMILY}" == stable ]]; then
  run_specs "wnli|42|2e-5|fixed|4"
fi

echo "Completed Identity GT-MHA ${FAMILY} GLUE sweep at $(date -Is)"
