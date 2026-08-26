#!/usr/bin/env bash
# Targeted STS-B stability diagnostic for GT-MHA residual.
# Nine runs (3 learning rates x 3 seeds) share one four-GH200 allocation.
# Each one-GPU process uses gradient accumulation 4 to preserve the original
# effective training batch of 128 used by the four-GPU GLUE runs.
#SBATCH --job-name=gtres-stsb-sweep
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
SWEEP_SCRIPT=${PROJECT_DIR}/lgma_jobs/finetune_bert_glue_stsb_sweep.py
HF_CHECKPOINT_ROOT=${PROJECT_DIR}/lgma_runs/hf_gt_mha_residual
CHECKPOINT_DIR="${HF_CHECKPOINT_ROOT}/BERT-Base Checkpoints/gt_mha_residual/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000"
EPOCHS=8
SEEDS=(42 43 44)
LEARNING_RATES=(1e-5 2e-5 3e-5)

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
test -f "${SWEEP_SCRIPT}" || { echo "Missing sweep trainer: ${SWEEP_SCRIPT}" >&2; exit 1; }

hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/gt_mha_residual/bert_base_gt_mha_residual_b4g8h12_random_seed44_explicit_fuseqkv_v2/checkpoint-100000/*" \
  --local-dir "${HF_CHECKPOINT_ROOT}"

export OMP_NUM_THREADS=4
export WANDB_PROJECT=bert-gt-mha-residual-stsb-sweep
export WANDB_RUN_GROUP=gt-mha-residual-stsb-lr-epoch-diagnostic
export WANDB_JOB_TYPE=glue-finetune
export WANDB_RESUME=never
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_DATASETS_CACHE}/glue" "${PROJECT_DIR}/logs"
legacy=${HF_DATASETS_CACHE}/glue/stsb
canonical=${HF_DATASETS_CACHE}/nyu-mll___glue/stsb
test -e "${canonical}" || { echo "Missing cached GLUE/STS-B: ${canonical}" >&2; exit 1; }
test -e "${legacy}" || ln -s ../nyu-mll___glue/stsb "${legacy}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "${REPO_DIR}"
export PYTHONPATH=${REPO_DIR}/src:${PYTHONPATH:-}

run_one() {
  local gpu=$1 seed=$2 learning_rate=$3
  local lr_tag=${learning_rate//-/m}
  local run_name=bert_base_gt_mha_residual_b4g8h12_seed44_glue_stsb_ftseed${seed}_${EPOCHS}ep_lr${lr_tag}_sweep_v1
  local output_dir=${PROJECT_DIR}/lgma_runs/${run_name}
  test ! -e "${output_dir}" || { echo "Refusing to overwrite ${output_dir}" >&2; return 1; }
  echo "Starting ${run_name} on GPU ${gpu} at $(date -Is)"
  CUDA_VISIBLE_DEVICES=${gpu} \
  WANDB_RUN_ID=gtres_stsb_s${seed}_${EPOCHS}ep_lr${lr_tag}_v1 \
  WANDB_NAME=${run_name} \
  python "${SWEEP_SCRIPT}" \
    --model-name-or-path "${CHECKPOINT_DIR}" --task stsb \
    --attention-type gt_mha_residual --num-base-heads 4 --num-generators 8 --fuse-base-qkv \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 4 --learning-rate "${learning_rate}" \
    --weight-decay 0.01 --warmup-ratio 0.1 --num-train-epochs "${EPOCHS}" \
    --metric-for-best-model combined_score --greater-is-better \
    --seed "${seed}" --bf16 --num-workers 2 \
    --output-dir "${output_dir}" --report-to wandb --run-name "${run_name}"
}

pids=()
labels=()
slot=0
failed=0
for learning_rate in "${LEARNING_RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${slot}" "${seed}" "${learning_rate}" &
    pids+=("$!")
    labels+=("seed=${seed},lr=${learning_rate}")
    slot=$((slot + 1))
    if (( slot == 4 )); then
      for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "Failed: ${labels[$i]}" >&2; failed=1; }
      done
      pids=()
      labels=()
      slot=0
    fi
  done
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "Failed: ${labels[$i]}" >&2; failed=1; }
done

(( failed == 0 )) || exit 1
echo "Completed GT-MHA STS-B sweep at $(date -Is)"
