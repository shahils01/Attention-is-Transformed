#!/usr/bin/env bash
# Build official GLUE prediction packages for GQA and Collaborative MHA.
# Public unlabeled test examples are used; official test labels are never read.
# Checkpoint selection is based only on the completed validation sweeps.
#SBATCH --job-name=glue-official-base
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=2
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bifg/arai3/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bifg/arai3
REPO_DIR=/u/arai3/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
JOB_DIR=${PROJECT_DIR}/lgma_jobs
RUN_ROOT=${PROJECT_DIR}/lgma_runs
TRAIN_SCRIPT=${JOB_DIR}/finetune_bert_glue_hivar_sweep.py
CACHE_SCRIPT=${JOB_DIR}/cache_glue_tasks.py
RESOLVE_SCRIPT=${JOB_DIR}/resolve_official_glue_baseline_checkpoints.py
PREDICT_SOURCE=${JOB_DIR}/generate_official_glue_submission.py
PREDICT_SCRIPT=/tmp/generate_official_glue_baselines_${SLURM_JOB_ID}.py
HF_ROOT=${RUN_ROOT}/hf_official_glue_baselines
STSB_ROOT=${RUN_ROOT}/hf_official_glue_stsb
OUTPUT_ROOT=${RUN_ROOT}/official_glue_submissions_baselines_v1
MANIFEST=${OUTPUT_ROOT}/selected_checkpoints.json

GQA_CHECKPOINT=${HF_ROOT}/BERT-Base\ Checkpoints/gqa/bert_base_gqa_kv4_random_seed42_v1/checkpoint-100000
COLLAB_CHECKPOINT=${HF_ROOT}/BERT-Base\ Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000
GQA_WNLI=${RUN_ROOT}/glue_official_wnli_fixed_gqa_lr2em5_seed42
COLLAB_WNLI=${RUN_ROOT}/glue_official_wnli_fixed_collaborative_lr2em5_seed42

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
for path in "${TRAIN_SCRIPT}" "${CACHE_SCRIPT}" "${RESOLVE_SCRIPT}" "${PREDICT_SOURCE}"; do
  test -f "${path}" || { echo "Missing required script: ${path}" >&2; exit 1; }
done
mkdir -p "${PROJECT_DIR}/logs" "${HF_ROOT}" "${STSB_ROOT}"
test ! -e "${OUTPUT_ROOT}" || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }

export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONPATH=${REPO_DIR}/src:${PYTHONPATH:-}

hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/gqa/bert_base_gqa_kv4_random_seed42_v1/checkpoint-100000/*" \
  --local-dir "${HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "BERT-Base Checkpoints/collaborative_mha/bert_base_collaborative_h12_random_seed43/checkpoint-100000/*" \
  --local-dir "${HF_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "Fine-tuned GLUE Checkpoints/stsb/gqa/stsb_papersweep_v2_gqa_lr3em5_seed44_grid/*" \
  --local-dir "${STSB_ROOT}"
hf download shahils/GT-MHA --repo-type model \
  --include "Fine-tuned GLUE Checkpoints/stsb/collaborative/stsb_papersweep_v2_collaborative_lr5em5_seed45_final/*" \
  --local-dir "${STSB_ROOT}"

test -f "${GQA_CHECKPOINT}/model.safetensors"
test -f "${COLLAB_CHECKPOINT}/model.safetensors"

HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python "${CACHE_SCRIPT}" \
  --tasks mnli qnli sst2 cola mrpc stsb qqp rte wnli ax
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

run_wnli() {
  local gpu=$1 method=$2 checkpoint=$3 output_dir=$4
  if [[ -f "${output_dir}/eval_results.json" ]]; then
    echo "Using completed fixed WNLI run: ${output_dir}"
    return 0
  fi
  test ! -e "${output_dir}" || { echo "Refusing incomplete WNLI output: ${output_dir}" >&2; return 1; }
  local -a attention_args
  case "${method}" in
    gqa) attention_args=(--attention-type gqa --num-kv-heads 4) ;;
    collaborative) attention_args=(--attention-type collaborative) ;;
    *) echo "Unknown method ${method}" >&2; return 1 ;;
  esac
  CUDA_VISIBLE_DEVICES=${gpu} python "${TRAIN_SCRIPT}" \
    --model-name-or-path "${checkpoint}" --task wnli "${attention_args[@]}" \
    --max-sequence-length 128 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 1 --learning-rate 2e-5 --weight-decay 0.01 \
    --warmup-ratio 0.1 --num-train-epochs 4 --save-total-limit 1 \
    --metric-for-best-model accuracy --greater-is-better --seed 42 --bf16 --num-workers 2 \
    --output-dir "${output_dir}" --report-to none --run-name "$(basename "${output_dir}")"
  test -f "${output_dir}/eval_results.json"
}

run_wnli 0 gqa "${GQA_CHECKPOINT}" "${GQA_WNLI}" &
pid_gqa=$!
run_wnli 1 collaborative "${COLLAB_CHECKPOINT}" "${COLLAB_WNLI}" &
pid_collab=$!
wait "${pid_gqa}"
wait "${pid_collab}"

mkdir -p "${OUTPUT_ROOT}"
python "${RESOLVE_SCRIPT}" \
  --run-root "${RUN_ROOT}" --stsb-root "${STSB_ROOT}" --output "${MANIFEST}"

cp "${PREDICT_SOURCE}" "${PREDICT_SCRIPT}"
sed -i 's/choices=("mha", "gt_mha_residual")/choices=("gqa", "collaborative")/' "${PREDICT_SCRIPT}"
sed -i 's/expected_layers = 0 if args.method == "mha" else 12/expected_layers = 12/' "${PREDICT_SCRIPT}"
grep -F 'choices=("gqa", "collaborative")' "${PREDICT_SCRIPT}"
grep -F 'expected_layers = 12' "${PREDICT_SCRIPT}"

CUDA_VISIBLE_DEVICES=0 python "${PREDICT_SCRIPT}" \
  --method gqa --checkpoint-manifest "${MANIFEST}" --repo-dir "${REPO_DIR}" \
  --output-root "${OUTPUT_ROOT}" --batch-size 128 &
pid_gqa=$!
CUDA_VISIBLE_DEVICES=1 python "${PREDICT_SCRIPT}" \
  --method collaborative --checkpoint-manifest "${MANIFEST}" --repo-dir "${REPO_DIR}" \
  --output-root "${OUTPUT_ROOT}" --batch-size 128 &
pid_collab=$!
wait "${pid_gqa}"
wait "${pid_collab}"

sha256sum "${OUTPUT_ROOT}"/*.zip
for archive in "${OUTPUT_ROOT}"/*.zip; do
  echo "=== $(basename "${archive}")"
  unzip -l "${archive}"
done
echo "Official GLUE baseline packages completed at $(date -Is)"
