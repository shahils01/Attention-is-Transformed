#!/usr/bin/env bash
# BERT MLM recovery/from-scratch runner for the official MHA checkpoint and
# paper-faithful GT-MHA. Submit once per ATTENTION_TYPE.
#
#   sbatch --export=ALL,ATTENTION_TYPE=mha deltaai/train_bert_mlm_gh200x4.slurm
#   sbatch --export=ALL,ATTENTION_TYPE=gt_mha_exact deltaai/train_bert_mlm_gh200x4.slurm
#
# Set INITIALIZATION=random for the primary from-scratch architecture test.
# The default checkpoint mode is the secondary pretrained-conversion test.
#SBATCH --job-name=bert-attn
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/biad/sshaik4
REPO_DIR=/u/sshaik4/Attention-is-Transformed
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
ATTENTION_TYPE=${ATTENTION_TYPE:?Set ATTENTION_TYPE to mha, gt_mha_exact, or gt_mha_residual}
INITIALIZATION=${INITIALIZATION:-checkpoint}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-100000}
MODEL_NAME=${MODEL_NAME:-google-bert/bert-base-uncased}
WANDB_PROJECT=${WANDB_PROJECT:-bert-gt-mha}
GENERATOR_MIXING=${GENERATOR_MIXING:-softmax}
ENFORCE_PAPER_GT_MHA=${ENFORCE_PAPER_GT_MHA:-true}

case "${ENFORCE_PAPER_GT_MHA}" in
  true) PAPER_ARGS=(--enforce-paper-gt-mha) ;;
  false) PAPER_ARGS=(--no-enforce-paper-gt-mha) ;;
  *) echo "ENFORCE_PAPER_GT_MHA must be true or false" >&2; exit 2 ;;
esac

case "${ATTENTION_TYPE}" in
  mha)
    ATTENTION_ARGS=(--attention-type mha)
    ;;
  gt_mha_exact)
    ATTENTION_ARGS=(
      --attention-type gt_mha_exact
      --num-base-heads 6
      --num-generators 8
      --generator-mixing "${GENERATOR_MIXING}"
      --use-sdpa
      --fuse-base-qkv
      --sdpa-gqa-mode native
      "${PAPER_ARGS[@]}"
    )
    ;;
  gt_mha_residual)
    ATTENTION_ARGS=(
      --attention-type gt_mha_residual
      --num-base-heads 6
      --num-generators 8
      --generator-mixing "${GENERATOR_MIXING}"
      --use-sdpa
      --fuse-base-qkv
      --sdpa-gqa-mode native
      "${PAPER_ARGS[@]}"
    )
    ;;
  *)
    echo "Unsupported ATTENTION_TYPE=${ATTENTION_TYPE}; use mha, gt_mha_exact, or gt_mha_residual" >&2
    exit 2
    ;;
esac

MIXING_SUFFIX=
if [[ "${ATTENTION_TYPE}" != "mha" && "${GENERATOR_MIXING}" != "softmax" ]]; then
  MIXING_SUFFIX=_mix${GENERATOR_MIXING}
fi
RUN_NAME=bert_base_${ATTENTION_TYPE}_b6g8h12_${INITIALIZATION}_seed${SEED}${MIXING_SUFFIX}
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4
export WANDB_PROJECT
export WANDB_ENTITY=${WANDB_ENTITY:-i2rLAB}
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

resume_args=()
latest_checkpoint=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
  | sort -t- -k2,2n \
  | tail -1 || true)
if [[ -n "${latest_checkpoint}" ]]; then
  resume_args=(--resume-from-checkpoint "${latest_checkpoint}")
fi

python -m torch.distributed.run --standalone --nproc_per_node=4 experiments/train_bert_mlm.py \
  --model-name-or-path "${MODEL_NAME}" \
  --initialization "${INITIALIZATION}" \
  "${ATTENTION_ARGS[@]}" \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --max-sequence-length 128 \
  --mlm-probability 0.15 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --warmup-steps 10000 \
  --max-steps "${MAX_STEPS}" \
  --logging-steps 100 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --seed "${SEED}" \
  --bf16 \
  --gradient-checkpointing \
  --output-dir "${OUTPUT_DIR}" \
  --report-to wandb \
  --run-name "${RUN_NAME}" \
  "${resume_args[@]}"
