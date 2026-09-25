#!/usr/bin/env bash
# Seed-42 explicit-attention control. Only --use-sdpa is changed from the original
# model/training configuration. A separate directory preserves the SDPA result.
#SBATCH --job-name=bert-idboth-exp
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --chdir=/work/hdd/biad/sshaik4/repos/Attention-is-Transformed-bert-identity-both
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR=/work/hdd/biad/sshaik4
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed-bert-identity-both
VENV_DIR=${PROJECT_DIR}/venvs/lgma-bert-no-te
SEED=42
RUN_NAME=bert_base_gt_mha_identity_both_b4g8h12_random_seed42_explicit_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}
module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=4
export WANDB_PROJECT=bert-gt-mha
export WANDB_ENTITY=i2rLAB
export WANDB_RUN_GROUP=bert-ablation-identity-both-b4g8h12-explicit
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_NAME=${RUN_NAME}
export WANDB_RESUME=allow
export WANDB_TAGS=bert-pretraining,identity-both,explicit-attention,seed-42,backend-control-v1
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/compat/no_transformer_engine:${REPO_DIR}/src:${PYTHONPATH:-}"
test -f experiments/train_bert_mlm.py
test -d "${HF_DATASETS_CACHE}"
if [[ -f "${OUTPUT_DIR}/train_results.json" ]]; then
  echo "Run already completed: ${OUTPUT_DIR}"
  exit 0
fi
resume_args=()
if [[ -d "${OUTPUT_DIR}" ]]; then
  latest_checkpoint=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)
  if [[ -z "${latest_checkpoint}" ]]; then
    echo "Existing output has no resumable checkpoint; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  resume_args=(--resume-from-checkpoint "${latest_checkpoint}")
fi
mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
echo "Starting ${RUN_NAME}; group=${WANDB_RUN_GROUP}; job=${SLURM_JOB_ID}; explicit attention"
sha256sum experiments/train_bert_mlm.py src/lgma/bert.py src/lgma/attention.py
python -m torch.distributed.run --standalone --nproc_per_node=4 experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random --attention-type gt_mha_identity_both \
  --num-base-heads 4 --num-generators 8 --generator-mixing softmax \
  --no-enforce-paper-gt-mha --no-use-sdpa --fuse-base-qkv --sdpa-gqa-mode native \
  --dataset-name wikitext --dataset-config wikitext-103-raw-v1 \
  --max-sequence-length 128 --mlm-probability 0.15 --validation-mask-seed 17029 \
  --per-device-train-batch-size 32 --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 2 --learning-rate 5e-5 --weight-decay 0.01 \
  --warmup-steps 10000 --max-steps 100000 --logging-steps 100 \
  --eval-steps 1000 --save-steps 1000 --save-total-limit 3 --num-workers 4 \
  --seed "${SEED}" --bf16 --gradient-checkpointing \
  --output-dir "${OUTPUT_DIR}" --report-to wandb --run-name "${RUN_NAME}" \
  "${resume_args[@]}"
