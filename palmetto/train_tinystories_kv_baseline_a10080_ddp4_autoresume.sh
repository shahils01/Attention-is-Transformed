#!/usr/bin/env bash
#SBATCH --job-name=kvbaseline-a10080-ddp4
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/shahils/logs/%x_%j.out
#SBATCH --error=/home/shahils/logs/%x_%j.err

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
DATA_DIR=/scratch/shahils/lgma_data/tinystory
TOTAL_STEPS=250000
BASELINE_TYPE=${BASELINE_TYPE:?Set BASELINE_TYPE to mqa or gqa}
SEED=${SEED:-0}

case "${BASELINE_TYPE}" in
  mqa)
    ATTENTION_ARGS=(--attention mqa)
    BASELINE_TAG=mqa-h16-kv1
    ;;
  gqa)
    NUM_KV_HEADS=${NUM_KV_HEADS:-4}
    ATTENTION_ARGS=(--attention gqa --num_kv_heads "${NUM_KV_HEADS}")
    BASELINE_TAG=gqa-h16-kv${NUM_KV_HEADS}
    ;;
  *)
    echo "Unsupported BASELINE_TYPE: ${BASELINE_TYPE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_${BASELINE_TAG}_a10080_ddp4_gb2048_seed${SEED}
RUN_NAME=tinystory_${BASELINE_TAG}_a10080_ddp4_gb2048_seed${SEED}

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail

export OMP_NUM_THREADS=2
export WANDB_RUN_ID=${RUN_NAME}-20260815
export WANDB_RESUME=allow

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

if [[ -f "${OUTPUT_DIR}/checkpoint_final.pt" ]]; then
  echo "Final checkpoint already exists; training is complete."
  exit 0
fi

latest_checkpoint="$({ python - "${OUTPUT_DIR}" <<'PY'
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
candidates = []
for path in output_dir.glob("checkpoint_step_*.pt"):
    match = re.fullmatch(r"checkpoint_step_(\d+)\.pt", path.name)
    if match:
        candidates.append((int(match.group(1)), path))
if candidates:
    print(max(candidates)[1])
PY
} )"

resume_args=()
if [[ -n "${latest_checkpoint}" ]]; then
  echo "Resuming from ${latest_checkpoint}"
  resume_args=(--resume_checkpoint "${latest_checkpoint}")
else
  echo "No checkpoint found; starting a fresh ${BASELINE_TYPE^^} run."
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT DATA_DIR OUTPUT_DIR TOTAL_STEPS RUN_NAME BASELINE_TYPE SEED
export ATTENTION_ARGS_STR="${ATTENTION_ARGS[*]}"
export RESUME_CHECKPOINT="${resume_args[1]:-}"

handle_time_limit() {
  echo "Received USR1 before wall-time limit; terminating distributed launch and requeueing ${SLURM_JOB_ID}."
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit $?
}
trap handle_time_limit USR1

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
  attention_args=( ${ATTENTION_ARGS_STR} )
  resume_args=()
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    resume_args=(--resume_checkpoint "${RESUME_CHECKPOINT}")
  fi
  exec torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=2 \
    --node_rank="${SLURM_NODEID}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/train_tinystories.py \
      --data_path "${DATA_DIR}/TinyStoriesV2-GPT4-train.txt" \
      --val_data_path "${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt" \
      --device cuda \
      --batch_size 256 \
      --grad_accum_steps 2 \
      --steps "${TOTAL_STEPS}" \
      --log_every 100 \
      --eval_every 1000 \
      --save_every 100 \
      --keep_last_checkpoints 3 \
      --milestone_checkpoint_every 50000 \
      --output_dir "${OUTPUT_DIR}" \
      --diagnostic_every 1000 \
      --diagnostic_batches 2 \
      --wandb_project lgma \
      --wandb_run_name "${RUN_NAME}" \
      --wandb_group tinystories \
      --wandb_tags "tinystory,baseline,${BASELINE_TYPE},a10080,ddp4,b256,gb2048,seed${SEED}" \
      --wandb_mode online \
      --wandb_dir "${OUTPUT_DIR}" \
      --d_model 1024 \
      --num_layers 12 \
      --head_dim 64 \
      --base_dim 64 \
      --value_dim 64 \
      --context_length 512 \
      --dropout 0.1 \
      --lr 3e-4 \
      --lr_schedule cosine \
      --lr_schedule_steps 500000 \
      --warmup_steps 0 \
      --lr_hold_steps 50000 \
      --min_lr 1e-4 \
      --weight_decay 0.01 \
      --precision bf16 \
      --num_heads 16 \
      --seed "${SEED}" \
      "${attention_args[@]}" \
      "${resume_args[@]}"
' &

TRAIN_PID=$!
wait "${TRAIN_PID}"
train_status=$?
trap - USR1

if [[ "${train_status}" -eq 0 ]]; then
  echo "Training completed normally; not requeueing."
else
  echo "Training exited with status ${train_status}; not requeueing automatically."
fi
exit "${train_status}"
