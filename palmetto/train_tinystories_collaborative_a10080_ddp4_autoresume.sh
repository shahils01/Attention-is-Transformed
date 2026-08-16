#!/usr/bin/env bash
#SBATCH --job-name=collab-h16-a10080-ddp4
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
OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_collaborative_h16_a10080_ddp4_gb2048
INITIAL_CHECKPOINT=/scratch/shahils/lgma_runs/tinystories_collaborative_h16_gh200_ddp4_gb2048/checkpoint_step_98300.pt
TOTAL_STEPS=250000

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail

export OMP_NUM_THREADS=2
export WANDB_RUN_ID=tinystory-collaborative-h16-gh200-ddp4-gb2048-20260808
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

if [[ -n "${latest_checkpoint}" ]]; then
  resume_checkpoint="${latest_checkpoint}"
else
  resume_checkpoint="${INITIAL_CHECKPOINT}"
fi

if [[ ! -f "${resume_checkpoint}" ]]; then
  echo "Resume checkpoint not found: ${resume_checkpoint}" >&2
  exit 2
fi
echo "Resuming from ${resume_checkpoint}"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT DATA_DIR OUTPUT_DIR TOTAL_STEPS resume_checkpoint

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
      --wandb_run_name tinystory_collaborative_h16_gh200_ddp4_gb2048 \
      --wandb_group tinystories \
      --wandb_tags tinystory,baseline,collaborative,h16,a10080,ddp4,b256,gb2048 \
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
      --attention collaborative \
      --num_heads 16 \
      --seed 0 \
      --resume_checkpoint "${resume_checkpoint}"
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
