#!/bin/bash

# Continue the one-GPU LGMA-residual run on one eight-H200 node. If the
# one-GPU producer is still queued or running when this allocation starts, the
# handoff logic stops it before selecting the newest completed checkpoint.
#SBATCH --job-name=lgma_residual_h200x8
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=36:00:00
#SBATCH --gpus-per-node=h200:8
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100
HANDOFF_JOB_ID=${HANDOFF_JOB_ID:-}
KEEP_CHECKPOINTS=3

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ -n "$HANDOFF_JOB_ID" ]] && squeue -h -j "$HANDOFF_JOB_ID" | grep -q .; then
  old_state=$(squeue -h -j "$HANDOFF_JOB_ID" -o %T | head -n 1)
  echo "Handing off from job ${HANDOFF_JOB_ID} in state ${old_state}."
  if [[ "$old_state" == "RUNNING" ]]; then
    # Invoke its USR1 handler so torchrun stops before the old allocation is
    # canceled. The last checkpoint reported as saved is already complete.
    scancel --batch --signal=USR1 "$HANDOFF_JOB_ID" || true
    for _ in {1..60}; do
      [[ $(squeue -h -j "$HANDOFF_JOB_ID" -o %T | head -n 1) != "RUNNING" ]] && break
      sleep 2
    done
  fi
  if squeue -h -j "$HANDOFF_JOB_ID" | grep -q .; then
    scancel "$HANDOFF_JOB_ID" || true
  fi
  for _ in {1..60}; do
    ! squeue -h -j "$HANDOFF_JOB_ID" | grep -q . && break
    sleep 2
  done
fi

prune_old_checkpoints() {
  local -a checkpoints
  local checkpoint_count index
  mapfile -t checkpoints < <(
    find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -printf '%T@:%p\n' 2>/dev/null \
      | sort -t: -k1,1nr \
      | cut -d: -f2-
  )
  checkpoint_count=${#checkpoints[@]}
  if (( checkpoint_count > KEEP_CHECKPOINTS )); then
    for ((index = KEEP_CHECKPOINTS; index < checkpoint_count; index++)); do
      rm -f -- "${checkpoints[index]}"
    done
  fi
}

LATEST_CHECKPOINT="$(
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -printf '%T@:%p\n' 2>/dev/null \
    | sort -t: -k1,1nr \
    | head -n 1 \
    | cut -d: -f2-
)"
RESUME_ARGS=()
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  echo "Resuming eight-GPU training from checkpoint: $LATEST_CHECKPOINT"
  RESUME_ARGS=(--resume_checkpoint "$LATEST_CHECKPOINT")
else
  echo "No checkpoint found; refusing to start a replacement run." >&2
  exit 2
fi

handle_time_limit() {
  echo "Received USR1; stopping training and requeueing job ${SLURM_JOB_ID}."
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
  fi
  if [[ -n "${PRUNE_PID:-}" ]] && kill -0 "$PRUNE_PID" 2>/dev/null; then
    kill "$PRUNE_PID" 2>/dev/null || true
    wait "$PRUNE_PID" || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit $?
}
trap handle_time_limit USR1

export OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node=8 experiments/train_tinystories.py \
  --data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-train.txt \
  --val_data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-valid.txt \
  --device cuda \
  --batch_size 256 \
  --grad_accum_steps 1 \
  --steps 250000 \
  --log_every 100 \
  --eval_every 1000 \
  --save_every 100 \
  --output_dir "$OUTPUT_DIR" \
  --diagnostic_every 1000 \
  --diagnostic_batches 2 \
  --wandb_project lgma \
  --wandb_run_name tinystory_lgma_residual_b4_h16_ospool \
  --wandb_group tinystories \
  --wandb_tags tinystory,lgma,ValueLie,residual,b4,h16,ospool,h200x8 \
  --induced_metric_diversity_weight 0.0 \
  --metric_diversity_weight 0.0 \
  --d_model 1024 \
  --num_layers 12 \
  --head_dim 64 \
  --base_dim 64 \
  --value_dim 64 \
  --num_generators 8 \
  --metric_beta 1.0 \
  --value_beta 1.0 \
  --no-stabilize_generators \
  --context_length 512 \
  --dropout 0.1 \
  --lr 3e-4 \
  --lr_schedule cosine \
  --warmup_steps 0 \
  --lr_hold_steps 50000 \
  --min_lr 1e-4 \
  --weight_decay 0.01 \
  --precision bf16 \
  --attention lgma_residual \
  --num_heads 16 \
  --num_base_heads 4 \
  --value_transform lie \
  "${RESUME_ARGS[@]}" &

TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    prune_old_checkpoints
    sleep 300
  done
) &
PRUNE_PID=$!

wait "$TRAIN_PID"
TRAIN_STATUS=$?
kill "$PRUNE_PID" 2>/dev/null || true
wait "$PRUNE_PID" 2>/dev/null || true
trap - USR1

if [[ "$TRAIN_STATUS" -eq 0 ]]; then
  echo "Training completed normally; not requeueing."
else
  echo "Training exited with status $TRAIN_STATUS; not requeueing automatically."
fi
exit "$TRAIN_STATUS"
