#!/bin/bash

# Usage: sbatch baseline_ddp_autoresume.sh {mha|mqa|gqa}
# The job requeues before each 12-hour limit and resumes the newest checkpoint.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=h200:4
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

ATTENTION="${1:-}"
case "$ATTENTION" in
  mha|mqa)
    ATTENTION_ARGS=()
    ;;
  gqa)
    # Four KV heads for sixteen query heads, matching the trainer default explicitly.
    ATTENTION_ARGS=(--num_kv_heads 4)
    ;;
  *)
    echo "Usage: sbatch $0 {mha|mqa|gqa}" >&2
    exit 2
    ;;
esac

# Slurm batch shells launched through non-interactive SSH do not inherit the
# shell function that provides `module`, so initialize it explicitly.
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail
cd /home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed/

OUTPUT_DIR="/scratch/shahils/lgma_runs/large_tinystories_${ATTENTION}_b128_h16_h200_4"
KEEP_CHECKPOINTS=3
mkdir -p "$OUTPUT_DIR"

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
    echo "Pruning $((checkpoint_count - KEEP_CHECKPOINTS)) old checkpoints; retaining the newest $KEEP_CHECKPOINTS."
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
  echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
  RESUME_ARGS=(--resume_checkpoint "$LATEST_CHECKPOINT")
else
  echo "No checkpoint found; starting a new run."
fi

handle_time_limit() {
  echo "Received USR1 before the Slurm wall-time limit; stopping training and requeueing job ${SLURM_JOB_ID}."
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

torchrun --standalone --nproc_per_node=4 experiments/train_tinystories.py \
  --data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-train.txt \
  --val_data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-valid.txt \
  --device cuda \
  --batch_size 128 \
  --grad_accum_steps 4 \
  --steps 500000 \
  --log_every 100 \
  --eval_every 1000 \
  --save_every 100 \
  --output_dir "$OUTPUT_DIR" \
  --diagnostic_every 1000 \
  --diagnostic_batches 2 \
  --wandb_project lgma \
  --wandb_run_name "tinystory_${ATTENTION}_b128_h200" \
  --wandb_group tinystories_baselines \
  --wandb_tags tinystory,"$ATTENTION",baseline,b128,h16,h200,sdpa \
  --d_model 1024 \
  --num_layers 12 \
  --head_dim 64 \
  --context_length 512 \
  --dropout 0.1 \
  --lr 3e-4 \
  --lr_schedule cosine \
  --warmup_steps 0 \
  --lr_hold_steps 50000 \
  --min_lr 1e-4 \
  --weight_decay 0.01 \
  --precision bf16 \
  --attention "$ATTENTION" \
  --num_heads 16 \
  "${ATTENTION_ARGS[@]}" \
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
