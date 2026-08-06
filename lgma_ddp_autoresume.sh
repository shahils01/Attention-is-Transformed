#!/bin/bash

# Run continuously across 12-hour allocations.  Slurm sends USR1 to this batch
# shell five minutes before the wall-time limit; the handler stops torchrun and
# requeues the job.  The next allocation resumes the newest completed checkpoint.
#SBATCH --job-name=lgma_b4h16
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=h200:4
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

# Slurm batch shells launched through non-interactive SSH do not inherit the
# shell function that provides `module`, so initialize it explicitly.
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail

cd /home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed/

OUTPUT_DIR=/scratch/shahils/lgma_runs/large_tinystories_lgma_b4_h16_g8_beta1_softmax_h200_4
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

# A checkpoint is only considered after torch.save has returned. Choosing the
# newest file by mtime also means an interrupted write is retained alongside two
# older completed checkpoints, rather than becoming the only recovery point.
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
  --batch_size 256 \
  --grad_accum_steps 2 \
  --steps 500000 \
  --log_every 100 \
  --eval_every 1000 \
  --save_every 100 \
  --output_dir "$OUTPUT_DIR" \
  --diagnostic_every 1000 \
  --diagnostic_batches 2 \
  --wandb_project lgma \
  --wandb_run_name tinystory_lgma_multibase_b4_h16_ospool \
  --wandb_group tinystories \
  --wandb_tags tinystory,lgma,ValueLie,multibase,b4,h16,ospool \
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
  --head_generator_symmetric_cap 6.0 \
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
  --compile \
  --attention lgma_multibase \
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
