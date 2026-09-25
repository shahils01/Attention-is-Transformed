#!/usr/bin/env bash
#SBATCH --job-name=tiny-qkvid-b4g8h16-s0
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/scratch/shahils/job_scripts/tinystories-qkv-identity-20260910/repo
#SBATCH --output=/home/shahils/logs/%x_%j.out
#SBATCH --error=/home/shahils/logs/%x_%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export WANDB_RUN_ID=tiny-qkvid-b4g8h16-s0-20260910
export WANDB_RESUME=allow
DATA_DIR=/scratch/shahils/lgma_data/tinystory
OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_qkv_identity_b4g8h16_seed0_20260910
mkdir -p "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/checkpoint_final.pt" ]]; then
  echo "Training already finished; refusing a duplicate run."
  exit 0
fi
python -u experiments/check_qkv_identity.py
if [[ "${1:-}" == "--check-only" ]]; then exit 0; fi

resume_args=()
latest_checkpoint=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'checkpoint_step_*.pt' | sort -V | tail -n 1)
if [[ -n "$latest_checkpoint" ]]; then
  resume_args=(--resume_checkpoint "$latest_checkpoint")
fi
handle_time_limit() {
  trap - USR1
  echo "Allocation ending: stopping workers and requeueing this same job."
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  wait "$TRAIN_PID" || true
  if ! find "$OUTPUT_DIR" -maxdepth 1 -name 'checkpoint_step_*.pt' -print -quit | read -r _; then
    echo "No checkpoint; refusing automatic restart." >&2
    exit 2
  fi
  if [[ -f "$OUTPUT_DIR/checkpoint_final.pt" ]]; then exit 0; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap handle_time_limit USR1

# Reference: i2rLAB/lgma/a2r6d0gn, source commit 427220e9.
# Four GPUs x batch 256 x accumulation 2 = 2048; reference global batch preserved.
# Compact character encoding preserves tokenizer/sampling; see CPU parity test.
torchrun --standalone --nproc_per_node=4 experiments/train_tinystories_qkv_identity.py \
  --data_path "$DATA_DIR/TinyStoriesV2-GPT4-train.txt" \
  --val_data_path "$DATA_DIR/TinyStoriesV2-GPT4-valid.txt" \
  --device cuda --batch_size 256 --grad_accum_steps 2 --steps 250000 \
  --log_every 100 --eval_every 1000 --eval_batches 8 --save_every 100 \
  --keep_last_checkpoints 3 --milestone_checkpoint_every 50000 \
  --output_dir "$OUTPUT_DIR" --diagnostic_every 1000 --diagnostic_batches 2 \
  --wandb_entity i2rLAB --wandb_project lgma \
  --wandb_run_name tinystories_qkv_identity_b4g8h16_seed0 \
  --wandb_group tinystories-ablation-qkv-identity-b4g8h16 \
  --wandb_tags tinystory,qkv-identity,ValueIdentity,b4,g8,h16,softmax,seed0,h200,ddp4,b256,ga2,gb2048,reference-a2r6d0gn \
  --wandb_mode online --wandb_dir "$OUTPUT_DIR" \
  --induced_metric_diversity_weight 0 --metric_diversity_weight 0 \
  --ah_norm_weight 0 --ah_norm_max 0 \
  --d_model 1024 --num_layers 12 --head_dim 64 --base_dim 64 --value_dim 64 \
  --num_generators 8 --generator_type full --generator_mixing softmax \
  --metric_beta 1 --value_beta 1 --no-stabilize_generators --no-normalize_generators \
  --theta_init random_sphere --theta_init_scale 0.02 --generator_init_scale 0.02 \
  --context_length 512 --dropout 0.1 --lr 3e-4 --lr_schedule cosine \
  --warmup_steps 0 --lr_hold_steps 50000 --min_lr 1e-4 \
  --weight_decay 0.01 --max_grad_norm 1 --precision bf16 \
  --attention lgma_qkv_identity --num_heads 16 --num_base_heads 4 \
  --value_transform lie --no-fuse_base_qkv --no-fold_value_transform_into_output \
  --sdpa_gqa_mode auto --seed 0 "${resume_args[@]}" &
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"
status=$?
trap - USR1
exit "$status"
