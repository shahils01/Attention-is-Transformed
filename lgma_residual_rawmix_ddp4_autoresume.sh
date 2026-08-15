#!/bin/bash
#SBATCH --job-name=lgma-res-raw-b4h16
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
REQUIRED_COMMIT=f9a7bc88b6b3f6888b9406a80b7f85c04a9fbefe
DATA_DIR=/scratch/shahils/lgma_data/tinystory
OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_lgma_residual_b4_h16_g8_mixnone_theta035_ddp4_gb2048
TOTAL_STEPS=250000

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID=tinystory-lgma-residual-b4-h16-g8-mixnone-theta035-ddp4-gb2048-20260815
export WANDB_RESUME=allow

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"

actual_commit=$(git rev-parse HEAD)
if ! git merge-base --is-ancestor "${REQUIRED_COMMIT}" "${actual_commit}"; then
  echo "Source revision ${actual_commit} does not contain required commit ${REQUIRED_COMMIT}" >&2
  exit 2
fi

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
  echo "No checkpoint found; starting a fresh raw-mixing run."
fi

handle_time_limit() {
  echo "Received USR1 before the wall-time limit; requeueing job ${SLURM_JOB_ID}."
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit $?
}
trap handle_time_limit USR1

torchrun --standalone --nproc_per_node=4 experiments/train_tinystories.py \
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
  --wandb_entity i2rLAB \
  --wandb_project lgma \
  --wandb_run_name tinystory_lgma_residual_b4_h16_g8_mixnone_theta035_ddp4_gb2048 \
  --wandb_group tinystories \
  --wandb_tags tinystory,gtmha,lgma,residual,ValueLie,b4,h16,d64,g8,mixnone,theta035,h100,ddp4,b256,ga2,gb2048 \
  --wandb_mode online \
  --wandb_dir "${OUTPUT_DIR}" \
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
  --generator_mixing none \
  --theta_init random_sphere \
  --theta_init_scale 0.3535533905932738 \
  --context_length 512 \
  --dropout 0.1 \
  --lr 3e-4 \
  --lr_schedule cosine \
  --lr_schedule_steps 250000 \
  --warmup_steps 0 \
  --lr_hold_steps 50000 \
  --min_lr 1e-4 \
  --weight_decay 0.01 \
  --precision bf16 \
  --attention lgma_residual \
  --num_heads 16 \
  --num_base_heads 4 \
  --value_transform lie \
  --fuse_base_qkv \
  --fold_value_transform_into_output \
  --sdpa_gqa_mode auto \
  "${resume_args[@]}" &

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
