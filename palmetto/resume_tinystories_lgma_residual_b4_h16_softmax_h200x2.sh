#!/usr/bin/env bash
#SBATCH --job-name=lgma-res-b4h16-h200x2
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/shahils/logs/%x_%j.out
#SBATCH --error=/home/shahils/logs/%x_%j.err

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
DATA_DIR=/scratch/shahils/lgma_data/tinystory
OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100
TOTAL_STEPS=250000

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail

export OMP_NUM_THREADS=2
export WANDB_RUN_ID=4qu5fx41
export WANDB_RESUME=allow
mkdir -p "${OUTPUT_DIR}" /home/shahils/logs
cd "${REPO_DIR}"

if [[ -f "${OUTPUT_DIR}/checkpoint_final.pt" ]]; then
  echo "Final checkpoint already exists; training is complete."
  exit 0
fi

latest_checkpoint="$({ python - "${OUTPUT_DIR}" <<'PY'
import re
import sys
from pathlib import Path

candidates = []
for path in Path(sys.argv[1]).glob("checkpoint_step_*.pt"):
    match = re.fullmatch(r"checkpoint_step_(\d+)\.pt", path.name)
    if match:
        candidates.append((int(match.group(1)), path))
if candidates:
    print(max(candidates)[1])
PY
} )"

if [[ -z "${latest_checkpoint}" || ! -f "${latest_checkpoint}" ]]; then
  echo "No checkpoint found in ${OUTPUT_DIR}; refusing to start a fresh run." >&2
  exit 2
fi
echo "Resuming from ${latest_checkpoint} on 2 H200 GPUs through step ${TOTAL_STEPS}."

handle_time_limit() {
  echo "Received USR1; terminating distributed launch and requeueing ${SLURM_JOB_ID}."
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit $?
}
trap handle_time_limit USR1

torchrun --standalone --nproc_per_node=2 experiments/train_tinystories.py \
  --data_path "${DATA_DIR}/TinyStoriesV2-GPT4-train.txt" \
  --val_data_path "${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt" \
  --device cuda --batch_size 256 --grad_accum_steps 4 --steps "${TOTAL_STEPS}" \
  --log_every 100 --eval_every 1000 --save_every 100 \
  --keep_last_checkpoints 3 --milestone_checkpoint_every 50000 \
  --output_dir "${OUTPUT_DIR}" --diagnostic_every 1000 --diagnostic_batches 2 \
  --wandb_project lgma --wandb_run_name tinystory_lgma_residual_b4_h16_ospool \
  --wandb_group tinystories \
  --wandb_tags tinystory,lgma,ValueLie,residual,b4,h16,softmax,h200,ddp2,b256,ga4,gb2048 \
  --wandb_mode online --wandb_dir "${OUTPUT_DIR}" \
  --induced_metric_diversity_weight 0.0 --metric_diversity_weight 0.0 \
  --d_model 1024 --num_layers 12 --head_dim 64 --base_dim 64 --value_dim 64 \
  --num_generators 8 --generator_mixing softmax --metric_beta 1.0 --value_beta 1.0 \
  --no-stabilize_generators --context_length 512 --dropout 0.1 \
  --lr 3e-4 --lr_schedule cosine --warmup_steps 0 --lr_hold_steps 50000 --min_lr 1e-4 \
  --weight_decay 0.01 --precision bf16 --attention lgma_residual --num_heads 16 \
  --num_base_heads 4 --value_transform lie --seed 0 \
  --resume_checkpoint "${latest_checkpoint}" &

TRAIN_PID=$!
wait "${TRAIN_PID}"
status=$?
trap - USR1
exit "${status}"
