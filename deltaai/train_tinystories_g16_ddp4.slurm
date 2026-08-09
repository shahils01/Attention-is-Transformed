#!/usr/bin/env bash
#SBATCH --job-name=lgma-g16-ddp4
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/u/sshaik4/Attention-is-Transformed
PROJECT_DIR=/work/hdd/biad/sshaik4
DATA_DIR=${PROJECT_DIR}/lgma_data/tinystories
NUM_GENERATORS=${NUM_GENERATORS:-16}
RUN_NAME=tinystory_lgma_quad_b4_h16_g${NUM_GENERATORS}_gh200_ddp4_gb2048
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/tinystories_lgma_quad_b4_h16_g${NUM_GENERATORS}_gh200_ddp4_gb2048
VENV_DIR=${PROJECT_DIR}/venvs/lgma
TOTAL_STEPS=250000

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=8
export WANDB_RUN_ID=tinystory-lgma-quad-b4-h16-g${NUM_GENERATORS}-gh200-ddp4-gb2048-20260808
export WANDB_RESUME=allow

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
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
    echo "No checkpoint found; starting a fresh run."
fi

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
  --wandb_project lgma \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_group tinystories \
  --wandb_tags "tinystory,lgma,quad,ValueLie,multibase,b4,h16,g${NUM_GENERATORS},gh200,ddp4,b256,gb2048" \
  --wandb_mode online \
  --wandb_dir "${OUTPUT_DIR}" \
  --induced_metric_diversity_weight 0.0 \
  --metric_diversity_weight 0.0 \
  --d_model 1024 \
  --num_layers 12 \
  --head_dim 64 \
  --base_dim 64 \
  --value_dim 64 \
  --num_generators "${NUM_GENERATORS}" \
  --metric_beta 1.0 \
  --value_beta 1.0 \
  --no-stabilize_generators \
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
  --attention lgma_quad \
  --num_heads 16 \
  --num_base_heads 4 \
  --value_transform lie \
  "${resume_args[@]}"
