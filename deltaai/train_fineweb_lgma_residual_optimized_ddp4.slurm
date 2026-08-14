#!/usr/bin/env bash
#SBATCH --job-name=fw-lgma-res-opt
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=36:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/biad/sshaik4
PACKED_DATA=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT_packed_v1
VENV_DIR=${PROJECT_DIR}/venvs/lgma
CODE_SNAPSHOT=${CODE_SNAPSHOT:?Set CODE_SNAPSHOT to the immutable source snapshot}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-7aa685755cb7751b9cdd1e8a623060a4721aed8a}
TOTAL_STEPS=9512
SEED=${SEED:-0}
RUN_NAME=fineweb_edu_10BT_lgma_quad_b4_h16_g8_theta1_optimized_gh200_ddp4_gb2048_seed${SEED}_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=2
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${CODE_SNAPSHOT}"

actual_commit=$(git rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Source snapshot mismatch: expected ${EXPECTED_COMMIT}, found ${actual_commit}" >&2
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
  echo "No checkpoint found; starting a fresh optimized LGMA-residual run."
fi

torchrun --standalone --nproc_per_node=4 experiments/train_text_lm.py \
  --packed_data_dir "${PACKED_DATA}" \
  --device cuda \
  --batch_size 64 \
  --grad_accum_steps 8 \
  --steps "${TOTAL_STEPS}" \
  --log_every 10 \
  --eval_every 100 \
  --eval_batches 32 \
  --save_every 100 \
  --keep_last_checkpoints 3 \
  --milestone_checkpoint_every 1900 \
  --output_dir "${OUTPUT_DIR}" \
  --diagnostic_every 500 \
  --diagnostic_batches 2 \
  --wandb_project lgma-fineweb-edu \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_group fineweb_edu_10BT \
  --wandb_tags "fineweb_edu,lgma,residual,ValueLie,multibase,b4,h16,g8,optimized,gh200,ddp4,b64,gb2048,seed${SEED}" \
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
  --lr_schedule_steps "${TOTAL_STEPS}" \
  --warmup_steps 200 \
  --lr_hold_steps 0 \
  --min_lr 3e-5 \
  --weight_decay 0.01 \
  --precision bf16 \
  --seed "${SEED}" \
  --attention lgma_quad \
  --num_heads 16 \
  --num_base_heads 4 \
  --num_generators 8 \
  --theta_init_scale 1.0 \
  --metric_beta 1.0 \
  --value_beta 1.0 \
  --no-stabilize_generators \
  --value_transform lie \
  --metric_diversity_weight 0.0 \
  --induced_metric_diversity_weight 0.0 \
  --fuse_base_qkv \
  --fold_value_transform_into_output \
  --sdpa_gqa_mode auto \
  "${resume_args[@]}"
