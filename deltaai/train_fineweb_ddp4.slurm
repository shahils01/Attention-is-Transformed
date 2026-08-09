#!/usr/bin/env bash
#SBATCH --job-name=fw-train
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

PROJECT_DIR=/work/hdd/biad/sshaik4
PACKED_DATA=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT_packed_v1
VENV_DIR=${PROJECT_DIR}/venvs/lgma
CODE_SNAPSHOT=${CODE_SNAPSHOT:?Set CODE_SNAPSHOT to an immutable source snapshot}
ATTENTION_TYPE=${ATTENTION_TYPE:?Set ATTENTION_TYPE to mha, collaborative, or lgma_quad}
SEED=${SEED:-0}
TOTAL_STEPS=9512

case "${ATTENTION_TYPE}" in
  mha)
    RUN_KEY=mha_h16
    ATTENTION_ARGS=(--attention mha --num_heads 16)
    TAGS=fineweb_edu,baseline,mha,h16,gh200,ddp4,b64,gb2048,seed${SEED}
    ;;
  collaborative)
    RUN_KEY=collaborative_h16
    ATTENTION_ARGS=(--attention collaborative --num_heads 16)
    TAGS=fineweb_edu,baseline,collaborative,h16,gh200,ddp4,b64,gb2048,seed${SEED}
    ;;
  lgma_quad)
    NUM_GENERATORS=${NUM_GENERATORS:?Set NUM_GENERATORS for LGMA}
    RUN_KEY=lgma_quad_b4_h16_g${NUM_GENERATORS}
    ATTENTION_ARGS=(
      --attention lgma_quad
      --num_heads 16
      --num_base_heads 4
      --num_generators "${NUM_GENERATORS}"
      --metric_beta 1.0
      --value_beta 1.0
      --no-stabilize_generators
      --value_transform lie
      --metric_diversity_weight 0.0
      --induced_metric_diversity_weight 0.0
    )
    TAGS=fineweb_edu,lgma,quad,ValueLie,multibase,b4,h16,g${NUM_GENERATORS},gh200,ddp4,b64,gb2048,seed${SEED}
    ;;
  *)
    echo "Unsupported ATTENTION_TYPE=${ATTENTION_TYPE}" >&2
    exit 2
    ;;
esac

RUN_NAME=fineweb_edu_10BT_${RUN_KEY}_gh200_ddp4_gb2048_seed${SEED}_v1
OUTPUT_DIR=${PROJECT_DIR}/lgma_runs/${RUN_NAME}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=8
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${CODE_SNAPSHOT}"

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
  --wandb_project lgma \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_group fineweb_edu_10BT \
  --wandb_tags "${TAGS}" \
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
  "${ATTENTION_ARGS[@]}" \
  "${resume_args[@]}"
