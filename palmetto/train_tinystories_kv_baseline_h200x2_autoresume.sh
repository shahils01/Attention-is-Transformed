#!/usr/bin/env bash
# Submit with: sbatch --export=ALL,BASELINE_TYPE=mqa ...  or BASELINE_TYPE=gqa,NUM_KV_HEADS=4
#SBATCH --job-name=kvbaseline-h200x2
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/shahils/logs/%x_%j.out
#SBATCH --error=/home/shahils/logs/%x_%j.err

REPO_DIR=/home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
DATA_DIR=/scratch/shahils/lgma_data/tinystory
TOTAL_STEPS=250000
BASELINE_TYPE=${BASELINE_TYPE:?Set BASELINE_TYPE to mqa or gqa}
SEED=${SEED:-0}
case "${BASELINE_TYPE}" in
  mqa) ATTENTION_ARGS=(--attention mqa); BASELINE_TAG=mqa-h16-kv1 ;;
  gqa) NUM_KV_HEADS=${NUM_KV_HEADS:-4}; ATTENTION_ARGS=(--attention gqa --num_kv_heads "${NUM_KV_HEADS}"); BASELINE_TAG=gqa-h16-kv${NUM_KV_HEADS} ;;
  *) echo "Unsupported BASELINE_TYPE: ${BASELINE_TYPE}"; exit 2 ;;
esac
OUTPUT_DIR=/scratch/shahils/lgma_runs/tinystories_${BASELINE_TAG}_a10080_ddp4_gb2048_seed${SEED}
RUN_NAME=tinystory_${BASELINE_TAG}_a10080_ddp4_gb2048_seed${SEED}

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -uo pipefail
export OMP_NUM_THREADS=2
export WANDB_RUN_ID=${RUN_NAME}-20260815
export WANDB_RESUME=allow
mkdir -p "${OUTPUT_DIR}" /home/shahils/logs
cd "${REPO_DIR}"

latest_checkpoint="$({ python - "${OUTPUT_DIR}" <<'PY'
import re, sys
from pathlib import Path
paths = [(int(match.group(1)), path) for path in Path(sys.argv[1]).glob('checkpoint_step_*.pt') if (match := re.fullmatch(r'checkpoint_step_(\d+)\.pt', path.name))]
if paths: print(max(paths)[1])
PY
} )"
[[ -n "${latest_checkpoint}" && -f "${latest_checkpoint}" ]] || { echo "No resume checkpoint found; refusing fresh start."; exit 2; }
echo "Resuming ${BASELINE_TYPE} from ${latest_checkpoint}"

trap 'scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1
torchrun --standalone --nproc_per_node=2 experiments/train_tinystories.py \
  --data_path "${DATA_DIR}/TinyStoriesV2-GPT4-train.txt" --val_data_path "${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt" \
  --device cuda --batch_size 256 --grad_accum_steps 4 --steps "${TOTAL_STEPS}" \
  --log_every 100 --eval_every 1000 --save_every 100 --keep_last_checkpoints 3 --milestone_checkpoint_every 50000 \
  --output_dir "${OUTPUT_DIR}" --diagnostic_every 1000 --diagnostic_batches 2 \
  --wandb_project lgma --wandb_run_name "${RUN_NAME}" --wandb_group tinystories \
  --wandb_tags "tinystory,baseline,${BASELINE_TYPE},h200,ddp2,b256,ga4,gb2048,seed${SEED}" --wandb_mode online --wandb_dir "${OUTPUT_DIR}" \
  --d_model 1024 --num_layers 12 --head_dim 64 --base_dim 64 --value_dim 64 --context_length 512 --dropout 0.1 \
  --lr 3e-4 --lr_schedule cosine --lr_schedule_steps 500000 --warmup_steps 0 --lr_hold_steps 50000 --min_lr 1e-4 \
  --weight_decay 0.01 --precision bf16 --num_heads 16 --seed "${SEED}" \
  "${ATTENTION_ARGS[@]}" --resume_checkpoint "${latest_checkpoint}"
