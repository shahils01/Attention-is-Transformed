#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${OSPOOL_RESULTS_DIR:-ospool_training_results}"
SEGMENT_DURATION="${OSPOOL_SEGMENT_DURATION:-4h}"
WANDB_MODE="${WANDB_MODE:-disabled}"

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Segment started: $(date --iso-8601=seconds)"
echo "Segment duration: ${SEGMENT_DURATION}"
nvidia-smi

archive="$(find . -maxdepth 1 -type f -name 'tinystory-v1-*.tar.gz' -print -quit)"
if [[ -z "${archive}" ]]; then
    echo "TinyStories archive was not transferred" >&2
    exit 2
fi

mkdir -p extracted_data "${RESULTS_DIR}"
tar -xzf "${archive}" -C extracted_data

train_path="$(find extracted_data -type f -name 'TinyStoriesV2-GPT4-train.txt' -print -quit)"
val_path="$(find extracted_data -type f -name 'TinyStoriesV2-GPT4-valid.txt' -print -quit)"
if [[ -z "${train_path}" || -z "${val_path}" ]]; then
    echo "Could not locate TinyStories train/validation files" >&2
    find extracted_data -type f -print >&2
    exit 2
fi

latest_valid_checkpoint() {
    local candidate
    while IFS= read -r candidate; do
        if python - "${candidate}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
required = {"step", "model_state", "optimizer_state", "model_config", "args"}
if not required.issubset(checkpoint):
    raise SystemExit(1)
PY
        then
            printf '%s\n' "${candidate}"
            return 0
        fi
        echo "Removing incomplete checkpoint: ${candidate}" >&2
        rm -f -- "${candidate}"
    done < <(
        find "${RESULTS_DIR}" -maxdepth 1 -type f \
            -name 'checkpoint_step_*.pt' -print | sort -V -r
    )
    return 1
}

prune_old_checkpoints() {
    local index=0
    local checkpoint
    while IFS= read -r checkpoint; do
        index=$((index + 1))
        if (( index > 2 )); then
            rm -f -- "${checkpoint}"
        fi
    done < <(
        find "${RESULTS_DIR}" -maxdepth 1 -type f \
            -name 'checkpoint_step_*.pt' -print | sort -V -r
    )
}

resume_args=()
if checkpoint="$(latest_valid_checkpoint)"; then
    echo "Resuming from ${checkpoint}"
    resume_args=(--resume_checkpoint "${checkpoint}")
else
    echo "No checkpoint found; starting at step 0"
fi

set +e
timeout --signal=TERM --kill-after=2m "${SEGMENT_DURATION}" \
    python ospool/train_tinystories_compat.py \
      --data_path "${train_path}" \
      --val_data_path "${val_path}" \
      --device cuda \
      --batch_size 4 \
      --grad_accum_steps 32 \
      --steps 500000 \
      --log_every 100 \
      --eval_every 1000 \
      --save_every 250 \
      --output_dir "${RESULTS_DIR}" \
      --diagnostic_every 1000 \
      --diagnostic_batches 2 \
      --wandb_project lgma \
      --wandb_run_name tinystory_lgma_multibase_b4_h16_ospool \
      --wandb_group tinystories \
      --wandb_tags tinystory,lgma,ValueLie,multibase,b4,h16,ospool \
      --wandb_mode "${WANDB_MODE}" \
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
      --attention lgma_multibase \
      --num_heads 16 \
      --num_base_heads 4 \
      --value_transform lie \
      "${resume_args[@]}"
training_status=$?
set -e

prune_old_checkpoints

if [[ ${training_status} -eq 124 ]]; then
    if checkpoint="$(latest_valid_checkpoint)"; then
        echo "Segment timed out normally; checkpoint ready: ${checkpoint}"
        echo "Requesting an HTCondor checkpoint transfer and restart"
        exit 85
    fi
    echo "Segment timed out before producing a valid checkpoint" >&2
    exit 2
fi

echo "Training process exited with status ${training_status}"
exit "${training_status}"
