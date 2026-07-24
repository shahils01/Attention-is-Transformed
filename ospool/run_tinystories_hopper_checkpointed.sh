#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${OSPOOL_RESULTS_DIR:-ospool_training_results}"
SEGMENT_DURATION="${OSPOOL_SEGMENT_DURATION:-10m}"
WANDB_MODE="${1:-${WANDB_MODE:-disabled}}"

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

case "${WANDB_MODE}" in
    disabled)
        echo "W&B tracking is disabled"
        ;;
    offline|online)
        wandb_bundle="$(find . -maxdepth 1 -type f -name 'wandb-py*.tar.gz' -print -quit)"
        if [[ -z "${wandb_bundle}" ]]; then
            echo "W&B dependency bundle was not transferred" >&2
            exit 2
        fi

        mkdir -p ospool_python_deps
        tar -xzf "${wandb_bundle}" -C ospool_python_deps
        export PYTHONPATH="${PWD}/ospool_python_deps/python${PYTHONPATH:+:${PYTHONPATH}}"

        if [[ "${WANDB_MODE}" == "online" ]]; then
            key_file="ospool/.wandb_api_key"
            if [[ ! -s "${key_file}" ]]; then
                echo "W&B API key file was not transferred: ${key_file}" >&2
                exit 2
            fi
            WANDB_API_KEY="$(tr -d '\r\n' < "${key_file}")"
            if [[ -z "${WANDB_API_KEY}" ]]; then
                echo "W&B API key file is empty" >&2
                exit 2
            fi
            export WANDB_API_KEY
            echo "Loaded a W&B API key with ${#WANDB_API_KEY} characters"
        fi

        wandb_run_id_file="${RESULTS_DIR}/wandb_run_id"
        if [[ ! -s "${wandb_run_id_file}" ]]; then
            python -c 'import secrets; print(secrets.token_hex(8))' > "${wandb_run_id_file}"
        fi
        WANDB_RUN_ID="$(tr -d '\r\n' < "${wandb_run_id_file}")"
        export WANDB_RUN_ID
        export WANDB_RESUME=allow
        export WANDB_CACHE_DIR="${PWD}/${RESULTS_DIR}/.wandb_cache"
        export WANDB_CONFIG_DIR="${PWD}/${RESULTS_DIR}/.wandb_config"
        mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

        python -c 'import wandb; print(f"W&B ready: {wandb.__version__}")'
        echo "W&B mode: ${WANDB_MODE}; resumable run ID: ${WANDB_RUN_ID}"
        ;;
    *)
        echo "Unsupported W&B mode: ${WANDB_MODE}" >&2
        exit 2
        ;;
esac

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
      --device cuda:0 \
      --batch_size 256 \
      --grad_accum_steps 8 \
      --steps 500000 \
      --log_every 100 \
      --eval_every 1000 \
      --save_every 100 \
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
