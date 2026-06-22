#!/usr/bin/env bash
set -euo pipefail

CALIBRATION_TEXT_FILE="${CALIBRATION_TEXT_FILE:?set CALIBRATION_TEXT_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/lgma_adapters/tinyllama_lgma_residual_stage1}"

python adapter_trainer/distill_tinyllama.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --calibration_text_file "$CALIBRATION_TEXT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --stage stage1 \
  --layers "${LAYERS:-all}" \
  --attention_variant "${ATTENTION_VARIANT:-lgma_residual}" \
  --generator_type "${GENERATOR_TYPE:-full}" \
  --num_generators "${NUM_GENERATORS:-4}" \
  --qk_num_base_heads "${QK_NUM_BASE_HEADS:-4}" \
  --value_num_base_heads "${VALUE_NUM_BASE_HEADS:-4}" \
  --sequence_length "${SEQUENCE_LENGTH:-512}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --steps_per_layer "${STEPS_PER_LAYER:-500}" \
  --lr "${LR:-2e-4}" \
  --device "${DEVICE:-cuda}" \
  --precision "${PRECISION:-bf16}"
