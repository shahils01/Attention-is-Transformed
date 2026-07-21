#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

ARCHIVE="$(find . -maxdepth 1 -type f \
  -name 'tinystory-v1-*.tar.gz' -print -quit)"

if [[ -z "$ARCHIVE" ]]; then
    echo "TinyStories archive was not transferred"
    exit 1
fi

mkdir -p extracted_data hopper_probe_results
tar -xzf "$ARCHIVE" -C extracted_data

TRAIN_SOURCE="$(find extracted_data -type f \
  -name 'TinyStoriesV2-GPT4-train.txt' -print -quit)"
VAL_SOURCE="$(find extracted_data -type f \
  -name 'TinyStoriesV2-GPT4-valid.txt' -print -quit)"

if [[ -z "$TRAIN_SOURCE" || -z "$VAL_SOURCE" ]]; then
    echo "Could not locate TinyStories train/validation files"
    find extracted_data -type f
    exit 1
fi

# Enough data for a model-memory and throughput test.
head -c 100000000 "$TRAIN_SOURCE" > extracted_data/train_probe.txt
head -c 10000000 "$VAL_SOURCE" > extracted_data/valid_probe.txt

python experiments/train_tinystories.py \
  --data_path extracted_data/train_probe.txt \
  --val_data_path extracted_data/valid_probe.txt \
  --device cuda \
  --batch_size 4 \
  --grad_accum_steps 32 \
  --steps 100 \
  --log_every 10 \
  --eval_every 50 \
  --save_every 50 \
  --output_dir hopper_probe_results \
  --diagnostic_every 50 \
  --diagnostic_batches 1 \
  --wandb_mode disabled \
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
  --lr_hold_steps 50 \
  --min_lr 1e-4 \
  --weight_decay 0.01 \
  --precision bf16 \
  --attention lgma_multibase \
  --num_heads 16 \
  --num_base_heads 4 \
  --value_transform lie

echo "Finished: $(date --iso-8601=seconds)"
