#!/usr/bin/env bash
set -euo pipefail

mkdir -p results smoke_data

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Container CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the assigned GPU")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Extracting TinyStories..."
tar -xzf tinystory-v1-20260720.tar.gz

TRAIN_FULL=$(find . -name 'TinyStoriesV2-GPT4-train.txt' -print -quit)
VALID_FULL=$(find . -name 'TinyStoriesV2-GPT4-valid.txt' -print -quit)

if [[ -z "$TRAIN_FULL" || -z "$VALID_FULL" ]]; then
    echo "TinyStories files were not found after extraction"
    find . -maxdepth 3 -type f
    exit 1
fi

echo "Full training file: $TRAIN_FULL"
echo "Full validation file: $VALID_FULL"

# Keep the first scheduler test small and predictable.
head -c 20000000 "$TRAIN_FULL" > smoke_data/train.txt
head -c 2000000 "$VALID_FULL" > smoke_data/valid.txt

wc -c smoke_data/train.txt smoke_data/valid.txt

python experiments/train_tinystories.py \
  --data_path smoke_data/train.txt \
  --val_data_path smoke_data/valid.txt \
  --attention lgma_multibase \
  --num_heads 8 \
  --num_base_heads 2 \
  --base_dim 32 \
  --num_generators 4 \
  --device cuda \
  --precision bf16 \
  --batch_size 8 \
  --steps 20 \
  --log_every 5 \
  --eval_every 20 \
  --eval_batches 2 \
  --save_every 20 \
  --output_dir results \
  --wandb_mode disabled

echo "Smoke test completed successfully"
find results -maxdepth 2 -type f -ls
