#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Started: $(date --iso-8601=seconds)"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("HTCondor assigned the job a GPU, but PyTorch cannot see it")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

mkdir -p results
python experiments/train_synthetic.py \
  --task copy \
  --attention lgma \
  --steps 20 \
  --device cuda \
  --wandb_mode disabled \
  > results/gpu_smoke.jsonl

echo "Finished: $(date --iso-8601=seconds)"
