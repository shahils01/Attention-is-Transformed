#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/hdd/bicn/anayak2/lgma_jobs/tinystories_multiseed_eval_20260924
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/allocation.log") 2>&1

for spec in GT-MHA_seed_0.pt:1495254180 GT-MHA_seed_100.pt:1495254180 GT-MHA_seed_200.pt:1495254180 MHA_seed_0.pt:1825428082 MHA_seed_100.pt:1825428082 MHA_seed_200.pt:1825428082; do
  name=${spec%%:*}
  expected=${spec##*:}
  actual=$(stat -c %s "$ROOT/checkpoints/$name")
  if [[ "$actual" != "$expected" ]]; then
    echo "Checkpoint size mismatch: $name ($actual != $expected)" >&2
    exit 1
  fi
done

echo "Requesting anayak2 interactive GH200 allocation"
salloc --account=bicn-dtai-gh --partition=ghx4 --nodes=1 --ntasks=1 \
  --gpus-per-node=1 --cpus-per-task=4 --mem=32G --time=02:00:00 \
  srun --ntasks=1 --pty bash -lc "bash $ROOT/run_interactive_anayak2.sh"
