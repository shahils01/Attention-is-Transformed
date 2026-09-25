#!/usr/bin/env bash
# Run inside an sshaik4 DeltaAI interactive GH200 allocation.
set -euo pipefail

ROOT=/work/hdd/biad/sshaik4/lgma_jobs/tinystories_multiseed_eval_20260924
DATA=/work/hdd/biad/sshaik4/lgma_data/tinystories

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-clean/bin/activate
export PYTHONPATH="$ROOT/source"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/results" "$ROOT/logs"

run_one() {
  local method=$1 seed=$2
  local label="${method}_seed_${seed}"
  local checkpoint="$ROOT/checkpoints/${label}.pt"
  local output="$ROOT/results/$label"
  test -s "$checkpoint"
  if [[ -s "$output/summary.json" ]]; then
    echo "Already complete: $label"
    return
  fi
  echo "Evaluating $label"
  python -u "$ROOT/evaluate.py" \
    --checkpoint "$checkpoint" \
    --data-dir "$DATA" \
    --output-dir "$output" \
    --method "$method" \
    --seed "$seed" \
    > "$ROOT/logs/${label}.log" 2>&1
  tail -n 18 "$ROOT/logs/${label}.log"
}

for seed in 0 100 200; do run_one GT-MHA "$seed"; done
for seed in 0 100 200; do run_one MHA "$seed"; done
echo MULTISEED_FULL_VALIDATION_COMPLETE
