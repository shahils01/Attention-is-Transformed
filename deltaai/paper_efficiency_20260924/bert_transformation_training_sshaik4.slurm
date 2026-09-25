#!/usr/bin/env bash
# Matched BERT training efficiency for MHA and all three GT-MHA maps.
# Four cyclic rounds give each method every execution position once.
#SBATCH --job-name=bert-map-eff
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.err

set -euo pipefail
ROOT=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924
OUT=${ROOT}/results/bert_transformation_training

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-clean/bin/activate
export PYTHONPATH="${ROOT}/src"
export HF_HOME=/work/hdd/biad/sshaik4/lgma_data/.hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUT}" "${ROOT}/logs"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail early if any transformation cannot complete a forward/backward/update step.
python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "MHA" "GT-MHA" "GT-MHA (quadratic)" "GT-MHA (exact exponential)" \
    --batch_size 2 --sequence_length 128 --gradient_accumulation_steps 1 \
    --precision bf16 --use_sdpa --warmup 0 --repeats 1 \
    --seed 20260924 --output "${OUT}/smoke.json"

run_round() {
    local round=$1
    shift
    python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
        --methods "$@" \
        --batch_size 32 --sequence_length 128 --gradient_accumulation_steps 2 \
        --precision bf16 --use_sdpa --warmup 3 --repeats 10 \
        --seed 20260924 --output "${OUT}/round_${round}.json"
}

run_round 01 "MHA" "GT-MHA" "GT-MHA (quadratic)" "GT-MHA (exact exponential)"
run_round 02 "GT-MHA" "GT-MHA (quadratic)" "GT-MHA (exact exponential)" "MHA"
run_round 03 "GT-MHA (quadratic)" "GT-MHA (exact exponential)" "MHA" "GT-MHA"
run_round 04 "GT-MHA (exact exponential)" "MHA" "GT-MHA" "GT-MHA (quadratic)"

# Keep profiler overhead outside the timed rounds. Operator FLOPs can omit
# fused SDPA and custom kernels; report them as a secondary diagnostic.
python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "MHA" "GT-MHA" "GT-MHA (quadratic)" "GT-MHA (exact exponential)" \
    --batch_size 32 --sequence_length 128 --gradient_accumulation_steps 2 \
    --precision bf16 --use_sdpa --warmup 1 --repeats 1 --profile_flops \
    --seed 20260924 --output "${OUT}/operator_flops.json"

echo BERT_TRANSFORMATION_TRAINING_COMPLETE
