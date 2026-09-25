#!/usr/bin/env bash
# Balanced repeated-block TinyStories prefill, decode, memory, and KV-cache benchmark.
#SBATCH --job-name=tiny-eff-inter
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
CHECKPOINTS=/work/hdd/biad/sshaik4/lgma_jobs/tinystories_ci_full_20260915
DATA=/work/hdd/biad/sshaik4/lgma_data/tinystories
OUT=${ROOT}/results/tinystories_inference_interleaved

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-clean/bin/activate
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUT}" "${ROOT}/logs"

run_model() {
    local round=$1
    local position=$2
    local label=$3
    local slug
    slug=$(printf '%s' "${label}" | tr '[:upper:] ' '[:lower:]_')
    python "${ROOT}/experiments/benchmark_tinystories_efficiency.py" \
        --checkpoint "${CHECKPOINTS}/${label}.pt" \
        --model_label "${label}" \
        --data_path "${DATA}/TinyStoriesV2-GPT4-train.txt" \
        --val_data_path "${DATA}/TinyStoriesV2-GPT4-valid.txt" \
        --batch_size 1 --precision bf16 \
        --warmup 3 --repeats 10 \
        --prefill_context_lengths 128 256 448 512 \
        --decode_context_lengths 128 256 448 \
        --decode_tokens 64 --skip_sliding \
        --fuse_base_qkv --fold_value_transform_into_output --sdpa_gqa_mode auto \
        --seed 20260924 \
        --output "${OUT}/round_${round}_position_${position}_${slug}.json"
}

run_round() {
    local round=$1
    shift
    local position=1
    for label in "$@"; do
        run_model "${round}" "${position}" "${label}"
        position=$((position + 1))
    done
}

# Five cyclic orders and their reverses balance execution position and direction.
run_round 01 "MHA" "GQA" "MQA" "Collaborative MHA" "GT-MHA residual"
run_round 02 "GQA" "MQA" "Collaborative MHA" "GT-MHA residual" "MHA"
run_round 03 "MQA" "Collaborative MHA" "GT-MHA residual" "MHA" "GQA"
run_round 04 "Collaborative MHA" "GT-MHA residual" "MHA" "GQA" "MQA"
run_round 05 "GT-MHA residual" "MHA" "GQA" "MQA" "Collaborative MHA"
run_round 06 "GT-MHA residual" "Collaborative MHA" "MQA" "GQA" "MHA"
run_round 07 "MHA" "GT-MHA residual" "Collaborative MHA" "MQA" "GQA"
run_round 08 "GQA" "MHA" "GT-MHA residual" "Collaborative MHA" "MQA"
run_round 09 "MQA" "GQA" "MHA" "GT-MHA residual" "Collaborative MHA"
run_round 10 "Collaborative MHA" "MQA" "GQA" "MHA" "GT-MHA residual"

echo TINYSTORIES_INTERLEAVED_INFERENCE_COMPLETE
