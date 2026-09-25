#!/usr/bin/env bash
#SBATCH --job-name=tiny-eff-infer
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bifg/arai3/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.err

set -euo pipefail
ROOT=/work/hdd/bifg/arai3/lgma_jobs/paper_efficiency_20260924
CHECKPOINTS=/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_full_20260915
DATA=/work/hdd/bifg/arai3/lgma_data/tinystories

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bifg/arai3/venvs/lgma-clean/bin/activate
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${ROOT}/results/tinystories_inference" "${ROOT}/logs"

run_model() {
    local pass=$1
    local label=$2
    local profile=$3
    local slug
    slug=$(printf '%s' "${label}" | tr '[:upper:] ' '[:lower:]_')
    extra=()
    if [[ "${profile}" == "yes" ]]; then
        extra+=(--profile_flops)
    fi
    python "${ROOT}/experiments/benchmark_tinystories_efficiency.py" \
        --checkpoint "${CHECKPOINTS}/${label}.pt" \
        --model_label "${label}" \
        --data_path "${DATA}/TinyStoriesV2-GPT4-train.txt" \
        --val_data_path "${DATA}/TinyStoriesV2-GPT4-valid.txt" \
        --batch_size 1 \
        --precision bf16 \
        --warmup 5 \
        --repeats 30 \
        --prefill_context_lengths 128 256 448 512 \
        --decode_context_lengths 128 256 448 \
        --decode_tokens 64 \
        --sliding_context_length 512 \
        --sliding_decode_tokens 128 \
        --output "${ROOT}/results/tinystories_inference/pass${pass}_${slug}.json" \
        "${extra[@]}"
}

# A second pass in reverse order exposes run-order drift on the same physical GPU.
for label in "MHA" "GQA" "MQA" "Collaborative MHA" "GT-MHA residual"; do
    run_model 1 "${label}" yes
done
for label in "GT-MHA residual" "Collaborative MHA" "MQA" "GQA" "MHA"; do
    run_model 2 "${label}" no
done

echo "TINYSTORIES_INFERENCE_BENCHMARK_COMPLETE"
