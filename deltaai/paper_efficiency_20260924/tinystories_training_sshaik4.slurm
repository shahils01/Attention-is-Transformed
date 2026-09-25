#!/usr/bin/env bash
#SBATCH --job-name=tiny-eff-train
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.err

set -uo pipefail
ROOT=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924
CHECKPOINTS=/work/hdd/biad/sshaik4/lgma_jobs/tinystories_ci_full_20260915

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-clean/bin/activate
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${ROOT}/results/tinystories_training" "${ROOT}/logs"

run_model() {
    local pass=$1
    local label=$2
    local slug
    slug=$(printf '%s' "${label}" | tr '[:upper:] ' '[:lower:]_')
    python "${ROOT}/experiments/benchmark_tinystories_training.py" \
        --checkpoint "${CHECKPOINTS}/${label}.pt" \
        --model_label "${label}" \
        --batch_size 256 \
        --context_length 512 \
        --precision bf16 \
        --warmup 3 \
        --repeats 10 \
        --fuse_base_qkv \
        --fold_value_transform_into_output \
        --sdpa_gqa_mode auto \
        --output "${ROOT}/results/tinystories_training/pass${pass}_${slug}.json"
    status=$?
    if [[ ${status} -ne 0 ]]; then
        printf '{"model_label":"%s","pass":%s,"status":"failed","exit_code":%s}\n' \
            "${label}" "${pass}" "${status}" \
            > "${ROOT}/results/tinystories_training/pass${pass}_${slug}_failure.json"
    fi
}

for label in "GT-MHA residual" "MHA" "Collaborative MHA" "GQA" "MQA"; do
    run_model 1 "${label}"
done
for label in "MQA" "GQA" "Collaborative MHA" "MHA" "GT-MHA residual"; do
    run_model 2 "${label}"
done

echo "TINYSTORIES_TRAINING_BENCHMARK_COMPLETE"
