#!/usr/bin/env bash
# Balanced interleaved comparison of the three GT-MHA transformation mappings.
# Each mapping occupies every execution position twice across six rounds.
#SBATCH --job-name=tiny-map-eff
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.err

set -euo pipefail
ROOT=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924
CHECKPOINT_ROOT="/work/hdd/biad/sshaik4/lgma_checkpoints/paper_gt_mha/TinyStories-v2 Checkpoints"
OUT=${ROOT}/results/tinystories_mapping_interleaved

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-clean/bin/activate
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUT}" "${ROOT}/logs"

checkpoint_for() {
    case "$1" in
        "First-order") printf '%s\n' "${CHECKPOINT_ROOT}/GT-MHA residual/checkpoint_step_250000.pt" ;;
        "Second-order") printf '%s\n' "${CHECKPOINT_ROOT}/GT-MHA quad/checkpoint_step_250000.pt" ;;
        "Exact exponential") printf '%s\n' "${CHECKPOINT_ROOT}/GT-MHA exact/checkpoint_step_250000.pt" ;;
        *) printf 'Unknown mapping: %s\n' "$1" >&2; return 2 ;;
    esac
}

slug_for() {
    case "$1" in
        "First-order") printf '%s\n' first_order ;;
        "Second-order") printf '%s\n' second_order ;;
        "Exact exponential") printf '%s\n' exact_exponential ;;
        *) return 2 ;;
    esac
}

run_round() {
    local round=$1
    shift
    local position=0
    for label in "$@"; do
        position=$((position + 1))
        local checkpoint
        local slug
        checkpoint=$(checkpoint_for "${label}")
        slug=$(slug_for "${label}")
        test -s "${checkpoint}"
        python "${ROOT}/experiments/benchmark_tinystories_training.py" \
            --checkpoint "${checkpoint}" \
            --model_label "${label}" \
            --batch_size 256 \
            --context_length 512 \
            --precision bf16 \
            --warmup 3 \
            --repeats 10 \
            --fuse_base_qkv \
            --fold_value_transform_into_output \
            --sdpa_gqa_mode auto \
            --output "${OUT}/round_${round}_position_${position}_${slug}.json"
    done
}

# Balanced six-sequence crossover: each mapping appears twice in each position.
run_round 01 "First-order" "Second-order" "Exact exponential"
run_round 02 "Second-order" "Exact exponential" "First-order"
run_round 03 "Exact exponential" "First-order" "Second-order"
run_round 04 "Exact exponential" "Second-order" "First-order"
run_round 05 "First-order" "Exact exponential" "Second-order"
run_round 06 "Second-order" "First-order" "Exact exponential"

echo TINYSTORIES_MAPPING_INTERLEAVED_BENCHMARK_COMPLETE
