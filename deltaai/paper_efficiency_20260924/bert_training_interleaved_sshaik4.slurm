#!/usr/bin/env bash
# Balanced repeated-block benchmark for BERT training efficiency.
#SBATCH --job-name=bert-eff-inter
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.err

set -euo pipefail
ROOT=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924
OUT=${ROOT}/results/bert_training_interleaved

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

run_round() {
    local round=$1
    shift
    python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
        --methods "$@" \
        --batch_size 32 \
        --sequence_length 128 \
        --gradient_accumulation_steps 2 \
        --precision bf16 \
        --use_sdpa \
        --warmup 10 \
        --repeats 30 \
        --seed 20260924 \
        --output "${OUT}/round_${round}.json"
}

# Three repetitions of a balanced Williams design. Each method occupies every
# execution position equally and each ordered neighbor occurs equally often.
run_round 01 "MHA" "GQA" "GT-MHA" "Collaborative MHA"
run_round 02 "GQA" "Collaborative MHA" "MHA" "GT-MHA"
run_round 03 "Collaborative MHA" "GT-MHA" "GQA" "MHA"
run_round 04 "GT-MHA" "MHA" "Collaborative MHA" "GQA"
run_round 05 "MHA" "GQA" "GT-MHA" "Collaborative MHA"
run_round 06 "GQA" "Collaborative MHA" "MHA" "GT-MHA"
run_round 07 "Collaborative MHA" "GT-MHA" "GQA" "MHA"
run_round 08 "GT-MHA" "MHA" "Collaborative MHA" "GQA"
run_round 09 "MHA" "GQA" "GT-MHA" "Collaborative MHA"
run_round 10 "GQA" "Collaborative MHA" "MHA" "GT-MHA"
run_round 11 "Collaborative MHA" "GT-MHA" "GQA" "MHA"
run_round 12 "GT-MHA" "MHA" "Collaborative MHA" "GQA"

# Profile once, after all timed rounds, so profiler overhead cannot affect timing.
python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "MHA" "GQA" "Collaborative MHA" "GT-MHA" \
    --batch_size 32 --sequence_length 128 --gradient_accumulation_steps 2 \
    --precision bf16 --use_sdpa --warmup 3 --repeats 3 --profile_flops \
    --seed 20260924 --output "${OUT}/operator_flops.json"

echo BERT_INTERLEAVED_BENCHMARK_COMPLETE
