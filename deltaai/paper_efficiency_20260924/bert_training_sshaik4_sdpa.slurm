#!/usr/bin/env bash
#SBATCH --job-name=bert-eff-sdpa
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
mkdir -p "${ROOT}/results/bert_training_sdpa" "${ROOT}/logs"

# Validate the native GT-MHA SDPA forward/backward path before the timed runs.
python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "GT-MHA" \
    --batch_size 2 \
    --sequence_length 128 \
    --gradient_accumulation_steps 1 \
    --precision bf16 \
    --use_sdpa \
    --warmup 1 \
    --repeats 1 \
    --output "${ROOT}/results/bert_training_sdpa/smoke.json"

python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "MHA" "GQA" "Collaborative MHA" "GT-MHA" \
    --batch_size 32 \
    --sequence_length 128 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --use_sdpa \
    --warmup 10 \
    --repeats 50 \
    --profile_flops \
    --output "${ROOT}/results/bert_training_sdpa/pass1.json"

python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "GT-MHA" "Collaborative MHA" "GQA" "MHA" \
    --batch_size 32 \
    --sequence_length 128 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --use_sdpa \
    --warmup 10 \
    --repeats 50 \
    --output "${ROOT}/results/bert_training_sdpa/pass2.json"

echo "BERT_SDPA_BENCHMARK_COMPLETE"
