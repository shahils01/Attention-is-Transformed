#!/usr/bin/env bash
#SBATCH --job-name=bert-eff-train
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

set -euo pipefail
ROOT=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924

module load python/miniforge3_pytorch/2.12.0
source /work/hdd/biad/sshaik4/venvs/lgma-clean/bin/activate
export PYTHONPATH="${ROOT}/source"
export HF_HOME=/work/hdd/biad/sshaik4/lgma_data/.hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${ROOT}/results/bert_training" "${ROOT}/logs"

python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "MHA" "GQA" "Collaborative MHA" "GT-MHA" \
    --batch_size 32 \
    --sequence_length 512 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --warmup 3 \
    --repeats 10 \
    --output "${ROOT}/results/bert_training/pass1.json"

python "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "GT-MHA" "Collaborative MHA" "GQA" "MHA" \
    --batch_size 32 \
    --sequence_length 512 \
    --gradient_accumulation_steps 2 \
    --precision bf16 \
    --warmup 3 \
    --repeats 10 \
    --output "${ROOT}/results/bert_training/pass2.json"

echo "BERT_TRAINING_BENCHMARK_COMPLETE"
