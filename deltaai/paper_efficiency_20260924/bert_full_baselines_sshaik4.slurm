#!/usr/bin/env bash
# Seven cyclic rounds compare all BERT attention methods on one GH200.
# Every method occupies each execution position once.
#SBATCH --job-name=bert-all-eff
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924/logs/%x_%j.err

set -euo pipefail
ROOT=/work/hdd/biad/sshaik4/lgma_jobs/paper_efficiency_20260924
OUT=${ROOT}/results/bert_baseline_comparison
PYTHON=/sw/user/python/miniforge3-pytorch-2.12.0/bin/python

module load python/miniforge3_pytorch/2.12.0
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

"${PYTHON}" "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "MQA" --batch_size 2 --sequence_length 128 \
    --gradient_accumulation_steps 1 --precision bf16 --use_sdpa \
    --warmup 0 --repeats 1 --seed 20260924 --output "${OUT}/smoke_mqa.json"

methods=("MHA" "GQA" "MQA" "Collaborative MHA" "GT-MHA" \
         "GT-MHA (quadratic)" "GT-MHA (exact exponential)")
for offset in 0 1 2 3 4 5 6; do
    printf -v round '%02d' "$((offset + 1))"
    order=("${methods[@]:offset}" "${methods[@]:0:offset}")
    "${PYTHON}" "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
        --methods "${order[@]}" \
        --batch_size 32 --sequence_length 128 --gradient_accumulation_steps 2 \
        --precision bf16 --use_sdpa --warmup 3 --repeats 10 \
        --seed 20260924 --output "${OUT}/round_${round}.json"
done

"${PYTHON}" "${ROOT}/experiments/benchmark_bert_training_efficiency.py" \
    --methods "${methods[@]}" \
    --batch_size 32 --sequence_length 128 --gradient_accumulation_steps 2 \
    --precision bf16 --use_sdpa --warmup 1 --repeats 1 --profile_flops \
    --seed 20260924 --output "${OUT}/operator_flops.json"

echo BERT_ALL_METHODS_BENCHMARK_COMPLETE
