#!/usr/bin/env bash
#SBATCH --job-name=ts-all-ckpt-infer
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/sshaik4/Attention-is-Transformed}
PROJECT_DIR=${PROJECT_DIR:-/work/hdd/biad/sshaik4}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/lgma_data/tinystories}
VENV_DIR=${VENV_DIR:-${PROJECT_DIR}/venvs/lgma}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/ckpts}
PROMPTS_FILE=${PROMPTS_FILE:-${REPO_DIR}/benchmarks/tinystories_100_prompts.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/lgma_runs/tinystories_all_checkpoints_100_x5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-300}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-20}
SEED=${SEED:-0}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-5}
CHECKPOINT_GLOB=${CHECKPOINT_GLOB:-**/checkpoint*.pt}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"

python experiments/compare_tinystories_prompts.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint_glob "${CHECKPOINT_GLOB}" \
  --prompts_file "${PROMPTS_FILE}" \
  --data_path "${DATA_DIR}/TinyStoriesV2-GPT4-train.txt" \
  --val_data_path "${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt" \
  --device cuda \
  --precision bf16 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --seed "${SEED}" \
  --samples_per_prompt "${SAMPLES_PER_PROMPT}" \
  --output_dir "${OUTPUT_DIR}"
