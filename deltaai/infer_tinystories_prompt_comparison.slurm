#!/usr/bin/env bash
#SBATCH --job-name=ts-lgma-mha-infer
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/sshaik4/Attention-is-Transformed}
PROJECT_DIR=${PROJECT_DIR:-/work/hdd/biad/sshaik4}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/lgma_data/tinystories}
VENV_DIR=${VENV_DIR:-${PROJECT_DIR}/venvs/lgma}
PROMPTS_FILE=${PROMPTS_FILE:-${REPO_DIR}/benchmarks/tinystories_100_prompts.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/lgma_runs/tinystories_lgma_mha_prompt_comparison}
LGMA_CHECKPOINT=${LGMA_CHECKPOINT:-${REPO_DIR}/ckpts/tinystories_lgma_quad_b4_h16_g8/checkpoint_step_250100.pt}
MHA_CHECKPOINT=${MHA_CHECKPOINT:-${REPO_DIR}/ckpts/tinystories_mha_h16/checkpoint_step_266200.pt}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-300}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-20}
SEED=${SEED:-0}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"

python experiments/compare_tinystories_prompts.py \
  --checkpoint "lgma=${LGMA_CHECKPOINT}" \
  --checkpoint "mha=${MHA_CHECKPOINT}" \
  --prompts_file "${PROMPTS_FILE}" \
  --data_path "${DATA_DIR}/TinyStoriesV2-GPT4-train.txt" \
  --val_data_path "${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt" \
  --device cuda \
  --precision bf16 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}"
