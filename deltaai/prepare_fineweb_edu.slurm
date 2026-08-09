#!/usr/bin/env bash
#SBATCH --job-name=fineweb-prep
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/u/sshaik4/Attention-is-Transformed
PROJECT_DIR=/work/hdd/biad/sshaik4
INPUT_DIR=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT/sample/10BT
OUTPUT_DIR=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT_packed_v1
VENV_DIR=${PROJECT_DIR}/venvs/lgma

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${REPO_DIR}"

python experiments/prepare_fineweb.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --phase all \
  --vocab_size 32768 \
  --validation_basis_points 10 \
  --test_basis_points 10 \
  --tokenizer_sample_modulus 50 \
  --batch_size 256
