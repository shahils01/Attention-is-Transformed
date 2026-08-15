#!/usr/bin/env bash
#SBATCH --job-name=fw-edu-10bt-prep
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200g
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bicn/anayak2
VENV_DIR=${PROJECT_DIR}/venvs/lgma
CODE_SNAPSHOT=${CODE_SNAPSHOT:?Set CODE_SNAPSHOT to an immutable source snapshot}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the snapshot commit}
DATASET_ROOT=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT
INPUT_DIR=${DATASET_ROOT}/sample/10BT
OUTPUT_DIR=${PROJECT_DIR}/lgma_data/fineweb_edu_10BT_packed_v1
HF_HOME=${PROJECT_DIR}/.cache/huggingface
DATASET_REVISION=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export HF_HOME DATASET_ROOT

mkdir -p "${DATASET_ROOT}" "${OUTPUT_DIR}" "${PROJECT_DIR}/logs" "${HF_HOME}"
cd "${CODE_SNAPSHOT}"

actual_commit=$(git rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Source snapshot mismatch: expected ${EXPECTED_COMMIT}, found ${actual_commit}" >&2
  exit 2
fi

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    allow_patterns=["sample/10BT/*.parquet"],
    local_dir=os.environ["DATASET_ROOT"],
    max_workers=8,
)
PY

shopt -s nullglob
shards=("${INPUT_DIR}"/*.parquet)
if [[ ${#shards[@]} -ne 14 ]]; then
  echo "Expected 14 FineWeb-Edu 10BT Parquet shards, found ${#shards[@]}" >&2
  exit 3
fi

du -sh "${INPUT_DIR}"

python experiments/prepare_fineweb.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --phase all \
  --vocab_size 32768 \
  --validation_basis_points 10 \
  --test_basis_points 10 \
  --tokenizer_sample_modulus 50 \
  --batch_size 256

du -sh "${OUTPUT_DIR}"
