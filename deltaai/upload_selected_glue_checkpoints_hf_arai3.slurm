#!/usr/bin/env bash
# Build, upload, and verify a lean archive of the 36 validation-selected GLUE checkpoints.
# No source run files are removed by this job.
#SBATCH --job-name=hf-glue-retain
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bifg/arai3/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/work/hdd/bifg/arai3
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
JOB_DIR=${PROJECT_DIR}/lgma_jobs
RUN_ROOT=${PROJECT_DIR}/lgma_runs
STAGING_ROOT=${RUN_ROOT}/hf_glue_selected_export_v1
BUILD_SCRIPT=${JOB_DIR}/build_glue_hf_retention_export.py
VERIFY_SCRIPT=${JOB_DIR}/verify_glue_hf_retention_export.py
MHA_GT_SELECTION=${RUN_ROOT}/official_glue_submissions_v3/selected_checkpoints_source.json
BASELINE_SELECTION=${RUN_ROOT}/official_glue_submissions_baselines_v1/selected_checkpoints.json
MHA_PROVENANCE=${RUN_ROOT}/official_glue_submissions_v3/GT-MHA_GLUE_official_mha_provenance.json
GT_PROVENANCE=${RUN_ROOT}/official_glue_submissions_v3/GT-MHA_GLUE_official_gt_mha_residual_provenance.json
GQA_PROVENANCE=${RUN_ROOT}/official_glue_submissions_baselines_v1/GT-MHA_GLUE_official_gqa_provenance.json
COLLAB_PROVENANCE=${RUN_ROOT}/official_glue_submissions_baselines_v1/GT-MHA_GLUE_official_collaborative_provenance.json

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
mkdir -p "${PROJECT_DIR}/logs"
for path in "${BUILD_SCRIPT}" "${VERIFY_SCRIPT}" "${MHA_GT_SELECTION}" "${BASELINE_SELECTION}" \
  "${MHA_PROVENANCE}" "${GT_PROVENANCE}" "${GQA_PROVENANCE}" "${COLLAB_PROVENANCE}"; do
  test -f "${path}" || { echo "Missing required file: ${path}" >&2; exit 1; }
done
test ! -e "${STAGING_ROOT}" || { echo "Refusing to overwrite ${STAGING_ROOT}" >&2; exit 1; }

python "${BUILD_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --selection-manifest "${MHA_GT_SELECTION}" \
  --selection-manifest "${BASELINE_SELECTION}" \
  --provenance "${MHA_PROVENANCE}" \
  --provenance "${GT_PROVENANCE}" \
  --provenance "${GQA_PROVENANCE}" \
  --provenance "${COLLAB_PROVENANCE}" \
  --output-root "${STAGING_ROOT}"

hf upload-large-folder shahils/GT-MHA "${STAGING_ROOT}" \
  --repo-type model --num-workers 4 --exclude ".cache/**"

python "${VERIFY_SCRIPT}" \
  --repo-id shahils/GT-MHA --staging-root "${STAGING_ROOT}"

echo "Selected GLUE checkpoint upload and verification completed at $(date -Is)"
