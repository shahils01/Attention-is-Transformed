#!/usr/bin/env bash
# Strictly reconstruct and load the four selected STS-B classifiers before HF publication.
#SBATCH --job-name=validate-stsb-hf
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bicn/anayak2/venvs/lgma-clean/bin/activate
export PYTHONPATH=/work/hdd/bicn/anayak2/repos/Attention-is-Transformed/src

ROOT=/work/hdd/bicn/anayak2/lgma_runs
LOADER=/work/hdd/bicn/anayak2/lgma_jobs/load_finetuned_glue_checkpoint.py
RUNS=(
  stsb_papersweep_v2_mha_lr5em5_seed46_final
  stsb_papersweep_v2_gqa_lr3em5_seed44_grid
  stsb_papersweep_v2_collaborative_lr5em5_seed45_final
  stsb_papersweep_v2_gt_mha_residual_lr5em5_seed46_final
)

scratch_dir=$(mktemp -d /tmp/validate-stsb-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "${scratch_dir}"' EXIT

for run in "${RUNS[@]}"; do
  echo "Strictly loading ${run}"
  local_dir=${scratch_dir}/${run}
  mkdir -p "${local_dir}"
  cp "${ROOT}/${run}/config.json" "${local_dir}/"
  cp "${ROOT}/${run}/bert_glue_manifest.json" "${local_dir}/"
  cp "${ROOT}/${run}/model.safetensors" "${local_dir}/"
  python "${LOADER}" "${local_dir}"
done
echo "All selected STS-B checkpoints passed strict reload validation."
