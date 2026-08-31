#!/usr/bin/env bash
# Download the authorized ImageNet localization competition archive on arai3.
# Requires /u/arai3/.kaggle/access_token with mode 600 and accepted rules.
#
#   sbatch deltaai/download_imagenet_archive_arai3.slurm
#SBATCH --job-name=imagenet-download
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bifg/arai3/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/logs/%x_%j.err

set -euo pipefail

VENV_DIR=${VENV_DIR:-/work/hdd/bifg/arai3/venvs/lgma-vision}
ARCHIVE_DIR=${ARCHIVE_DIR:-/work/hdd/bifg/arai3/lgma_data/imagenet/archive}
ARCHIVE=${ARCHIVE_DIR}/imagenet-object-localization-challenge.zip
KAGGLE_CONFIG_DIR=${KAGGLE_CONFIG_DIR:-/u/arai3/.kaggle}
KAGGLE_ACCESS_TOKEN=${KAGGLE_CONFIG_DIR}/access_token

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
mkdir -p "${ARCHIVE_DIR}" /work/hdd/bifg/arai3/logs

test -f "${KAGGLE_ACCESS_TOKEN}"
test "$(stat -c %a "${KAGGLE_ACCESS_TOKEN}")" = "600"
if [[ -f "${ARCHIVE}" ]]; then
  test "$(stat -c %s "${ARCHIVE}")" = "166496672546"
  echo "Validated archive already exists: ${ARCHIVE}"
  exit 0
fi

# KGAT tokens require the current CLI. SDK 0.1.37 has a malformed wheel;
# 0.1.36 is the compatible release verified on DeltaAI.
python -m pip install kaggle==2.2.4
# Reinstall after Kaggle because replacing the legacy CLI can remove files
# shared by the SDK while leaving its package metadata behind.
python -m pip install --force-reinstall --no-deps kagglesdk==0.1.36
export KAGGLE_CONFIG_DIR
kaggle competitions download \
  --competition imagenet-object-localization-challenge \
  --path "${ARCHIVE_DIR}"
test -f "${ARCHIVE}"
test "$(stat -c %s "${ARCHIVE}")" = "166496672546"
echo "Validated archive download: ${ARCHIVE}"
