#!/usr/bin/env bash
# Install the pinned vision dependencies into the existing project environment.
#
#   sbatch deltaai/setup_vision_env_sshaik4.slurm
#SBATCH --job-name=vision-env
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/sshaik4/Attention-is-Transformed-vision}
VENV_DIR=/work/hdd/biad/sshaik4/venvs/lgma-clean

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
python -m pip install -e ".[vision,tracking]"
python -c 'import timm, torch, webdataset; print(torch.__version__, timm.__version__, webdataset.__version__)'
