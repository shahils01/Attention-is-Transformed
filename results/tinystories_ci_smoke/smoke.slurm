#!/bin/bash
#SBATCH --job-name=tiny-ci-smoke
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_smoke_20260915
#SBATCH --output=/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_smoke_20260915/%j.out
#SBATCH --error=/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_smoke_20260915/%j.err
set -eo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bifg/arai3/venvs/lgma-clean/bin/activate
export PYTHONPATH=/u/arai3/Attention-is-Transformed/src
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python smoke.py
