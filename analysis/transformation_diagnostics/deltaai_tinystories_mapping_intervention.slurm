#!/usr/bin/env bash
#SBATCH --job-name=tiny-map-intervene
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/hdd/bifg/arai3/lgma_jobs/tinystories_mapping_intervention_20260923
#SBATCH --output=/work/hdd/bifg/arai3/lgma_jobs/tinystories_mapping_intervention_20260923/%j.out
#SBATCH --error=/work/hdd/bifg/arai3/lgma_jobs/tinystories_mapping_intervention_20260923/%j.err

set -eo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/hdd/bifg/arai3/venvs/lgma-clean/bin/activate
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python evaluate_mapping_intervention.py
