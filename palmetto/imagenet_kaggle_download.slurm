#!/bin/bash
#SBATCH --job-name=imagenet-download
#SBATCH --partition=work1
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/shahils/imagenet/logs/download-%j.out

set -euo pipefail
umask 077

SCRIPT=/scratch/shahils/imagenet/scripts/download_imagenet_kaggle.py
DESTINATION=/scratch/shahils/imagenet/archive

mkdir -p "$DESTINATION" /scratch/shahils/imagenet/logs
python3 "$SCRIPT" --probe
python3 "$SCRIPT" --destination "$DESTINATION"
