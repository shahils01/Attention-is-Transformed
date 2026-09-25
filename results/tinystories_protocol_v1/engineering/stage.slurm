#!/bin/bash
#SBATCH --job-name=tiny-eval-stage
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shahils/tinystories_eval_20260919/logs/stage_%j.out
#SBATCH --error=/scratch/shahils/tinystories_eval_20260919/logs/stage_%j.err
set -euo pipefail
cd /scratch/shahils/tinystories_eval_20260919
python3 -u stage.py
