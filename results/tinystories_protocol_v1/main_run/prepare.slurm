#!/bin/bash
#SBATCH --job-name=tiny-prompts
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shahils/tinystories_eval_20260919/main_run/logs/prompts_%j.out
#SBATCH --error=/scratch/shahils/tinystories_eval_20260919/main_run/logs/prompts_%j.err
set -euo pipefail
cd /scratch/shahils/tinystories_eval_20260919/main_run
python3 -u prepare_prompts.py
