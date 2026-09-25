#!/bin/bash
#SBATCH --job-name=tiny-gen-main
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_40gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shahils/tinystories_eval_20260919/main_run/logs/generate_%A_%a.out
#SBATCH --error=/scratch/shahils/tinystories_eval_20260919/main_run/logs/generate_%A_%a.err
set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /scratch/shahils/tinystories_eval_20260919/main_run
python -u generate_main.py "$SLURM_ARRAY_TASK_ID"
