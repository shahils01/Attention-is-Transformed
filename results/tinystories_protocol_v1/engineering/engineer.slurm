#!/bin/bash
#SBATCH --job-name=tiny-eval-dev
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shahils/tinystories_eval_20260919/logs/dev_%A_%a.out
#SBATCH --error=/scratch/shahils/tinystories_eval_20260919/logs/dev_%A_%a.err
set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u
cd /scratch/shahils/tinystories_eval_20260919
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -u engineer.py "$SLURM_ARRAY_TASK_ID"
