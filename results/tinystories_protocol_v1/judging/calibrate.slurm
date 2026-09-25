#!/bin/bash
#SBATCH --job-name=tiny-judge-dev
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shahils/tinystories_eval_20260919/judging/dev_%j.out
#SBATCH --error=/scratch/shahils/tinystories_eval_20260919/judging/dev_%j.err
set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u
cd /scratch/shahils/tinystories_eval_20260919
python -u judging/judge.py run judging/development_v1 --budget-usd 5 --limit 160
