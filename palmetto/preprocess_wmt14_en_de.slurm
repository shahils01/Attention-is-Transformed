#!/usr/bin/env bash
# Shared WMT14 EN→DE processing.  No GPU and no attention-method-specific logic.
#SBATCH --job-name=wmt14-en-de-prep
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shahils/lgma_data/wmt14_en_de/logs/preprocess_%j.out
#SBATCH --error=/scratch/shahils/lgma_data/wmt14_en_de/logs/preprocess_%j.err

source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -euo pipefail

cd /home/shahils/Desktop/gitBackupRepo/Attention-is-Transformed
python experiments/prepare_wmt14_en_de.py --data-root /scratch/shahils/lgma_data/wmt14_en_de
