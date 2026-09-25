#!/usr/bin/env bash
#SBATCH --job-name=tiny-eval-filter
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/shahils/logs/%x_%j.out
#SBATCH --error=/home/shahils/logs/%x_%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

ROOT=/scratch/shahils/job_scripts/tinystories-multiseed-eval-20260924
RUNS=/scratch/shahils/lgma_runs
DEST="$ROOT/filtered"
python "$ROOT/filter_for_eval.py" "$RUNS/tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100_seed_100/checkpoint_step_250000.pt" "$DEST/GT-MHA_seed_100.pt"
python "$ROOT/filter_for_eval.py" "$RUNS/tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100_seed_200/checkpoint_step_250000.pt" "$DEST/GT-MHA_seed_200.pt"
python "$ROOT/filter_for_eval.py" "$RUNS/large_tinystories_mha_b128_h16_h200_4_seed_100/checkpoint_step_250000.pt" "$DEST/MHA_seed_100.pt"
python "$ROOT/filter_for_eval.py" "$RUNS/large_tinystories_mha_b128_h16_h200_4_seed_200/checkpoint_step_250000.pt" "$DEST/MHA_seed_200.pt"
echo FILTER_COMPLETE
