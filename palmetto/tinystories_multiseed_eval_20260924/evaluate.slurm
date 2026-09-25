#!/usr/bin/env bash
#SBATCH --job-name=tiny-3seed-val
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/home/shahils/logs/%x_%A_%a.out
#SBATCH --error=/home/shahils/logs/%x_%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video

ROOT=/scratch/shahils/job_scripts/tinystories-multiseed-eval-20260924
RUNS=/scratch/shahils/lgma_runs
DATA=/scratch/shahils/lgma_data/tinystory
export PYTHONPATH="$ROOT/source"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

case "$SLURM_ARRAY_TASK_ID" in
  0) METHOD=GT-MHA; SEED=0;   RUN=tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100_seed_0;   FILE=checkpoint_step_250000.pt ;;
  1) METHOD=GT-MHA; SEED=100; RUN=tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100_seed_100; FILE=checkpoint_step_250000.pt ;;
  2) METHOD=GT-MHA; SEED=200; RUN=tinystories_lgma_residual_b4_h16_g8_beta1_softmax_h100_seed_200; FILE=checkpoint_step_250000.pt ;;
  3) METHOD=MHA;    SEED=0;   RUN=large_tinystories_mha_b128_h16_h200_4_seed_0;   FILE=checkpoint_step_2500000.pt ;;
  4) METHOD=MHA;    SEED=100; RUN=large_tinystories_mha_b128_h16_h200_4_seed_100; FILE=checkpoint_step_250000.pt ;;
  5) METHOD=MHA;    SEED=200; RUN=large_tinystories_mha_b128_h16_h200_4_seed_200; FILE=checkpoint_step_250000.pt ;;
esac

OUT="$ROOT/results/${METHOD}_seed_${SEED}"
test -r "$RUNS/$RUN/$FILE"
test -r "$DATA/TinyStoriesV2-GPT4-train.txt"
test -r "$DATA/TinyStoriesV2-GPT4-valid.txt"
mkdir -p "$OUT"
python -u "$ROOT/evaluate.py" \
  --checkpoint "$RUNS/$RUN/$FILE" \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --method "$METHOD" \
  --seed "$SEED"
