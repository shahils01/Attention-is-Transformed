#!/bin/bash
#SBATCH --job-name=imagenet-preflight
#SBATCH --partition=interact
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/shahils/imagenet/logs/preflight-%j.out

set -euo pipefail
umask 077

ROOT=/scratch/shahils/imagenet
VENV=/scratch/shahils/.venvs/kaggle

mkdir -p "$ROOT/logs" /scratch/shahils/.venvs

if [[ ! -x "$VENV/bin/kaggle" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet kaggle
fi

"$VENV/bin/kaggle" competitions files \
  -c imagenet-object-localization-challenge \
  --csv > "$ROOT/competition-files.csv"

echo "Kaggle access verified."
echo "Competition files:"
sed -n '1,80p' "$ROOT/competition-files.csv"
