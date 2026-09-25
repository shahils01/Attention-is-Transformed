#!/usr/bin/env bash
#SBATCH --job-name=bert-transform-diag
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/shahils/bert_transform_analysis/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shahils/bert_transform_analysis/logs/%x_%A_%a.err

set -eo pipefail
source /etc/profile.d/modules.sh
module load anaconda3
source activate llava_video
set -u

ANALYSIS_ROOT=/scratch/shahils/bert_transform_analysis
SOURCE_ROOT=/scratch/shahils/tinystories_eval_20260919/repo
SEEDS=(42 43 44)
CHECKPOINT_SHAS=(
  5e06d2f12801e464876efbeb668df1dc6540471dbbe2109675fc1bbbc9757989
  242a3d784f41bf492055b50a15c36eba29f98537fabbfae01c83490b4a5370f3
  f37a0546162df0a386c445050ba9f9fa664f65eacf0a1fa9bd5523fd45d9b643
)

SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT_SHA="${CHECKPOINT_SHAS[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT_DIR="${ANALYSIS_ROOT}/checkpoints/seed${SEED}"
OUTPUT_DIR="${ANALYSIS_ROOT}/results/seed${SEED}"

test -f "${CHECKPOINT_DIR}/config.json"
test -f "${CHECKPOINT_DIR}/model.safetensors"
mkdir -p "${ANALYSIS_ROOT}/logs" "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

python -u "${ANALYSIS_ROOT}/analyze_bert_checkpoint.py" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --expected-sha256 "${CHECKPOINT_SHA}" \
  --repo-root "${SOURCE_ROOT}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}"
