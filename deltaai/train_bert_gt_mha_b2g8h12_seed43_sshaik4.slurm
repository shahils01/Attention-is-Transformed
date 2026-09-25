#!/usr/bin/env bash
# Matched to the historical B4/G8/H12 residual seed42 explicit-attention run.
# The paper-configuration guard is disabled solely to allow B=2.
# Historical source preserves random_sphere initialization at scale 0.02.
#SBATCH --job-name=bert-gt-b2g8h12-s43
#SBATCH --account=biad-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --chdir=/work/hdd/biad/sshaik4/lgma_jobs/bert_b2g8h12_seed43_20260911/repo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/biad/sshaik4/logs/%x_%j.out
#SBATCH --error=/work/hdd/biad/sshaik4/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR=/work/hdd/biad/sshaik4
STAGE_DIR=${PROJECT_DIR}/lgma_jobs/bert_b2g8h12_seed43_20260911
REPO_DIR=${STAGE_DIR}/repo
module load python/miniforge3_pytorch/2.12.0
source "${PROJECT_DIR}/venvs/lgma-bert-no-te/bin/activate"
export OMP_NUM_THREADS=4
export PYTHONPATH="${STAGE_DIR}/compat/no_transformer_engine:${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME=${PROJECT_DIR}/lgma_data/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=bert-gt-mha WANDB_ENTITY=i2rLAB
export WANDB_RUN_GROUP=bert-ablation-b2g8h12-explicit
export WANDB_RUN_ID=bert_base_gt_mha_residual_b2g8h12_random_seed43_explicit_20260911
export WANDB_NAME=${WANDB_RUN_ID} WANDB_RESUME=allow
export WANDB_TAGS=bert-pretraining,base-count-ablation,b2g8h12,seed-43,historical-init
cd "${REPO_DIR}"
python "${STAGE_DIR}/run_matched.py" "$@"
