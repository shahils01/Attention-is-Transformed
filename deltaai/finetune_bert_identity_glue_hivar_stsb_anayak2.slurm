#!/usr/bin/env bash
# Run the high-variance GLUE tasks followed by STS-B in one allocation so that
# the account needs only two queued jobs for the complete Identity ablation.
#SBATCH --job-name=id-glue-hv-stsb
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/work/hdd/bicn/anayak2/repos/Attention-is-Transformed-bert-identity
SWEEP_SCRIPT=${REPO_DIR}/deltaai/finetune_bert_identity_glue_adaptive_sweep_anayak2.slurm

cd "${REPO_DIR}"
FAMILY=hivar bash "${SWEEP_SCRIPT}"
FAMILY=stsb bash "${SWEEP_SCRIPT}"

