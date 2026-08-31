#!/usr/bin/env bash
# Build the exact ImageNet-1K WebDataset cache for arai3 directly from the
# authorized competition archive, matching the successful sshaik4 build.
#
#   sbatch deltaai/prepare_imagenet_wds_arai3.slurm
#SBATCH --job-name=imagenet-wds
#SBATCH --account=bifg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bifg/arai3/logs/%x_%j.out
#SBATCH --error=/work/hdd/bifg/arai3/logs/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/u/arai3/Attention-is-Transformed-vision}
VENV_DIR=${VENV_DIR:-/work/hdd/bifg/arai3/venvs/lgma-vision}
ARCHIVE=${ARCHIVE:-/work/hdd/bifg/arai3/lgma_data/imagenet/archive/imagenet-object-localization-challenge.zip}
OUTPUT_DIR=${OUTPUT_DIR:-/work/nvme/bifg/arai3/lgma_data/imagenet_wds_official}
REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-/work/hdd/bifg/arai3/lgma_jobs/reference_imagenet_manifest_sshaik4.json}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1024}
TRAIN_SHUFFLE_SEED=${TRAIN_SHUFFLE_SEED:-0}

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
mkdir -p "${OUTPUT_DIR}" /work/hdd/bifg/arai3/logs
cd "${REPO_DIR}"

test -f "${ARCHIVE}"
test "$(stat -c %s "${ARCHIVE}")" = "166496672546"
test -f "${REFERENCE_MANIFEST}"

python -m experiments.prepare_imagenet_kaggle \
  --archive "${ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --samples-per-shard "${SAMPLES_PER_SHARD}" \
  --train-shuffle-seed "${TRAIN_SHUFFLE_SEED}"

python - "${OUTPUT_DIR}/manifest.json" "${REFERENCE_MANIFEST}" <<'PY'
import json
import sys

actual = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
assert actual["num_classes"] == reference["num_classes"] == 1000
assert actual["class_to_idx"] == reference["class_to_idx"]
assert actual["train_order"] == reference["train_order"]
assert actual["train_shuffle_seed"] == reference["train_shuffle_seed"] == 0
for split, expected_samples, expected_shards in (
    ("train", 1_281_167, 1_252),
    ("val", 50_000, 49),
):
    observed = actual["splits"][split]
    reference_split = reference["splits"][split]
    assert observed["samples"] == reference_split["samples"] == expected_samples
    assert len(observed["shards"]) == len(reference_split["shards"]) == expected_shards
    assert [row["samples"] for row in observed["shards"]] == [
        row["samples"] for row in reference_split["shards"]
    ]
print("Exact ImageNet-1K manifest validation passed")
PY
