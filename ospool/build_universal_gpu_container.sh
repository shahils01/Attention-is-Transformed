#!/usr/bin/env bash
set -euo pipefail

: "${DATA:?DATA must point to your OSPool data directory}"

OUTPUT_IMAGE="${DATA}/containers/pytorch-2.9.0-cuda12.8-v1.sif"
SOURCE_IMAGE="docker://pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime"

mkdir -p "${HOME}/tmp" "${DATA}/containers"
export TMPDIR="${HOME}/tmp"
export APPTAINER_TMPDIR="${HOME}/tmp"
export APPTAINER_CACHEDIR="${HOME}/tmp"

apptainer build --ignore-proot "${OUTPUT_IMAGE}" "${SOURCE_IMAGE}"

apptainer exec "${OUTPUT_IMAGE}" \
    python -c 'import torch; print(torch.__version__, torch.version.cuda)'

echo "Created ${OUTPUT_IMAGE}"
ls -lh "${OUTPUT_IMAGE}"
