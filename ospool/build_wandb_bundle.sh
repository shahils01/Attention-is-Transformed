#!/usr/bin/env bash
set -euo pipefail

: "${DATA:?DATA must point to your OSPool data directory}"

BASE_IMAGE="${OSPOOL_BASE_IMAGE:-${DATA}/containers/pytorch-2.9.0-cuda12.6-v1.sif}"
OUTPUT_DIR="${DATA}/dependencies"
OUTPUT_BUNDLE="${OUTPUT_DIR}/wandb-py311.tar.gz"

if [[ ! -f "${BASE_IMAGE}" ]]; then
    echo "Container image not found: ${BASE_IMAGE}" >&2
    exit 2
fi

mkdir -p "${HOME}/tmp" "${OUTPUT_DIR}"
work_dir="$(mktemp -d "${HOME}/tmp/wandb-bundle.XXXXXX")"

cleanup() {
    rm -rf -- "${work_dir}"
}
trap cleanup EXIT

echo "Installing W&B with the Python runtime from ${BASE_IMAGE}"
apptainer exec "${BASE_IMAGE}" \
    python -m pip install --no-cache-dir --target "${work_dir}/python" wandb

APPTAINERENV_PYTHONPATH="${work_dir}/python" \
    apptainer exec "${BASE_IMAGE}" \
    python -c 'import wandb; print(f"W&B bundle version: {wandb.__version__}")'

tar -C "${work_dir}" -czf "${OUTPUT_BUNDLE}" python
tar -tzf "${OUTPUT_BUNDLE}" >/dev/null

echo "Created ${OUTPUT_BUNDLE}"
ls -lh "${OUTPUT_BUNDLE}"
