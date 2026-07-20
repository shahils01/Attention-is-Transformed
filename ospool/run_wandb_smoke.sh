#!/usr/bin/env bash
set -euo pipefail

bundle="$(find . -maxdepth 1 -type f -name 'wandb-py*.tar.gz' -print -quit)"
if [[ -z "${bundle}" ]]; then
    echo "W&B dependency bundle was not transferred" >&2
    exit 2
fi
if [[ ! -s ospool/.wandb_api_key ]]; then
    echo "W&B API key was not transferred" >&2
    exit 2
fi

mkdir -p ospool_python_deps wandb_smoke_output
tar -xzf "${bundle}" -C ospool_python_deps
export PYTHONPATH="${PWD}/ospool_python_deps/python${PYTHONPATH:+:${PYTHONPATH}}"
WANDB_API_KEY="$(tr -d '\r\n' < ospool/.wandb_api_key)"
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "W&B API key file contains only whitespace or line endings" >&2
    exit 2
fi
export WANDB_API_KEY
export WANDB_DIR="${PWD}/wandb_smoke_output"

echo "Loaded a W&B API key with ${#WANDB_API_KEY} characters"

python - <<'PY'
import os
import wandb

assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY was not inherited by Python"
wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)

with wandb.init(
    project="lgma",
    name="ospool-wandb-connectivity-smoke",
    tags=["ospool", "connectivity-smoke"],
) as run:
    run.log({"ospool_connectivity": 1})
    print(f"W&B run URL: {run.url}")
PY
