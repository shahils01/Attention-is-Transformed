#!/usr/bin/env bash

set -euo pipefail

attention_type=${1:?Usage: submit_fineweb_chain.sh ATTENTION_TYPE CODE_SNAPSHOT [SEGMENTS]}
code_snapshot=${2:?Usage: submit_fineweb_chain.sh ATTENTION_TYPE CODE_SNAPSHOT [SEGMENTS]}
segments=${3:-3}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
job_script=${script_dir}/train_fineweb_ddp4.slurm
previous_job=""

case "${attention_type}" in
  mha) job_prefix=fw-mha ;;
  collaborative) job_prefix=fw-collab ;;
  lgma_quad) job_prefix=fw-lgma-g${NUM_GENERATORS:?Set NUM_GENERATORS for LGMA} ;;
  *) echo "Unsupported attention type: ${attention_type}" >&2; exit 2 ;;
esac

for ((segment = 1; segment <= segments; segment++)); do
  submit_args=(
    --parsable
    --job-name="${job_prefix}-s${segment}"
    --export="ALL,ATTENTION_TYPE=${attention_type},CODE_SNAPSHOT=${code_snapshot}"
  )
  if [[ -n "${previous_job}" ]]; then
    submit_args+=(--dependency="afternotok:${previous_job}")
  fi
  job_id="$(sbatch "${submit_args[@]}" "${job_script}")"
  echo "${attention_type} segment ${segment}: job ${job_id}"
  previous_job=${job_id}
done
