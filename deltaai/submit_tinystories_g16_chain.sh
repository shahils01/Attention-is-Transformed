#!/usr/bin/env bash

set -euo pipefail

segments=${1:-6}
generators=${2:-16}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
job_script=${script_dir}/train_tinystories_g16_ddp4.slurm
previous_job=""

if ! [[ "${generators}" =~ ^[1-9][0-9]*$ ]]; then
    echo "generators must be a positive integer" >&2
    exit 2
fi

for ((segment = 1; segment <= segments; segment++)); do
    if [[ -z "${previous_job}" ]]; then
        job_id="$(
            sbatch --parsable \
              --job-name="lgma-g${generators}-ddp4" \
              --export="ALL,NUM_GENERATORS=${generators}" \
              "${job_script}"
        )"
    else
        job_id="$(
            sbatch --parsable \
              --job-name="lgma-g${generators}-ddp4" \
              --export="ALL,NUM_GENERATORS=${generators}" \
              --dependency="afternotok:${previous_job}" \
              "${job_script}"
        )"
    fi
    echo "segment ${segment}: job ${job_id}"
    previous_job=${job_id}
done
