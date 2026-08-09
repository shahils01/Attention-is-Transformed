#!/usr/bin/env bash

set -euo pipefail

segments=${1:-6}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
job_script=${script_dir}/train_tinystories_collaborative_ddp4.slurm
previous_job=""

for ((segment = 1; segment <= segments; segment++)); do
    if [[ -z "${previous_job}" ]]; then
        job_id="$(sbatch --parsable "${job_script}")"
    else
        job_id="$(
            sbatch --parsable \
              --dependency="afternotok:${previous_job}" \
              "${job_script}"
        )"
    fi
    echo "segment ${segment}: job ${job_id}"
    previous_job=${job_id}
done
