# Running LGMA on OSPool

These files are configured for:

- Access point: `ap40`
- Username: `shahil.shaik`
- Data directory: `/ospool/ap40/data/shahil.shaik`

## 1. Put the repository on the access point

From a local terminal (not from inside the SSH session):

```bash
scp -r "/Users/shahilshaik/Documents/Attention is Transformed" \
  shahil.shaik@ap40.uw.osg-htc.org:~/attention-is-transformed
```

Alternatively, clone the GitHub repository on the access point. Make sure the
OSPool files have been committed and pushed first if using this route.

## 2. Create the PyTorch container image

Run these commands on the access point. OSPool requires Apptainer temporary and
cache files to be placed outside its shared home filesystem.

```bash
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
export APPTAINER_TMPDIR="$HOME/tmp"
export APPTAINER_CACHEDIR="$HOME/tmp"

apptainer pull \
  "$DATA/pytorch-2.9.0.sif" \
  docker://pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime
```

This downloads an existing PyTorch Docker image and converts it to an Apptainer
`.sif` image. It does not build PyTorch from source.

Confirm that the image exists:

```bash
ls -lh "$DATA/pytorch-2.9.0.sif"
```

## 3. Submit one short GPU job

From the repository root on the access point:

```bash
chmod +x ospool/run_gpu_smoke.sh
mkdir -p logs
condor_submit ospool/gpu_smoke.sub
condor_watch_q
```

Press `Ctrl-C` to exit `condor_watch_q`; this does not cancel the job.

After completion, inspect the output:

```bash
cat logs/lgma_gpu_smoke_*.0.out
cat logs/lgma_gpu_smoke_*.0.err
tail -n 30 logs/lgma_gpu_smoke_*.log
tail -n 20 results/gpu_smoke.jsonl
```

Useful diagnostic commands:

```bash
condor_q -nobatch
condor_q -hold
condor_q -better-analyze JOB_ID
condor_history -limit 5
```

Do not submit the 500,000-step training configuration until this smoke job
finishes successfully and its actual memory, disk, and runtime are reviewed.

## OSPool-only full TinyStories compatibility layer

The full-training files below are isolated from the Clemson/Palmetto pipeline:

- `train_tinystories_compat.py` installs compact token encoding only in the
  OSPool Python process, then delegates to the unchanged
  `experiments/train_tinystories.py` entry point.
- `run_tinystories_hopper_checkpointed.sh` runs one H100/H200 GPU for four-hour
  segments, validates and resumes the newest checkpoint, and exits with code 85
  after each successful segment.
- `tinystories_hopper_checkpointed.sub` tells HTCondor to preserve the results
  directory on exit code 85 and restart the same job.

The Palmetto commands and all existing source files remain unchanged. Palmetto
should continue to invoke `experiments/train_tinystories.py` directly.

Before submitting, verify the short smoke and Hopper configuration probe. Then:

```bash
chmod +x ospool/run_tinystories_hopper_checkpointed.sh
mkdir -p logs

condor_submit -dry-run /tmp/tinystories_hopper_checkpointed.ad \
  ospool/tinystories_hopper_checkpointed.sub
condor_submit ospool/tinystories_hopper_checkpointed.sub
condor_watch_q
```

The default OSPool configuration uses a microbatch of 4, 32 accumulation steps,
and context length 512, for an effective batch size of 128 on one GPU. W&B is
disabled by default because the base PyTorch container may not include it.

Useful checkpoint-job commands:

```bash
condor_q -nobatch
condor_tail -f JOB_ID
condor_q JOB_ID -better-analyze
```

Do not remove or manually release a checkpointing job merely because it returns
exit code 85; HTCondor uses that code to transfer its checkpoint and requeue it.
