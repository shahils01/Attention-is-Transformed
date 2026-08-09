# Lie-Generated Metric Attention

This repository contains a PyTorch MVP for Lie-Generated Metric Attention (LGMA).
The first milestone is correctness, diagnostics, and small controlled experiments
for parameter and KV-cache efficiency claims.

LGMA replaces independent head-specific query/key projections with shared base
query/key/value projections and generated head metrics:

```text
M_h = exp(sum_r theta[h, r] A_r)
score_ij^h = q_i^T M_h k_j
```

## Layout

```text
src/lgma/
  attention.py
  baselines.py
  transformer.py
  diagnostics.py
  accounting.py
  synthetic.py
experiments/
  train_synthetic.py
  train_tinystories.py
  configs/
tests/
```

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Smoke Experiments

```bash
python experiments/train_synthetic.py --task copy --attention lgma --steps 20
python experiments/train_synthetic.py --task reverse --attention mha --steps 20
```

Synthetic experiments are non-causal by default so tasks like reverse and
future-token modular arithmetic are valid controlled tests. Use `--causal` for
decoder-style synthetic tasks:

```bash
python experiments/train_synthetic.py --task previous --attention lgma --causal --steps 100
python experiments/train_synthetic.py --task cumsum_mod --attention mha --causal --steps 100
```

Use `collaborative` for the direct mixing-vector baseline from *Multi-Head
Attention: Collaborate Instead of Concatenate*. It shares Q/K projections,
learns one unrestricted diagonal mixing vector per head, and retains independent
per-head value projections:

```bash
python experiments/train_synthetic.py --task multi_relation --attention collaborative --steps 5000
python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention collaborative \
  --steps 50000
```

LGMA head diversity can be encouraged with wider head-coordinate initialization
and an optional metric-diversity regularizer:

```bash
python experiments/train_synthetic.py --task reverse --attention lgma --steps 2000 \
  --theta_init_scale 0.25 --generator_init_scale 0.1 \
  --metric_diversity_weight 1e-3

python experiments/train_synthetic.py --task reverse --attention lgma --steps 2000 \
  --theta_init_scale 0.5 --generator_init_scale 0.2 \
  --metric_diversity_weight 1e-2 --metric_diversity_squared
```

By default, the metric-diversity regularizer compares metric deviations
`M_h - I`, not full `M_h`, because full-metric cosine is dominated by the
shared identity component early in training.

For exponential metrics, an optional SVD-free stability cap bounds the
Frobenius norm of the symmetric part of each combined head generator `A_h`.
Setting the radius to `log(4)` conservatively bounds metric singular values to
`[0.25, 4.0]` while leaving the skew-symmetric component unchanged:

```bash
python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --attention lgma \
  --head_generator_symmetric_cap 1.38629436112
```

The existing trace-zero generator stabilizer remains enabled by default. Use
`--no-stabilize_generators` to disable it for an ablation; the cap and
trace-zero stabilizer can be enabled independently.

For the first harder controlled benchmark, use `multi_relation`. Position 0 is
a relation-control token selecting one of four transformations over the rest of
the sequence: copy, reverse, previous-token lookup, or next-token lookup. The
runner reports overall validation loss and per-relation validation loss.

```bash
python experiments/train_synthetic.py --task multi_relation --attention mha --steps 5000
python experiments/train_synthetic.py --task multi_relation --attention shared_identity --steps 5000
python experiments/train_synthetic.py --task multi_relation --attention lgma --steps 5000
python experiments/train_synthetic.py --task multi_relation --attention lgma --steps 5000 \
  --theta_init_scale 0.5 --generator_init_scale 0.2 \
  --metric_diversity_weight 1e-2
```

## Dataset Downloads

Dataset download scripts live in `dataset_downloads/`. On Palmetto, use scratch
storage instead of home:

```bash
mkdir -p $SCRATCH/lgma_data $SCRATCH/hf_cache
export DATA_DIR=$SCRATCH/lgma_data
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
pip install -e ".[data]"
```

Download real text datasets:

```bash
python dataset_downloads/download_tinystories.py
python dataset_downloads/download_wikitext103.py
python dataset_downloads/download_c4_subset.py --rows 50000
```

`experiments/train_tinystories.py` expects a plain text file and trains a
character-level decoder-only language model. It now reports validation
loss/perplexity, attention accounting, and LGMA diversity diagnostics:

```bash
python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention mha \
  --steps 1000

python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention lgma \
  --steps 1000

python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention lgma \
  --theta_init_scale 0.5 \
  --generator_init_scale 0.2 \
  --metric_diversity_weight 1e-2 \
  --steps 1000
```

For longer A100 runs, keep the corpus on CPU and move sampled batches to CUDA:

```bash
python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention lgma \
  --device cuda \
  --precision bf16 \
  --batch_size 64 \
  --grad_accum_steps 4 \
  --steps 50000 \
  --log_every 100 \
  --eval_every 1000 \
  --save_every 5000 \
  --output_dir $SCRATCH/lgma_runs/tinystories_lgma_bf16
```

Resume a saved run:

```bash
python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention lgma \
  --device cuda \
  --precision bf16 \
  --steps 50000 \
  --output_dir $SCRATCH/lgma_runs/tinystories_lgma_bf16 \
  --resume_checkpoint $SCRATCH/lgma_runs/tinystories_lgma_bf16/checkpoint_step_5000.pt
```

Evaluate a saved TinyStories checkpoint:

```bash
python experiments/eval_tinystories.py \
  --checkpoint $SCRATCH/lgma_runs/tinystories_lgma_bf16/checkpoint_final.pt \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --device cuda \
  --eval_batches 100
```

Sample from the trained character-level network interactively:

```bash
python experiments/chat_tinystories.py \
  --checkpoint $SCRATCH/lgma_runs/tinystories_lgma_bf16/checkpoint_final.pt \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --device cuda
```

This is not an instruction-tuned chatbot; it is a character-level language
model. The "chat" script prompts for text and samples a continuation from the
trained network.

For multi-A100 DDP, launch with `torchrun`. The runner automatically enables
DDP when `WORLD_SIZE > 1`; rank 0 writes logs/checkpoints/reports.

```bash
torchrun --standalone --nproc_per_node=4 experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --attention lgma \
  --device cuda \
  --precision bf16 \
  --batch_size 64 \
  --grad_accum_steps 4 \
  --steps 50000 \
  --log_every 100 \
  --eval_every 1000 \
  --save_every 5000 \
  --output_dir $SCRATCH/lgma_runs/tinystories_lgma_ddp4_bf16
```

For Slurm multi-GPU jobs, use the same script under `srun` or `torchrun`
according to the Palmetto job template. Disable auto-DDP with `--no_ddp` for
single-process debugging.

### Fair TinyStories Runtime Comparisons

The TinyStories runner logs normalized efficiency coordinates for reviewer-safe
comparisons across different GPU counts and batch sizes:

- `tokens_seen`
- `tokens_per_step`
- `effective_global_batch`
- `tokens_per_second_per_gpu`
- `elapsed_training_seconds`
- `gpu_hours`
- `peak_memory_gib`

Summarize multiple run directories with:

```bash
python experiments/summarize_tinystories_runs.py \
  /path/to/tinystories_mha_ddp4 \
  /path/to/tinystories_lgma_ddp8 \
  --csv /path/to/summary.csv \
  --markdown /path/to/summary.md \
  --plot_dir /path/to/plots
```

Use the generated validation-loss-vs-tokens and validation-loss-vs-GPU-hours
plots for paper figures instead of step-only W&B screenshots.
