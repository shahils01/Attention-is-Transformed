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
