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

`experiments/train_tinystories.py` is intentionally dependency-light. It expects
a plain text file and trains a small character-level decoder-only language model:

```bash
python experiments/train_tinystories.py --data_path /path/to/tinystories.txt --steps 100
```
