---
library_name: transformers
tags:
  - bert
  - attention
  - gt-mha
  - masked-language-modeling
---

# Group-Transformed Multi-Head Attention (GT-MHA)

This repository contains pretrained BERT-Base checkpoints for comparing attention mechanisms, with **Group-Transformed Multi-Head Attention (GT-MHA)** as the main contribution of this work.

GT-MHA transforms groups of attention heads while retaining the multi-head structure. The repository includes both the exact formulation and a residual formulation, alongside standard attention baselines for controlled comparisons.

## BERT-Base checkpoints

The table reports the lowest recorded validation (`eval_loss`) value during masked-language-model pretraining. Each entry points to the corresponding saved checkpoint directory.

| Attention type | Seed | Parameters | Eval loss | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| MHA | 44 | 109,514,298 | 2.0972 | `checkpoint-100000` |
| GQA | 42 | 100,064,826 | 2.1450 | `checkpoint-100000` |
| MQA | 42 | 96,521,274 | 2.2177 | `checkpoint-100000` |
| Collaborative MHA | 43 | 96,530,490 | 2.1196 | `checkpoint-100000` |
| **GT-MHA exact** | **42** | **96,128,826** | **2.1428** | `checkpoint-100000` |
| **GT-MHA residual** | **44** | **96,128,826** | **2.0961** | `checkpoint-100000` |

The selected pretrained checkpoint trees are organized under `BERT-Base Checkpoints/`:

```text
BERT-Base Checkpoints/
├── mha/
├── gqa/
├── mqa/
├── collaborative_mha/
├── gt_mha_exact/
└── gt_mha_residual/
```

Each selected checkpoint includes the model weights, configuration, tokenizer, trainer state, optimizer and scheduler state, RNG state, and training arguments. This makes the runs usable both for evaluation and for exact training resumption.

## GLUE fine-tuning comparison

Four-epoch fine-tuning was run independently from the corresponding MHA and GQA BERT-Base checkpoints. MNLI, QNLI, and SST-2 report accuracy; CoLA reports Matthews correlation coefficient (MCC). The table also includes the final validation loss.

| Task | MHA metric | GQA metric | MHA eval loss | GQA eval loss |
| --- | ---: | ---: | ---: | ---: |
| MNLI (accuracy) | 0.7565 | 0.7469 | 0.6109 | 0.6130 |
| QNLI (accuracy) | 0.8589 | 0.8525 | 0.3437 | 0.3499 |
| SST-2 (accuracy) | 0.8704 | 0.8509 | 0.3178 | 0.3463 |
| CoLA (MCC) | 0.2398 | 0.2979 | 0.5902 | 0.5883 |

The final fine-tuning checkpoints are organized under `Fine-tuned GLUE Checkpoints/` by attention type and task. Each task starts from its own pretrained BERT-Base checkpoint; tasks do not initialize from one another.

```text
Fine-tuned GLUE Checkpoints/
├── mha/{mnli,qnli,sst2,cola}/
└── gqa/{mnli,qnli,sst2,cola}/
```

## Loading a checkpoint

```python
from transformers import AutoModel, AutoTokenizer

repo = "shahils/GT-MHA"
path = "BERT-Base Checkpoints/gt_mha_exact/bert_base_gt_mha_exact_b4g8h12_random_seed42_explicit_fuseqkv_v2/checkpoint-100000"

tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=path)
model = AutoModel.from_pretrained(repo, subfolder=path)
```

The repository is intended for research comparison and reproducibility. See the main code repository for model definitions, training scripts, and evaluation procedures.
