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

## Official GLUE test results

The table reports hidden-test results returned by the official GLUE evaluation server on August 29, 2026. Each task model was independently fine-tuned from its architecture's pretrained checkpoint. Hyperparameters and checkpoints were selected using validation data only; official test labels were never accessed or used for model selection. Scores are on a 0–100 scale. Paired metrics are reported in the same order as the official server.

| Task | Metric | MHA | GQA | collaborativeMHA | GT-MHA residual |
| --- | --- | ---: | ---: | ---: | ---: |
| CoLA | Matthews correlation | 28.0 | 29.0 | **30.8** | **31.5** |
| SST-2 | Accuracy | 88.2 | 87.4 | 88.4 | **89.2** |
| MRPC | F1 / Accuracy | **81.9 / 73.7** | 81.4 / 74.6 | 78.0 / 68.0 | 79.1 / 69.7 |
| STS-B | Pearson / Spearman correlation | **78.4 / 76.9** | 76.3 / 74.6 | 73.1 / 71.2 | 77.5 / 76.0 |
| QQP | F1 / Accuracy | 68.1 / **87.5** | 67.9 / 87.4 | 66.7 / 86.9 | **68.3** / **87.5** |
| MNLI | Matched / Mismatched accuracy | **77.7** / 76.7 | 76.9 / 75.8 | 75.4 / 74.4 | **77.7** / **76.9** |
| QNLI | Accuracy | 86.1 | 85.5 | 85.5 | **86.3** |
| RTE | Accuracy | 54.7 | **58.2** | 53.7 | 56.3 |
| WNLI | Accuracy | **65.1** | **65.1** | **65.1** | **65.1** |
| **Official GLUE score** | Official aggregate | 70.3 | 70.3 | 68.9 | **70.5** |

Under the GLUE submission convention, which excludes word or word-part embeddings, MHA has 86,043,651 parameters, GQA has 76,594,179, collaborativeMHA has 73,059,843, and GT-MHA residual has 72,658,179. GQA and collaborativeMHA use four key-value heads and shared query-key projections, respectively, while all task models are fine-tuned independently.

### Fine-tuning and checkpoint-selection protocol

The eight primary tasks used an equal-budget task-specific learning-rate sweep for every evaluated attention type, with effective batch size 128 and learning rates {2e-5, 3e-5, 5e-5}. MNLI, QNLI, SST-2, and QQP used four epochs; CoLA, MRPC, STS-B, and RTE used eight epochs. For each architecture and task, the learning rate was selected by its mean primary validation metric over seeds 42–44, followed by confirmation runs with seeds 45 and 46. The submitted checkpoint was the highest-scoring seed at that preselected learning rate. Validation-metric ties were broken by lower validation loss.

WNLI was not swept because its tiny, adversarial validation split makes architecture-specific hyperparameter selection unreliable. Its submission checkpoint used the fixed common setting of learning rate 2e-5, seed 42, and four epochs.

| Task | MHA selected LR | GQA selected LR | collaborativeMHA selected LR | GT-MHA residual selected LR |
| --- | ---: | ---: | ---: | ---: |
| MNLI | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| QNLI | 3e-5 | 3e-5 | 3e-5 | 3e-5 |
| SST-2 | 2e-5 | 5e-5 | 5e-5 | 2e-5 |
| CoLA | 2e-5 | 2e-5 | 3e-5 | 2e-5 |
| MRPC | 5e-5 | 5e-5 | 2e-5 | 2e-5 |
| STS-B | 5e-5 | 3e-5 | 5e-5 | 5e-5 |
| QQP | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| RTE | 5e-5 | 3e-5 | 5e-5 | 5e-5 |

## TinyStories-v2 checkpoints

The table reports validation loss at step 250000 (the selected checkpoint may be a nearby saved step where noted).

| Attention type | Seed | Parameters | Eval loss | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| MQA | 0 | 128,509,952 | 0.3022347093 | [`checkpoint_step_249800.pt`](TinyStories-v2%20Checkpoints/MQA/checkpoint_step_249800.pt) |
| GQA | 0 | 133,228,544 | 0.2958051562 | [`checkpoint_step_250000.pt`](TinyStories-v2%20Checkpoints/GQA/checkpoint_step_250000.pt) |
| Collaborative MHA | 0 | 128,522,240 | 0.2971141338 | [`checkpoint_step_250000.pt`](TinyStories-v2%20Checkpoints/Collaborative%20MHA/checkpoint_step_250000.pt) |
| MHA | 0 | 152,102,912 | 0.2945130467 | [`checkpoint_step_250000.pt`](TinyStories-v2%20Checkpoints/MHA/checkpoint_step_250000.pt) |
| **GT-MHA residual** | 0 | 134,018,240 | **0.2906896472** | [`checkpoint_step_250000.pt`](TinyStories-v2%20Checkpoints/GT-MHA%20residual/checkpoint_step_250000.pt) |
| **GT-MHA quad** | 0 | 124,581,056 | **0.2938950062** | [`checkpoint_step_250000.pt`](TinyStories-v2%20Checkpoints/GT-MHA%20quad/checkpoint_step_250000.pt) |
| **GT-MHA exact** | 0 | 124,581,056 | **0.2942119241** | [`checkpoint_step_250000.pt`](TinyStories-v2%20Checkpoints/GT-MHA%20exact/checkpoint_step_250000.pt) |

TinyStories checkpoints are organized under `TinyStories-v2 Checkpoints/`.

## Loading a checkpoint

```python
from lgma.bert import load_bert_sequence_classifier
from transformers import AutoTokenizer

repo = "shahils/GT-MHA"
path = "BERT-Base Checkpoints/gt_mha_exact/bert_base_gt_mha_exact_b4g8h12_random_seed42_explicit_fuseqkv_v2/checkpoint-100000"

tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=path)
model, audit = load_bert_sequence_classifier(
    "/local/path/to/checkpoint-100000",
    num_labels=2,
    attention_type="gt_mha_residual",
)
```

For non-MHA checkpoints, keep `bert_gt_mha_manifest.json` and `gt_mha_state_dict.pt` in the same checkpoint directory as `config.json` and `model.safetensors`. The loader uses these files to restore the trained attention modules; loading only `model.safetensors` with a generic Transformers class will fall back to newly initialized attention.

The repository is intended for research comparison and reproducibility. See the main code repository for model definitions, training scripts, and evaluation procedures.
