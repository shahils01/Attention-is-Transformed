# TinyLlama LGMA Adapter Trainer

This folder is intentionally separate from `src/lgma`. It is for post-hoc
teacher-student replacement of TinyLlama/Llama attention blocks with LGMA
attention blocks.

The trainer does not need TinyLlama's original 3T-token pretraining dataset.
Use a calibration text file that represents the activation distribution you
care about. Start with 10M-100M tokens for serious runs; use a tiny file for
smoke tests.

## Install

```bash
pip install -e ".[dev]"
pip install transformers datasets accelerate
```

## Stage 1: Per-Layer Attention Distillation

```bash
python adapter_trainer/distill_tinyllama.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --calibration_text_file /path/to/calibration.txt \
  --output_dir /scratch/$USER/lgma_adapters/tinyllama_stage1 \
  --stage stage1 \
  --layers 0-3 \
  --attention_variant lgma_residual \
  --qk_num_base_heads 4 \
  --value_num_base_heads 4 \
  --num_generators 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --steps_per_layer 500 \
  --device cuda \
  --precision bf16
```

At startup the script prints, for each selected layer, the teacher hidden size,
head count, KV heads, head dim, projection shapes, LGMA replacement shape,
parameter counts, and KV-cache bytes/token.

## Stage 2/3: Whole-Model Recovery

```bash
python adapter_trainer/distill_tinyllama.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --calibration_text_file /path/to/calibration.txt \
  --output_dir /scratch/$USER/lgma_adapters/tinyllama_stage2 \
  --adapter_dir /scratch/$USER/lgma_adapters/tinyllama_stage1 \
  --stage stage2 \
  --layers 0-3 \
  --attention_variant lgma_residual \
  --qk_num_base_heads 4 \
  --value_num_base_heads 4 \
  --steps 1000 \
  --device cuda \
  --precision bf16
```

Stage 2 trains the swapped attention modules against teacher logits and final
hidden states. Stage 3 uses the same command with `--stage stage3`; optionally
add `--train_output_projections` and `--train_norms` for light recovery.

## Evaluation

```bash
python adapter_trainer/evaluate_swapped.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter_dir /scratch/$USER/lgma_adapters/tinyllama_stage2 \
  --calibration_text_file /path/to/heldout.txt \
  --layers 0-3 \
  --sequence_length 512 \
  --eval_steps 50 \
  --device cuda \
  --precision bf16
```

## Current Scope

- Text-only TinyLlama/Llama attention replacement.
- Training runs with `use_cache=False`.
- Generation KV-cache support is intentionally deferred until non-cache
  distillation quality is verified.
