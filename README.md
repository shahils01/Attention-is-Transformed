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

## BERT checkpoint and GT-MHA experiments

Install the optional BERT stack:

```bash
pip install -e '.[bert,tracking]'
```

`experiments/train_bert_mlm.py` uses one Hugging Face BERT training path for
all attention baselines. `mha` leaves the official checkpoint's attention untouched.
`gqa` keeps all 12 query heads while sharing key/value projections across four
KV heads by default, while `mqa` shares one key/value head across all queries.
`collaborative` uses shared Q/K projections, a learned diagonal mixing vector
for each of the 12 attention heads, and independent per-head values. These
conventional baselines are distinct from GT-MHA's
`--sdpa-gqa-mode`, which only selects an internal execution path.
`gt_mha_exact` replaces only `bert.encoder.layer[*].attention.self`; the
checkpoint-initialized embeddings, feed-forward blocks, output projections,
normalization layers, and MLM head remain unchanged. GT-MHA's reduced Q/K/V
bases are initialized by averaging contiguous groups of checkpoint heads, and
the replacement audit is written to `bert_gt_mha_manifest.json`.

Checkpoint conversion followed by MLM recovery:

```bash
python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization checkpoint \
  --attention-type gt_mha_exact \
  --num-base-heads 4 \
  --num-generators 8 \
  --enforce-paper-gt-mha \
  --output-dir outputs/bert-base-gt-mha-recovery
```

The primary from-scratch comparison uses the same command and data pipeline
for both attention mechanisms, changing only `--attention-type`:

```bash
python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type mha \
  --output-dir outputs/bert-base-mha-scratch

python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type gqa \
  --num-kv-heads 4 \
  --output-dir outputs/bert-base-gqa4-scratch

python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type mqa \
  --output-dir outputs/bert-base-mqa-scratch

python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type collaborative \
  --output-dir outputs/bert-base-collaborative-scratch

python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type gt_mha_exact \
  --output-dir outputs/bert-base-gt-mha-scratch
```

For checkpoint conversion, GQA copies BERT's query projection exactly and
averages contiguous groups of key/value heads. Random GQA initialization uses
BERT's configured normal initializer. `--num-kv-heads` must divide 12. The
explicit `mqa` type fixes this value at one. Collaborative checkpoint conversion
averages Q/K heads into shared projections, copies BERT's full value projection,
and initializes each diagonal mixing vector to identity.

GT-MHA base Q/K/V projections use BERT's configured normal initializer rather
than the generic module's fan-out-dependent Xavier initializer. Training keeps
dynamic 80/10/10 MLM masking, while validation is pre-masked once with the
model-independent `--validation-mask-seed` (default `17029`) so every attention
variant is evaluated on identical masked tokens. The GT-MHA head-coordinate
parameters `theta` and `value_theta`, plus collaborative MHA's `mixing_vector`,
are placed in the optimizer's zero-weight-decay group; generator matrices retain
ordinary weight decay. Because this
changes optimizer parameter groups, start a fresh run instead of resuming a
checkpoint created by an older version of the BERT runner.

Generator mixing defaults to the paper configuration, `softmax`. Experimental
raw signed coefficients can be selected only with paper enforcement disabled:

```bash
python experiments/train_bert_mlm.py \
  --model-name-or-path google-bert/bert-base-uncased \
  --initialization random \
  --attention-type gt_mha_residual \
  --generator-mixing none \
  --no-enforce-paper-gt-mha \
  --use-sdpa \
  --fuse-base-qkv \
  --sdpa-gqa-mode native \
  --output-dir outputs/bert-base-gt-mha-raw-mixing
```

The optimized flags preserve the GT-MHA computation while using a fused base
Q/K/V projection and PyTorch SDPA with four native GQA key/value heads. Calls
that request attention weights or supply a BERT head mask automatically retain
the explicit reference path.

The DeltaAI launcher accepts `ATTENTION_TYPE=mqa`,
`ATTENTION_TYPE=gqa,NUM_KV_HEADS=4`, and `ATTENTION_TYPE=collaborative`.

This runner implements fixed-length masked-language-model training. It does not
silently add teacher-student distillation or next-sentence prediction. The
DeltaAI launcher `deltaai/train_bert_mlm_gh200x4.slurm` supports all of these
attention types with identical optimization settings.

Fine-tune the official MHA checkpoint or its direct GT-MHA conversion on any
GLUE task with the same runner and hyperparameters:

```bash
python experiments/finetune_bert_glue.py \
  --task mnli \
  --attention-type mha \
  --output-dir outputs/bert-base-mha-mnli

python experiments/finetune_bert_glue.py \
  --task mnli \
  --attention-type gt_mha_exact \
  --output-dir outputs/bert-base-gt-mha-mnli
```

The GLUE path starts both variants from the same official checkpoint and uses
ordinary supervised task loss. It does not use a teacher or distillation loss.
Use `deltaai/finetune_bert_glue_gh200x4.slurm` for matched DeltaAI runs; vary
`TASK`, `ATTENTION_TYPE`, and `SEED` through `sbatch --export`.

On the DeltaAI `sshaik4` account, populate the shared offline cache before
submitting either training launcher:

```bash
sbatch deltaai/precache_bert_data_sshaik4.slurm
```

The cache is stored under
`/work/hdd/biad/sshaik4/lgma_data/.hf_cache`; both DeltaAI BERT launchers use
that path with Hugging Face offline mode enabled.

The equivalent Palmetto pre-cache job is:

```bash
sbatch palmetto/precache_bert_data_shahils.slurm
```

It uses `/scratch/shahils/hf_cache` and installs the missing `evaluate` package
into the existing `llava_video` Conda environment when necessary.

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

### Post-training paper evaluation

Use a deterministic complete-split evaluation for the values reported in a
paper. Unlike the quick validation checks during training, this command visits
every next-character target in fixed, non-overlapping context windows exactly
once. Use the same checkpoint step, split, precision, and batch size for every
model.

```bash
python experiments/evaluate_tinystories_full.py \
  --checkpoint /path/to/run/checkpoint_step_250000.pt \
  --split val \
  --batch_size 64 \
  --device cuda \
  --precision bf16 \
  --output /path/to/paper_results/model_seed0_full_eval.json
```

For a separate held-out test file, add `--split file --eval_data_path
/path/to/test.txt`. The evaluator intentionally refuses characters that are
absent from the checkpoint vocabulary instead of silently changing tokenization.

Measure single-GPU prefill latency, greedy KV-cached autoregressive decoding,
throughput, peak CUDA memory, parameter counts, attention-score FLOPs, and both
analytic and measured KV-cache size with:

```bash
python experiments/benchmark_tinystories_inference.py \
  --checkpoint /path/to/run/checkpoint_step_250000.pt \
  --context_lengths 128 256 480 \
  --batch_size 1 \
  --decode_tokens 32 \
  --warmup 5 \
  --repeats 20 \
  --device cuda \
  --precision bf16 \
  --profile_flops \
  --output /path/to/paper_results/model_seed0_inference.json \
  --csv /path/to/paper_results/model_seed0_inference.csv
```

The benchmark performs one cache-producing prefill followed by single-token
cached decoding. Prompt length plus `--decode_tokens` cannot exceed the
checkpoint's trained positional context (currently 512). It records both the
architecture-level cache estimate and the bytes in the returned per-layer cache;
these values should agree per token and layer. Profiler FLOPs are operator-reported
and may omit unsupported fused or custom kernels.

Finally, copy `experiments/paper_runs.example.json`, add one entry per seed, and
combine training logs, deterministic evaluation, inference measurements, and
Slurm accounting into paper-ready JSON, CSV, and Markdown tables:

```bash
python experiments/aggregate_paper_results.py \
  --manifest experiments/paper_runs.json \
  --output_dir /path/to/paper_results/tables
```

Run the aggregator on DeltaAI to collect cumulative GPU-hours from every Slurm
job segment. Use `--skip_slurm` on a machine without `sacct`, or `--fetch_wandb`
to add W&B run state and URLs. Group names in the manifest define the rows used
for multi-seed mean and standard-deviation summaries.

### FineWeb-Edu preprocessing

The FineWeb pipeline creates deterministic document-level splits, trains one
shared byte-level BPE tokenizer using training documents only, and packs each
source Parquet shard into resumable little-endian `uint16` token files. Every
document is terminated by `<eos>`. Validation and test assignments use a stable
BLAKE2 hash of the FineWeb document ID, so all attention variants receive
identical data without materializing duplicate Parquet datasets.

```bash
python experiments/prepare_fineweb.py \
  --input_dir /path/to/fineweb_edu_10BT/sample/10BT \
  --output_dir /path/to/fineweb_edu_10BT_packed_v1 \
  --phase all \
  --vocab_size 32768 \
  --validation_basis_points 10 \
  --test_basis_points 10 \
  --tokenizer_sample_modulus 50 \
  --batch_size 256
```

The output includes `split_manifest.json`, `tokenizer.json`, tokenizer metadata,
one binary file per source shard and split, per-shard integrity metadata, and an
aggregate `dataset_metadata.json`. Completed shards are verified and reused when
the command is resumed. The DeltaAI batch template is
`deltaai/prepare_fineweb_edu.slurm`.

Train any supported attention architecture against the packed corpus with the
same memory-mapped loader:

```bash
torchrun --standalone --nproc_per_node=4 experiments/train_text_lm.py \
  --packed_data_dir /path/to/fineweb_edu_10BT_packed_v1 \
  --attention mha \
  --device cuda \
  --precision bf16 \
  --batch_size 64 \
  --grad_accum_steps 8 \
  --context_length 512 \
  --steps 9512
```

Training batches are sampled uniformly over valid start positions across all
binary shards, with replacement. At the batch configuration above, 9,512
optimizer steps provide a one-corpus-equivalent 9.974-billion-token budget.
The sampler has a dedicated, checkpointed random-number stream, so runs using
the same seed receive identical training batches regardless of attention
implementation and resume at the exact next batch after preemption. In-training
packed validation also uses a fixed random seed, making the sampled validation
batches identical across steps and runs. The original `--data_path`
character-text backend remains available for TinyStories.

On DeltaAI, `deltaai/train_fineweb_ddp4.slurm` provides the common full-training
configuration for MHA, Collaborative MHA, and LGMA. Submit three resumable
12-hour segments with `deltaai/submit_fineweb_chain.sh`; later segments run only
when the preceding segment ends before reaching the final checkpoint.
