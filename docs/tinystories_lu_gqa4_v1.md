# Frozen TinyStories training comparison, version 1

**Question.** How does fixed-base LU GT-MHA compare with native GQA4 on next-character prediction at the same model width, training data, update count and token exposure? The primary result is the paired difference in final full-validation negative log-likelihood (NLL), in nats per target character. A negative difference favors LU. The estimand applies to this shared training recipe. The two architectures have different parameter counts and can consume different wall time and FLOPs.

The freeze precedes the six comparison runs. It draws on prior TinyStories experiments, including their published validation results. Earlier runs with different initializers, schedules, hardware layouts or endpoints are historical context. The official validation file is a known benchmark split. A claim about a new blind test set would require another dataset or split.

## Six fresh training runs

| Setting | Frozen choice |
|---|---|
| Arms | Compact fixed-base LU GT-MHA and native GQA4 |
| Paired seeds | 0, 100, 200 for each architecture |
| Backbone | 12 pre-layer-normalized causal Transformer blocks, width 1,024, feed-forward width 4,096, 16 attention heads of width 64, learned absolute positions, context 512, dropout 0.1 |
| Fixed-base LU | Four shared Q/K/V bases. Heads 0, 4, 8 and 12 have fixed identity transforms. The other 12 use positive-pivot LU with initialization scale 0.02. There are no generator banks, separate value transforms, routers or auxiliary penalties |
| GQA4 | 16 query heads and four shared K/V heads |
| Score scaling | Standard division by the square root of head width in both arms. No learned temperature |
| Training data | Canonical UTF-8 `TinyStoriesV2-GPT4-train.txt`. Uniform random windows sampled with replacement by a dedicated CPU generator per rank, seeded with run seed plus rank. Training windows can cross the original document delimiters |
| Global batch | One node, four NVIDIA GH200 120GB GPUs, one rank per GPU, 256 sequences per rank per microbatch, two accumulation steps. Exactly 2,048 sequences and 1,048,576 target characters per optimizer update |
| Training horizon | Exactly 250,000 optimizer updates, hence 262,144,000,000 **sampled** targets per run. Repeated windows count toward exposure |
| Optimizer | AdamW over all trainable parameters, including biases and normalization weights. Betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, global gradient norm clipped at 1.0 after accumulation |
| Learning rate | No warmup. Hold 3e-4 through update 50,000. Then cosine decay to 1e-4 at update 250,000. The schedule and training horizon both end at 250,000, including on resume |
| Numerics | BF16 CUDA autocast with FP32 master weights and optimizer state, TF32 disabled, eager training. Reference environment uses PyTorch 2.12.0+cu130 and Triton 3.7.0 |
| Monitoring | Log every 100 updates. Every 1,000 updates evaluate the same seeded panel of eight validation batches. Save every 1,000, retain the latest two regular checkpoints and 50,000-update milestones. Periodic diversity diagnostics are disabled |

After update 50,000, the learning rate for update `u` is `1e-4 + 1e-4 × [1 + cos(π × (u − 50,000) / 200,000)]`. The value at update 150,000 is 2e-4. A machine allocation ending early interrupts the run at its current position on this curve.

Common backbone and K/V/output projections use matched starting weights within each seed pair. Query projections differ in shape, and the LU factors have their own near-identity initializer. The step-zero record identifies the actual tensors and initialization. Fused LU training is enabled. Primary quality evaluation uses the compact trained checkpoint. Exported and packed inference are separate deployment measurements.

## Data identity and split discipline

| Artifact | SHA-256 |
|---|---|
| GPT4 training file, 2,227,753,162 bytes | `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443` |
| GPT4 validation file, 22,502,601 bytes | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |

Both arms use the frozen ordered list of 230 character IDs. The historical vocabulary was constructed from the union of the two files. Training contains 228 distinct characters. U+0093 and U+0094 occur only in validation. We fix the mapping before these runs, retain the same vocabulary for comparability, and draw gradient-bearing examples exclusively from the training file.

A byte-level audit split both official files at `<|endoftext|>`, stripped surrounding ASCII whitespace and compared SHA-256 hashes of all nonempty records. It found 2,717,495 training records and 27,630 unique nonblank validation records, with **zero exact matches between splits**. This check only detects exact matches under that normalization.

## Endpoint and analysis

1. Use the checkpoint at exactly update 250,000 from each run, regardless of the sampled validation curve. Use all three seed pairs. Report crashes, preemptions and incomplete runs. Resume an interrupted run from its own recorded state. A learning-rate change, replacement seed, hardware-layout change, or a shorter schedule is a separately labeled protocol revision or sensitivity experiment.
2. Evaluate every eligible official validation segment on a single GH200 in evaluation mode, batch size one, BF16 autocast and FP32 cross-entropy. Split at the document delimiter, preserve the original characters, and keep segments longer than one character. Process up to 512 inputs per chunk, reset positions and context between chunks, and score every within-segment next character once. Accumulate summed loss in higher precision and divide once by the total target count.
3. Expect **27,631 segments and 22,106,566 targets**. The historical rule includes one two-newline segment contributing one target. Aggregate by target count; averaging story perplexities would give short stories disproportionate weight.
4. Publish all six per-run NLLs and perplexities, the mean and sample SD for each arm, and the three within-seed NLL differences (LU minus GQA4). Report the mean paired difference with a descriptive 95% paired Student interval using two degrees of freedom. The experimental units are the three training seeds. With three pairs, uncertainty can be large. A near-zero difference alone does not establish equivalence.

There is **no per-arm hyperparameter search, early stopping or best-checkpoint selection** in the primary comparison. Monitoring supports fault detection and reporting the learning curve. The recipe does not estimate either architecture's best attainable loss under separate tuning. Any later tuning study needs the same search budget and selection rule for both arms.

Report trainable parameters, actual sampled targets, training targets/s and GPU-hours alongside quality. For each arm and seed, retain the allocation IDs. Sum allocated GPUs times elapsed hours across every training allocation, including preempted attempts. The trainer's elapsed-time counter starts again on resume. Compute end-to-end throughput from retained training targets divided by total training allocation wall time and report evaluation cost separately. Benchmark inference separately at matched batch, context, precision and cache layout. Compare ordinary GQA4 against exported LU and apply any additional QKV packing to both. Report training parameters and exported weight storage separately.

## Running the frozen comparison

Use Git tag `tinystories-lu-gqa4-v1`. The protocol JSON has SHA-256 `543154cf73832ca6bd223052889de0a5a6b9d7dfef467a7f97aa651670b2d78b`. Inspect the lock and render both arms' effective launch commands before requesting a training allocation. Assign each arm and seed a unique run directory in advance, for example `$COHORT/$ARM/seed-$SEED`, and record every attempt there. Set `TRAIN` and `VALID` to the official files and `RUN` to that run directory. The launcher runs one seed on the four-GPU allocation.

```sh
# Reference environment on DeltaAI
module load python/miniforge3_pytorch/2.12.0
export PYTHONNOUSERSITE=1
python experiments/tinystories_lu_protocol.py show
python experiments/tinystories_lu_protocol.py print-command --arm gt_mha_lu --seed 0 \
  --train-path "$TRAIN" --val-path "$VALID" --output-dir "$RUN"
python experiments/tinystories_lu_protocol.py launch --arm gt_mha_lu --seed 0 \
  --train-path "$TRAIN" --val-path "$VALID" --output-dir "$RUN"
```

Repeat for `--arm gqa`, with a distinct run directory, and for seeds 100 and 200. To resume, use the same arm, seed and run identity and pass `--resume-checkpoint` pointing to its last complete checkpoint. Keep the initial weights, manifest and endpoint checkpoint together.

```sh
python experiments/evaluate_lu_protocol.py evaluate --arm gt_mha_lu --seed 0 \
  --run-dir "$RUN" --data-dir "$DATA_DIR" --results-dir "$RESULTS"
python experiments/evaluate_lu_protocol.py aggregate --results-dir "$RESULTS"
```

Run `evaluate` for each of the six runs before aggregation. Keep the machine lock, source receipt, checkpoint hashes and per-run evaluation records with the result table.
