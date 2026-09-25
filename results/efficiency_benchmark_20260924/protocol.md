# Paper efficiency benchmark protocol

## Submitted jobs

- DeltaAI job `3202780`: TinyStories prefill, KV-cached decode, sliding-window decode, KV-cache accounting, FLOPs, and phase-specific peak memory.
- DeltaAI job `3202781`: TinyStories forward/backward/AdamW training throughput and peak memory.
- DeltaAI job `3202782`: BERT MLM forward/backward/AdamW training throughput and peak memory.

All jobs request one GH200 from the `ghx4` partition under the `bifg-dtai-gh` account. This avoids cross-device comparisons and isolates single-device implementation efficiency.

## TinyStories inference

- Checkpoints: the five pinned checkpoints used by the paper's full-validation comparison.
- Precision: BF16.
- Batch size: 1.
- Prefill lengths: 128, 256, 448, and 512 character tokens.
- Cached decode: 64 supplied tokens after prefixes of length 128, 256, and 448.
- Sliding-window decode: 128 supplied tokens with full 512-token window recomputation.
- Timing: 5 warm-up repetitions and 30 timed repetitions.
- Decode tokens: fixed across methods with seed `20260924`; no sampling or EOS stopping occurs in timed regions.
- Execution-order check: one forward model order and one reverse model order on the same GPU.
- Memory: allocated and reserved peaks are measured separately for prefill, cached decode, and sliding-window decode.
- Cache: tensor storage is measured directly and checked against the analytical byte count.
- FLOPs: dominant analytical attention FLOPs are reported for every context length. PyTorch operator-profiled FLOPs are retained as a secondary diagnostic because fused operations may be omitted.

## TinyStories training

- Precision: BF16.
- Per-device batch: 256.
- Sequence length: 512.
- Timing: 3 warm-up steps and 10 timed optimizer steps.
- Execution-order check: one fixed order and its reverse.
- Measurement includes forward, backward, and AdamW update time.

## BERT training

- Architectures: MHA, GQA, Collaborative MHA, and residual GT-MHA with `C=4` and eight generators.
- Precision: BF16.
- Per-device batch: 32; two accumulated microbatches per optimizer step.
- Sequence length: 512.
- Gradient checkpointing: enabled, matching pretraining.
- Timing: 3 warm-up optimizer steps and 10 timed optimizer steps.
- Inputs and 15% MLM label mask are fixed across methods with seed `20260924`.
- Execution-order check: one fixed order and its reverse.

## Reporting safeguards

Throughput results are implementation- and hardware-specific. Parameter counts, analytical projection FLOPs, and persistent KV-cache bytes are architecture-level quantities. Peak allocated memory includes weights and transient tensors; it must not be described as KV-cache memory. The two execution passes measure run-order drift, not training-seed uncertainty.
