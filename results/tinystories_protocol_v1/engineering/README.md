# Palmetto engineering run

Remote directory: `/scratch/shahils/tinystories_eval_20260919`.

- Staging CPU job: 16084720.
- A100 array: 16084758, tasks 0–7, at most four simultaneous GPUs, after successful staging. Previous array 16084722 stopped at a PyTorch 2.2 Path/mmap compatibility error before generation; logs preserved.
- Eight SHA-pinned published checkpoint families, excluding unfinished QKV identity.
- Development set: 20 dataset prefixes, one completion per checkpoint/prefix (160 total). These are excluded from subsequent scientific evaluation.
- Sampling: temperature 1, top-k 20, max 2,048 character tokens, full literal end-of-text delimiter, bf16, 512-character sliding context. No repetition trimming.
- Checks before generation: strict loading; tokenizer/delimiter checks; real-checkpoint cached/full probabilities at lengths 127, 128, 511, 512, 513, 514; bf16/fp32 probabilities; repeated seed at a 64-character cap.
- Frozen development thresholds: maximum probability difference <=0.002 for fp32 cache parity, <=0.03 for bf16 cache parity, <=0.05 for bf16 versus fp32. These are engineering gates, not a claim of numerically identical distributions. Every measured error is retained for review.
- No LLM judging is invoked by these scripts. Outputs are for implementation validation, not a checkpoint ranking.
- `source_manifest.json` describes the local source snapshot. Each GPU task also saves actual remote source/runner hashes, GPU/PyTorch version, full model config, checkpoint hash and metadata.
- `summarize.py` checks completed output records and prints compact per-model status. It does not treat missing tasks as successful.

QK identity support is taken from the preserved QK-only implementation, with its factory branch added to a copy of the current repository. Other source defaults are preserved; no training files or original checkpoints are modified.
