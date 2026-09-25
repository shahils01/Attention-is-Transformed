# Inference reporting plan

## Author-requested generation changes

The main run uses a maximum of **3,000 character tokens including the stop delimiter**, five samples per prefix, and eight concurrent Slurm tasks with **one A100 40GB PCIe per checkpoint**. The 512-character context is unchanged. Normal delimiter stopping remains enabled. A larger cap cannot guarantee termination; cap hits must remain visible. The completed 2,048-cap engineering run stays unchanged as a historical validation artifact.

Per-story generation timings are synchronized around the entire call, after warm-up. They include prompt encoding, sampling, host token extraction, stop checks, decoding, and window recomputation; checkpoint/data loading and JSON logging are outside the interval. Report these as operational end-to-end measurements, not isolated model-forward throughput. Aggregate tokens/second as total generated character tokens divided by total generation seconds, not the mean of per-story rates. Story duration is affected by generated length and termination behavior.

## Controlled benchmark for paper tables

The manuscript already promises TinyStories prefill, decoding, and persistent KV-cache metrics, while BERT training throughput/memory are separate measurements (iclr2026_conference.tex around lines 545–568). Prefer inference results alongside training measurements. If replacing training measurements, revise the scope and remove unsupported training-efficiency claims rather than treating the two as interchangeable.

Use a separate fixed-workload benchmark on the same physical A100 40GB GPU for every model when feasible. Load checkpoints outside timing; fix PyTorch/CUDA, bf16, CPU threads, SDPA backend policy, compilation, and batch settings. No simultaneous work on that GPU. Record model/device config, driver, temperature/clocks/power state where accessible. Randomize model order across repeated passes to check run-order drift. Do not silently compare optimized one-model kernels against unoptimized baselines.

Primary batch size 1; throughput sensitivity at batch sizes 8 and 32, reported separately. Do not raise batch size only for one model to inflate throughput. Report OOM honestly if encountered.

1. **Prefill:** prefix lengths 128, 256, 448, and 512 character tokens. Measure cache-producing forward latency (median/p95 milliseconds) and character tokens/second. This is model prefill latency, not full serving time-to-first-token.
2. **Cached decode:** prefixes 128, 256, and 448; exactly 64 forward decode steps, so cached length stays <=512. Use a fixed predeclared token stream common to models to isolate computation; disable EOS stopping only for this synthetic timing workload. Exclude setup prefill from decode timing. Report aggregate character tokens/second and milliseconds per decode step, stating batch size. This is distinct from story sampling throughput.
3. **Sliding-window decode:** start with a full 512-character prefix and time 128 further steps using the validated full-window recomputation behavior. Retain the common supplied token stream, fixed length, and no sampling. Report separately; after the window shifts, our implementation recomputes the window rather than retaining a growing KV cache.
4. **Memory:** measure persistent K/V tensor bytes directly and normalize by batch, cached tokens, and layers. Also measure peak allocated/reserved GPU memory for prefill and decode separately, after clearing stale tensors and resetting peak statistics. Weight memory, transient activations, and allocator reservations are not KV-cache memory.

Use 5 full warm-up repetitions and 30 timed repetitions per configuration, CUDA synchronization at timing boundaries, and retain every raw timing. Report repeat-level median and p95 as measurement variability, not training-seed uncertainty. Repeat the benchmark in a second pass with changed model order before publication. Keep synthetic timing inputs separate from the quality-evaluation dataset.

Existing code: experiments/benchmark_tinystories_inference.py already implements much of prefill/cached-decode timing and actual cache accounting. It needs the verified QK-capable loader, a sliding-window workload, full-length warmups, and clean per-phase memory measurement for this protocol. It currently times greedy autoregression; the proposed common-token timing stream is a documented adaptation. Do not run the unmodified script and label it this full protocol.

Primary source on warm-up/synchronization: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html

## Interpretation

Fewer parameters and a smaller persistent KV cache do not guarantee lower latency. Transformation computation, transient expansions, memory bandwidth, kernels, and the sliding-window implementation determine realized speed. Report measured results even if GT-MHA is slower. Eight concurrent generation jobs shorten time to collect the evaluation; they do not constitute an eightfold per-model inference speedup.

Suggested paper columns: method, prefill ms (B=1,L=256), cached decode char-tokens/s (B=1,L=256,N=64), sliding decode char-tokens/s (B=1,L=512,N=128), KV bytes/token/layer, and peak allocated memory. Put other lengths/batches and end-to-end generation timing distributions in the appendix. Label tokens as character tokens; these rates are not directly comparable to BPE-token rates of external models.

Status: per-story synchronized timing is implemented in the main generation harness; the controlled benchmark protocol is designed but not yet executed. No inference or training speedup is claimed.
