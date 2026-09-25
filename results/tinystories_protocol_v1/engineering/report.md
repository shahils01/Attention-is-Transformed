# Completed TinyStories engineering run

All eight published checkpoints passed on Palmetto. All 160 development generations are saved locally and remotely. No LLM quality scoring or full 200-prompt evaluation has been run.

## Verification

- All eight files match the pinned Hugging Face SHA-256 values.
- Strict model-state loading succeeded, including completed QK identity. Unfinished QKV identity was excluded.
- Original train+validation tokenizer reconstruction produced 230 characters; hashes are in tokenizer.json.
- Cached/full checks passed at lengths 127, 128, 511, 512, 513, 514, including window shifts.
- A 64-character generation repeated exactly at the same seed for each model.
- Raw-token counts, text, stop reasons, and row identities passed the output validator.
- Local executable source and runner hashes match all eight GPU tasks. macOS archive metadata sidecars are recorded remotely but are not executable modules.

Maximum probability differences: fp32 cache/full 2.8376235e-10; bf16 cache/full 2.1994114e-05; bf16/fp32 0.0016551614. Checks cover selected positions in one development story per model, not every possible input.

## Termination

| Model | Delimiter stop | Cap hits | Mean completion characters |
|---|---:|---:|---:|
| Collaborative MHA | 19/20 | 1/20 | 700.5 |
| GQA | 20/20 | 0/20 | 714.5 |
| GT-MHA QK identity B4G8H16 | 19/20 | 1/20 | 709.1 |
| GT-MHA exact | 20/20 | 0/20 | 759.2 |
| GT-MHA quad | 20/20 | 0/20 | 667.7 |
| GT-MHA residual | 19/20 | 1/20 | 775.2 |
| MHA | 20/20 | 0/20 | 654.0 |
| MQA | 20/20 | 0/20 | 623.2 |

157/160 emitted the delimiter; 3/160 hit the 2,048-character cap. 154/160 required a sliding-window shift. Delimiter stopping does not establish narrative closure or coherence. These 20 engineering prompts, with one sample per model, are not a model-quality ranking.

## Execution

Staging job 16084720 and corrected A100 array 16084758_0 through _7 completed; all GPU exit codes were 0. Initial array 16084722 stopped before generation because PyTorch 2.2 mmap requires a string filename. This compatibility issue was fixed without changing checkpoints or decoding. Logs are preserved.

PyTorch 2.2.0+cu121, bf16; temperature 1, top-k 20; identical prompt-specific seeds; cap 2,048 character tokens including delimiter; no repetition trimming. All used A100s: exact GT-MHA received a 40GB model, the others 80GB. Timings are not a controlled architecture throughput benchmark. Use a common GPU subtype for the full run where practical.

MHA checkpoint metadata says step 266000; MQA 249800; others 250000. We record these without claiming equal training budgets.

Remote root: `/scratch/shahils/tinystories_eval_20260919`. Local outputs include raw generations, model metadata, numerical checks, source manifests, and logs.

## Next stage

Freeze the disjoint 200-prefix evaluation set and generation manifest, including training-overlap checks. Five samples per prefix across eight checkpoints gives 8,000 completions. Retain raw outputs and the first 320 characters as within-context excerpts. Freeze the exact judge configuration before inspecting final evaluation outputs; use the appropriate rubric for each mode and paired prompt-bootstrap intervals. No paid judge calls have been made.
