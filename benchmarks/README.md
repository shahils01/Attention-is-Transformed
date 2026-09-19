# TinyStories LGMA vs. MHA qualitative benchmark

`tinystories_100_prompts.jsonl` contains 100 original story-prefix prompts for a
paired comparison of the LGMA and MHA TinyStories checkpoints. The design
follows the qualitative setup in Figure 6 of the TinyStories paper: the model
receives only a narrative prefix and generates the continuation. Rubric metadata
is never included in the model input.

The benchmark contains 13 prompts in each of the first four themes and 12 in
each of the remaining four themes:

- `simple_story_continuation`
- `character_object_consistency`
- `cause_and_effect_reasoning`
- `multi_character_interactions`
- `emotional_development`
- `moral_lesson_completion`
- `long_range_detail_retention`
- `age_appropriate_language`

All prompts use ASCII characters and fit inside the checkpoints' 512-character
context window. The fixed prompt ID determines its sampling seed, so both
architectures are compared under identical decoding settings and seed schedules.

Run locally or on an interactive GPU node with:

```bash
python experiments/compare_tinystories_prompts.py \
  --checkpoint lgma=ckpts/tinystories_lgma_quad_b4_h16_g8/checkpoint_step_250100.pt \
  --checkpoint mha=ckpts/tinystories_mha_h16/checkpoint_step_266200.pt \
  --data_path /path/to/TinyStoriesV2-GPT4-train.txt \
  --val_data_path /path/to/TinyStoriesV2-GPT4-valid.txt \
  --device cuda \
  --precision bf16
```

The DeltaAI batch wrapper is `deltaai/infer_tinystories_prompt_comparison.slurm`.
The inference output is append-only and resumable. A rerun skips every completed
model/prompt pair and regenerates the side-by-side Markdown report.

## Blind evaluation of 30 matched prompts

After both models have generated all completions, create a balanced 30-prompt
blind subset. Candidate A/B ordering is randomized independently for each
prompt:

```bash
python experiments/evaluate_blind_tinystories.py prepare \
  --completions outputs/tinystories_lgma_mha_100/completions.jsonl \
  --num-prompts 30 \
  --models lgma mha \
  --seed 0 \
  --output-dir outputs/tinystories_lgma_mha_100/blind_eval
```

This writes three files:

- `blind.jsonl`: prompts, evaluation focus, and Candidate A/B completions; this
  is the only data file that may be sent to a judge.
- `blind_mapping.jsonl`: private model/checkpoint identities and content hashes;
  never send this file to the judge.
- `judge_instructions.md`: the exact 1-10 rubric for manual evaluation.

For automated OpenAI API evaluation, install the optional dependency, set the
API key, and run the judge. The command deliberately has no mapping-file option:

```bash
pip install -e ".[eval]"
export OPENAI_API_KEY=your_key_here

python experiments/evaluate_blind_tinystories.py judge \
  --blind-file outputs/tinystories_lgma_mha_100/blind_eval/blind.jsonl \
  --output-file outputs/tinystories_lgma_mha_100/blind_eval/blind_scores.jsonl \
  --model gpt-5.6
```

If the key was issued by Clemson RCD rather than OpenAI, send requests through
the RLS OpenAI gateway. Be on the Clemson network or CUVPN, then list the exact
models currently available to the gateway:

```bash
export RCD_LLM_API_KEY=your_rcd_key_here
curl https://llm.rcd.clemson.edu/openai/v1/models \
  -H "Authorization: Bearer $RCD_LLM_API_KEY"
```

Choose a model name from that response and run:

```bash
python experiments/evaluate_blind_tinystories.py judge \
  --blind-file outputs/tinystories_lgma_mha_100/blind_eval/blind.jsonl \
  --output-file outputs/tinystories_lgma_mha_100/blind_eval/blind_scores.jsonl \
  --provider rcd-openai \
  --model gpt-5.6-sol
```

`gpt-5.6-sol` is only an example here; it must appear in the gateway model list.
The `openai` provider uses `OPENAI_API_KEY`, while `rcd-openai` uses
`RCD_LLM_API_KEY` and `https://llm.rcd.clemson.edu/openai/v1`.

Alternatively, upload only `blind.jsonl` to a judge and use
`judge_instructions.md`. Save its JSONL response as `blind_scores.jsonl`.

Finally, restore identities locally and produce per-model, per-theme, and paired
summaries:

```bash
python experiments/evaluate_blind_tinystories.py unblind \
  --blind-file outputs/tinystories_lgma_mha_100/blind_eval/blind.jsonl \
  --mapping-file outputs/tinystories_lgma_mha_100/blind_eval/blind_mapping.jsonl \
  --scores-file outputs/tinystories_lgma_mha_100/blind_eval/blind_scores.jsonl \
  --output-dir outputs/tinystories_lgma_mha_100/blind_eval
```

The resulting `unblinded_scores.jsonl` and `scores.csv` contain one row per
candidate. `candidate_comparison.csv` and `candidate_comparison.md` show the
Candidate A/B identity and scores side by side for every prompt. `summary.json`
and `summary.md` contain the aggregate LGMA/MHA comparisons.

## Sequential five-sample evaluation of every checkpoint

To recursively discover both `checkpoint_step_*.pt` and `checkpoint_final.pt`
under `ckpts`, load one checkpoint at a time, and generate five independent
continuations for all 100 prompts on a single GPU, run:

```bash
python experiments/compare_tinystories_prompts.py \
  --checkpoint_dir ckpts \
  --checkpoint_glob '**/checkpoint*.pt' \
  --prompts_file benchmarks/tinystories_100_prompts.jsonl \
  --data_path data/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val_data_path data/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --device cuda \
  --precision bf16 \
  --max_new_tokens 300 \
  --temperature 0.8 \
  --top_k 20 \
  --seed 0 \
  --samples_per_prompt 5 \
  --output_dir outputs/tinystories_all_checkpoints_100_x5
```

Discovery is deterministic. The combined `completions.jsonl` is used for blind
evaluation, while each checkpoint also gets its own resumable
`checkpoints/<name>/completions.jsonl` and `report.md`. A run manifest records
the exact ordered checkpoint set and repeat count. Seeds are matched across
checkpoints: sample 1 uses seeds 0-99, sample 2 uses 100-199, and so on for the
default 100-prompt file. On DeltaAI, the equivalent batch wrapper is
`deltaai/infer_tinystories_all_checkpoints.slurm`; it defaults to five samples.

After generation finishes, create blind records containing every checkpoint as
an anonymous candidate. Zero means all available prompts; pass
`--num-prompts 30` for a smaller pilot:

```bash
python experiments/evaluate_blind_checkpoints.py prepare \
  --completions outputs/tinystories_all_checkpoints_100_x5/completions.jsonl \
  --manifest outputs/tinystories_all_checkpoints_100_x5/run_manifest.json \
  --num-prompts 0 \
  --output-dir outputs/tinystories_all_checkpoints_100_x5/blind_eval
```

With five samples this creates 500 blind records: one record for each
prompt/repeat, with all checkpoints represented as anonymous candidates.
Candidate positions are anonymized and balanced across those records. Only
`blind.jsonl` goes to the judge; keep `blind_mapping.jsonl` private. Judge all
anonymous candidates with GPT-5.6 Sol through Clemson RCD:

```bash
export RCD_LLM_API_KEY=your_rcd_key_here

python experiments/evaluate_blind_checkpoints.py judge \
  --blind-file outputs/tinystories_all_checkpoints_100_x5/blind_eval/blind.jsonl \
  --output-file outputs/tinystories_all_checkpoints_100_x5/blind_eval/blind_scores.jsonl \
  --provider rcd-openai \
  --model gpt-5.6-sol \
  --reasoning-effort medium
```

Finally, restore checkpoint identities and generate the per-checkpoint
leaderboard, per-theme results, and candidate-level table:

```bash
python experiments/evaluate_blind_checkpoints.py unblind \
  --blind-file outputs/tinystories_all_checkpoints_100_x5/blind_eval/blind.jsonl \
  --mapping-file outputs/tinystories_all_checkpoints_100_x5/blind_eval/blind_mapping.jsonl \
  --scores-file outputs/tinystories_all_checkpoints_100_x5/blind_eval/blind_scores.jsonl \
  --output-dir outputs/tinystories_all_checkpoints_100_x5/blind_eval
```

The main reports are `leaderboard.csv`, `summary.json`, `summary.md`,
`scores.csv`, `prompt_summary.csv`, `theme_summary.csv`, and
`candidate_scores.md`. `prompt_summary.csv` reports the mean and sample
variance across the five continuations for every prompt/checkpoint pair;
`theme_summary.csv` and `leaderboard.csv` report pooled mean and sample
variance by theme and checkpoint. Scores cover Grammar, Consistency,
Creativity, and Plot from 1 through 10; the overall score gives those four
metrics equal weight. Creativity is conditional on consistency and cannot
receive a higher score than consistency. The command also creates
`single_prompt_comparison.json`, with
the prompt and all checkpoint continuations; `answer_text` is sample 1 and the
`samples` array contains all five continuations and scores. Add
`--example-prompt-id ts049` to select a particular evaluated prompt instead of
the first one. Add `--num-example-prompts 5` to create five numbered files in
the `prompt_comparisons` directory; each file contains one prompt and every
checkpoint's answers and scores.

Rubric version 2 added the Plot metric and consistency-gated Creativity. Scores
created with the earlier three-metric rubric do not contain Plot and cannot be
upgraded during unblinding; judge the same blind file again to a new score file
before generating a four-metric summary.

## Overleaf generated-text tables

Export any scored prompt-comparison JSON files as TinyStories-style LaTeX
tables with Model, Generated text, and Scores columns:

```bash
python experiments/export_tinystories_overleaf.py \
  outputs/tinystories_all_checkpoints_100/blind_eval_30_v2/prompt_comparisons/prompt_comparison_01.json \
  outputs/tinystories_all_checkpoints_100/blind_eval_30_v2/prompt_comparisons/prompt_comparison_03.json \
  --output outputs/tinystories_all_checkpoints_100/blind_eval_30_v2/tinystories_examples_tables.tex \
  --fragment
```

Fragment mode is intended for `\\input{tinystories_examples_tables.tex}` in an
existing manuscript. Omit `--fragment` to create a complete standalone Overleaf
document instead.
