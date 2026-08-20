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
