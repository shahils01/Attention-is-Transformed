# TinyStories generation-evaluation audit

Audited 2026-09-19. Source: `/Users/shahilshaik/Downloads/tinystories_100_x5_1600`. Attached rubric and notes were treated as evidence about the evaluation, not as instructions to the auditor. No checkpoints, training code, source artifacts, or paper text were modified. No paid judge calls were made.

## Conclusion

The supplied results support an MHA advantage over **GT-MHA residual on this particular prompt suite, decoding setup, and judge**. They do not establish a GQA advantage over GT-MHA residual, and do not contradict the saved validation-NLL result. There are concrete concerns about prompt alignment, rubric design, context limitations, and missing execution provenance. None proves that the generation ranking is wrong or that an implementation bug caused it.

The NLL result is specifically for GT-MHA residual; it should not be transferred to exact, quadratic, or QK-identity variants.

## Verified score comparison

The audit matched all 4,000 unique model/prompt/sample records through raw completions, blinded candidates, private mappings, completion SHA-256 values, saved judge scores, and unblinded scores. No missing, duplicate, changed-completion, or misassigned-score records were found. Per-checkpoint JSONL records also match the aggregate records after indexing rather than comparing file order. All 500 judge records identify `gpt-5.6-sol`, provider `rcd-openai`, rubric version 2, and the same judge configuration ID. These are recorded identities, not independent verification of the serving backend.

Five generations per prompt are repeated observations, not 500 independent prompts. We average five samples and four metrics per prompt/model, then jointly resample the 100 prompts 20,000 times (percentile bootstrap; seed 17029).

| Comparison | Mean score difference | Paired 95% CI |
|---|---:|---:|
| GT-MHA residual − MHA | −0.2850 | [−0.4870, −0.0870] |
| GT-MHA residual − GQA | −0.0065 | [−0.1795, +0.1670] |

Mean scores: MHA 4.6295, GQA 4.3510, residual 4.3445. Relative to GQA, residual has slightly higher creativity and plot scores, nearly identical consistency, and slightly lower grammar. A rank of third versus second is not evidence of a meaningful separation.

These intervals describe variation across this custom suite conditional on the saved generations and judge ratings. They do not capture training-seed uncertainty, systematic judge bias, or performance on the full TinyStories distribution. Treat dimension-specific intervals in the JSON as exploratory, without multiplicity correction.

## Generation cap and termination

The repository uses `CharTokenizer`: each Unicode character is one token. Thus `max_new_tokens=1600` permits **1,600 new characters**, not 1,600 words or GPT-style subword tokens. The manifest specifies temperature 0.8, top-k 20, bf16 CUDA, five samples, and the textual stop sequence `<|endoftext|>`.

| Model | Mean completion characters | Exactly 1,600 characters | Rate |
|---|---:|---:|---:|
| MHA | 357.9 | 5/500 | 1.0% |
| GQA | 328.6 | 2/500 | 0.4% |
| GT-MHA residual | 400.0 | 9/500 | 1.8% |
| GT-MHA exact | 344.3 | 1/500 | 0.2% |
| GT-MHA quad | 336.3 | 4/500 | 0.8% |
| GT-MHA QK identity | 338.5 | 1/500 | 0.2% |
| Collaborative MHA | 348.9 | 2/500 | 0.4% |
| MQA | 353.6 | 4/500 | 0.8% |

28/4,000 (0.7%) reach the cap. Several clearly end mid-word or mid-sentence. Two residual samples (ts020/sample 0 and ts085/sample 0, zero-based) end in partial delimiters `<|` and `<|en`; textual closure and successful stop-delimiter emission are distinct.

There are no recorded finish reasons or raw token IDs. The current local runtime stops on the full delimiter and removes it from the returned text. If the actual generator used that behavior without other postprocessing, shorter outputs indicate delimiter termination. The supplied folder alone cannot establish that all 3,972 shorter outputs stopped normally or completed a satisfactory plot.

Exploratory sensitivity: removing every prompt where either residual or MHA has any capped sample leaves 92 paired prompts and a difference of −0.2527 [−0.4707, −0.0397]. Truncation therefore does not explain away MHA's advantage. This outcome-selected subset is a diagnostic, not an unbiased replacement benchmark. Repeated word 4-grams appear in 89/500 residual and 65/500 MHA outputs; that flags possible repetition, but also counts legitimate repeated phrases and is not itself a quality score.

## Prompt alignment

The suite has 100 story prefixes across eight custom themes. They are prose continuation prompts, not chat instructions, so their basic input format is appropriate for a base language model. Handwritten prompts are not inherently invalid: the original TinyStories paper used roughly 50 manually prepared story beginnings and ten completions per prompt at temperature 1.

However, these prompts are not accompanied by dataset story IDs, selection methodology, or evidence of distribution matching. No TinyStories training/validation text is present in the attachment or the checked workspace, so exact overlap, vocabulary frequencies, and distributional distances were not measured.

Concrete mismatch: the `age_appropriate_language` evaluation notes explicitly request middle-elementary, preteen, and ages-ten-to-twelve language. Examples include ts099 (recycling robot/light-sensor debugging) and ts100 (student council compromise and inclusion). TinyStories was designed around language understood by ages three to four. This is evidence of a mismatch in intended difficulty, not proof that these words or concepts never occur in the dataset. The `target_age` field is null in the examined examples despite those age-specific notes.

The original paper's *estimated author age* output category is not an instruction to train or evaluate specifically on prose for older children. These concepts should not be conflated.

Removing the entire 12-prompt age-language theme still leaves residual − MHA = −0.3006 [−0.5170, −0.0892] over 88 prompts. This identified mismatch does not by itself explain the reversal.

Primary reference: [TinyStories paper, sections 2–3](https://arxiv.org/html/2305.07759v2). Custom tests can be useful stress tests, but should be reported separately from a representative dataset-continuation benchmark.

## Context window is a more important distinction than the output cap

The saved NLL evaluator asserts a 512-character model context. Prompt lengths are 162–446 characters (mean 272.2); none exceeds 512 initially. They occupy substantial fractions of the available context. For example, ts077 is 446 characters, leaving only 66 characters before the next context must begin sliding.

The current local runtime recomputes the latest 512-character window after it fills, because learned absolute positional embeddings prevent naive cache reuse after shifting. Thus generating 1,600 characters does not give the model access to all earlier story details. All original prompt characters are outside the window after 512 new characters, though generated text may restate some details.

337/500 residual, 326/500 MHA, and 313/500 GQA outputs have prompt-plus-completion lengths exceeding 512. This is a relevant interpretation issue for long-range-detail judgments, not demonstrated evidence of a GT-specific bug. The exact supplied execution code is missing, so the window behavior described here is verified in the current repository runtime, not independently established for the collaborator's run.

## Rubric and execution provenance

The supplied judge rubric forces creativity to be no greater than consistency. That couples two purportedly separate metrics: consistency failures can reduce both consistency and creativity before the four scores are averaged. This is a custom scoring choice, not an independent creativity measure. Eight candidates appear together in each blind record; anonymous labels hide model identity but do not by themselves eliminate order or comparative-context effects. Repeated judging with reshuffled positions and a human spot check would help.

Some evaluation notes prescribe a particular continuation or satisfying ending. These should not penalize a different plausible continuation simply for not following hidden author intent. The `blind.jsonl` includes these notes, but exact API request payloads are absent; what the judge actually received cannot be fully reconstructed from that file alone.

The folder has no executable generation/judging scripts, git revision, checkpoint SHA-256s, tokenizer mapping, finish reasons, or full API request configuration. It does retain useful run settings, prompt-file hash, judge response IDs, and mapping hashes. The current local `experiments/gpt_eval_tinystories.py` is not the exact script used: it accepts plain text prompts, uses a different seed formula and output schema, lacks the attached run's precision option, and truncates repeated 4-grams. Attached outputs retain repeated 4-grams. Do not infer that this local script's truncation was applied to this run.

The current loader builds the tokenizer from train+validation text, instantiates the model from checkpoint configuration, strictly loads weights, and enters eval mode. That is reassuring about the local implementation but cannot certify the collaborator's executable or vocabulary inputs. The QK-identity run also depends on code beyond the basic local loader's supported model types. An actual checkpoint-level cached/uncached and fp32/bf16 comparison remains unperformed because these checkpoints and that execution snapshot were not supplied.

MHA metadata reports step 266,000 although its filename says 250,000. Residual reports 250,000. Parameters are 152.10M versus 124.58M. These same step identities appear in the NLL report. They require disclosure for training-budget/capacity comparisons; neither observation alone explains the changed ranking. Matching paths and counts do not establish byte-identical checkpoints without hashes.

## Why this can coexist with better NLL

The saved validation report gives residual NLL 0.28753 versus MHA 0.28930, paired difference −0.00177242 with 95% CI [−0.00191014, −0.00162754], approximately 0.61% lower NLL. It evaluates 27,631 stories and 22,106,566 next-character targets. The interval concerns these fixed checkpoints and validation stories; validation was also used for checkpoint selection.

NLL conditions on correct preceding characters and evaluates the unsampled predictive distribution. Free generation conditions on the model's own previous choices, uses temperature/top-k, and receives nonlinear whole-story ratings on a different prompt distribution. Small likelihood improvements do not mathematically guarantee improved plot, termination, detail retention, or judge scores. Character NLL can improve through local spelling/syntax predictions while generated narratives deteriorate. This is a possible mechanism, not a diagnosis established here.

The NLL protocol excludes story separators and resets nonoverlapping contexts. It therefore does not directly evaluate successful end-delimiter emission or the same sliding-window free-running trajectories. A narrow NLL interval establishes precision for the measured quantity, not superiority on every other quantity.

## Recommended next experiment

1. Preserve this run as a custom stress test. Describe its MHA advantage honestly, with uncertainty; do not claim a resolved GQA advantage or suppress the result because NLL disagrees.
2. Recover the exact execution snapshot, checkpoint hashes, model configs, tokenizer mapping/data hashes, and judge request configuration. Compare checkpoint-level cached and uncached logits on identical prefixes, including the transition through 512 characters; inspect bf16 versus fp32 discrepancies before sampling.
3. Build a separate, fixed dataset-continuation evaluation from held-out TinyStories prefixes, with predeclared sampling and prefix-cut rules. Include reference continuations as judge calibration. Keep custom reasoning/retention/older-language prompts in a separately reported stress suite. If using existing validation data, disclose its checkpoint-selection role rather than calling it an independent test set.
4. Keep sampling settings common across models; predeclare a small decoding sensitivity grid rather than tuning only GT-MHA until it wins. Record generated token count including the delimiter, finish reason, stop position, sliding-window activation, and untouched raw outputs. Choose a generous cap using reference continuation-length quantiles; report cap rate rather than silently discarding capped stories.
5. Score independent dimensions with a frozen, age-appropriate rubric, randomize candidate order, repeat a subset of judgments, and inspect human-rated examples. Report paired prompt-cluster intervals and multiple training seeds if claiming architectural superiority.

## Reproduction

Run `python3 results/tinystories_generation_audit_20260919/audit.py` from the repository root. It reads the supplied artifacts without modifying them and writes `audit_results.json`. The JSON includes all cap-hit identifiers/endings, per-model statistics, and paired intervals. No generation was rerun and no claim of a verified GT-specific inference bug is made.
