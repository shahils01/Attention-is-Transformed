# Independent TinyStories evaluation protocol — proposed v1

Status: designed, not executed or preregistered. No retraining, positional-extension tricks, checkpoint changes, or paper edits. Freeze this protocol, the prompts, checkpoint manifest, and exact judge configuration before examining new test outputs. This is a new evaluation after seeing a collaborator's preliminary results, not a retrospectively independent preregistration.

## Generation update

For the main run, the author requests a 3,000-character-token cap and eight simultaneous checkpoint tasks, each on an A100 40GB. This supersedes the 2,048-token proposed cap below; the completed engineering run is retained unchanged. Continue recording all cap hits. See `inference_benchmark_plan.md` for separate operational timings and controlled paper benchmarks.

## Scope update from the author

Residual GT-MHA is the primary approach. Evaluate all eight published TinyStories checkpoint families in `checkpoint_manifest.json`, including exact, quadratic, and QK identity. Exclude unfinished QKV identity. The published Hugging Face files at the pinned revision are the source of truth, rather than arbitrary cluster training snapshots. Record actual checkpoint metadata; paper wording and matched-budget reconciliation do not block the generation engineering run. Final-versus-best-validation sensitivity remains contingent on availability and is not inferred from the published filename. This update supersedes conflicting model-priority and pre-run budget-gate statements below.

Eight published checkpoints give 8,000 primary generations/ratings, 8,000 optional within-context excerpt ratings, and 7,200 short stress-test generations under the proposed counts.

## Purpose and paper commitments

Assess whether attention sharing preserves predictive performance and open-ended generation quality, and identify failure modes within a 512-character context. A useful result may favor either architecture. Do not assume lower NLL implies better generation or choose prompts to recover a preferred ranking.

Current manuscript: `iclr2026_conference.tex`, experiment overview around lines 429–438, TinyStories section around 473–479, and GPT-Eval appendix at 1097. It promises:

| Commitment | Implementation |
|---|---|
| Controlled single training seed | Existing checkpoints; report seed and do not imply training-seed uncertainty |
| Matched training-token budget | Verify cumulative training tokens, not checkpoint filenames or step labels alone |
| Same prompts, decoding, lengths, seeds | Frozen common prompt IDs, generation budgets, deterministic per-sample seeds; allow natural stopping |
| Anonymous, randomized grading | One completion per judge request; opaque IDs; shuffled request order; private mapping |
| Separate grammar, creativity, consistency, plot scores | Four independent 1–10 ratings, no creativity<=consistency constraint |
| Prompt bootstrap CIs | Average samples within prompts; paired common prompt resamples |
| Final vs best-validation sensitivity | Both selection rules fixed without using generation results; deduplicate identical checkpoint hashes |
| Release evaluation materials | Prompt corpus, rubric, manifests, raw generations, finish metadata, raw judge responses, aggregation code |

The main table currently names **quadratic GT-MHA**, whereas the saved full-validation CI result is for **residual GT-MHA**. Primary methods: MHA, CoMHA, MQA, GQA (4 KV heads), GT-MHA quadratic C=4, and GT-MHA residual C=4. Quadratic is the manuscript's primary GT variant; residual is a separately named planned comparison. Exact and QK-identity variants can be labeled additional ablations, not silently substituted. The appendix table must eventually include all main methods and distinguish GT variants.

The original [TinyStories paper, section 3](https://arxiv.org/html/2305.07759v2) uses manually composed prefixes and ten completions at temperature 1. This protocol retains its grading framework but uses dataset prefixes and five completions; describe it as **TinyStories-style GPT-Eval with specified adaptations**, not an exact replication.

## Checkpoint gate

Before a scientific run, record each checkpoint's SHA-256, model config, training seed, cumulative trained tokens, optimizer step, dataset/tokenizer hashes, selection rule, and source revision. Preserve the exact vocabulary order. Use strict state loading and eval mode. Record all parameter counts and context lengths.

“Final” means the checkpoint at the common declared training-token endpoint. “Best validation” means minimum logged validation NLL among eligible saved checkpoints at or before that endpoint under a common selection protocol, with ties resolved by earliest step. It cannot be reconstructed honestly from one final checkpoint alone. If logs or checkpoints are missing, mark that sensitivity unavailable and revise the promise explicitly. Do not substitute best GPT-Eval results.

Known issue: MHA metadata says step 266,000 despite a 250,000 filename. Determine actual training-token exposure from logs/configs. If a matched existing snapshot exists, use it; otherwise report available-checkpoint results and remove the matched-budget assertion. This does not require retraining.

On a disjoint 20-prefix engineering set, verify tokenizer round trips, cached versus uncached next-token logits, the sliding-window transition, exact delimiter matching, and cap accounting. Establish reasonable numerical tolerances and log errors for each architecture; do not require identical sampled trajectories under tiny floating-point differences. Run a small fp32/bf16 logit sensitivity check, then freeze bf16 for the main common execution setup. This is implementation verification, not selection by quality score.

## Track A — primary open-ended generation

Question: how good are stories sampled from the existing checkpoints on TinyStories-distribution prefixes?

- Source: exact TinyStoriesV2-GPT4 validation text used by the project, version/hash recorded. Split on the literal `<|endoftext|>`; retain original text offsets. Exclude only empty/malformed records, exact training duplicates (if the train file is available), and records lacking a whitespace boundary between character offsets 120 and 160 or having fewer than 256 characters. Count and publish every exclusion. If overlap checking is unavailable, disclose that rather than claiming disjointness.
- Select 200 distinct stories with fixed selection seed 20260919: divide eligible stories into four equal-count bins by character length (ties by story ID), randomly sample 50 per bin. No quality-based manual editing or excluding “hard” stories. Keep this eligibility-conditioned population explicit. If any bin has fewer than 50 stories, stop and amend the design before generation.
- Prefix: original story start through the first whitespace at or after character 120, no later than 160, excluding that whitespace from the prefix. No chat wrapper, task instructions, summaries, or hidden constraints. The next continuation character may be whitespace. Prefixes may end mid-sentence but not mid-word. Character counting uses Python Unicode code points, matching CharTokenizer.
- Reserve the engineering set before test sampling; keep it disjoint. Do not reuse collaborator prompts. Validation was used for model selection, so label this **validation-prefix generation**, not an independent held-out test set.
- Five independent sampled continuations per prefix/checkpoint. Temperature 1.0, top-k 20, no top-p, beam search, repetition penalties, minimum length, or post-hoc repetition trimming. Maximum 2,048 generated character tokens including the stop delimiter. This cap is an operational ceiling, not a claim about an optimal story length.
- Stop only at the first full literal `<|endoftext|>` or the cap. Preserve raw text and IDs including delimiter; remove only the full delimiter for judge display. Keep partial delimiters visible and log them. No punctuation-based stopping, forced ending, or resampling of unattractive outputs.
- Use current 512-character sliding-window behavior for all models: valid cache while the window fits, recompute the full recent window after it shifts. Do not naively crop the cache or carry unverified older state. Generation seed is a deterministic hash of suite ID, prompt ID, sample index, and a fixed master seed; exclude model/checkpoint identity so seeds match across methods. Record exact integer seeds. Matching seeds does not guarantee identical semantic choices across models.
- Save raw token count, decoded count, finish reason (`stop_sequence` or `max_new_tokens`), delimiter position, whether/when context shifted, runtime/configuration, and all failures. Retry only infrastructure failures under the same ID/seed and keep the failure log. Do not replace difficult prompts.
- Judge grammar, creativity, consistency, and plot coherence separately. Assess story closure separately as `closed`, `open`, or `unclear`; report stop-token success separately from narrative closure. Early stopping alone does not establish a complete plot. Primary quality ratings concern the actual delivered story; a ceiling-truncated story remains in the primary analysis.

The headline remains full-generation scores even if GT-MHA loses. Do not replace it with an easier diagnostic after seeing results.

## Track B — within-context continuation diagnostic

Question: is a quality gap already visible before any prompt context is removed?

Reuse the same 200 prompts and raw Track A trajectories. Display at most the first 320 generated characters, ending sooner if the model stopped. With prefixes <=160 characters this remains within 480 characters, so all shown content was generated without dropping the prefix. Strip a full stop delimiter if present; retain raw data. Assert from generation metadata that no context shift contributed to any displayed content.

A dedicated excerpt rubric scores grammar, creativity, consistency, and local event coherence. An evaluator-imposed cut, including a mid-word or mid-sentence cut, is explicitly marked and must not reduce grammar or closure scores. Natural early termination is marked separately. No demand for a full plot resolution in a deliberately cut excerpt. Never directly pool these ratings with Track A plot scores or claim that this replaces promised open-ended evaluation.

Judge all excerpts rather than choosing successful ones. This adds grading calls but no generation. Raw versus excerpt differences are diagnostics, not a causal estimate of context loss, because text length and grading target also differ.

## Track C — controlled context and reasoning stress test

Question: what information can each model use while the necessary evidence still fits inside its trained context?

Build 60 underlying story scenarios: 20 name/attribute bindings, 20 ownership/location tracking, 20 simple causal/negation continuations. Each scenario has three natural narrative variants with neutral distractors producing prefix lengths 128–160, 288–320, and 416–448 characters. Preserve the same facts and semantic completion cue across its variants. Use names, toys, food, family, animals, and simple everyday events; no older-reader vocabulary targets, arbitrary number strings, or chat-style commands.

Example semantic structure, before constructing length-controlled variants:

`Lily put her red ball in the box. Tom put his blue ball in the bag. ... Lily wanted her own ball. She looked in the`

The completion must be a natural story continuation. For each scenario define accepted semantic answers, contradictions, and ambiguity rules in advance. Do not assume exact next-word matching is enough; “little box” and “box” can agree. Two reviewers check naturalness, clarity, fact retention, and plausible alternate answers without seeing model outputs. Revise or reject ambiguous scenarios before freezing. Surface changes for counterbalancing may be included, but remain nested in the same scenario family for statistics.

Generate five continuations per variant, at the same temperature/top-k as Track A, but only 48 new character tokens. Even the longest prefix plus completion is <=496. No evidence is dropped; difficulty comes from distance/distractors within the available window, not testing inaccessible facts. Stop on the same delimiter.

Score the first completed answer phrase/clause for correct, incorrect, or unresolved. Report three-way rates; primary success is correct/all attempts, so unresolved answers do not disappear from the denominator. Use deterministic scoring only for unambiguous cases, with blinded human adjudication for remaining cases and all rule exceptions. Report task success and contradictions separately from literary quality. Do not use these prompts to rank full-story creativity or plot.

Bootstrap scenario families, keeping all three lengths and five samples together. Report per-task, per-length results and paired degradation from short to long. Length, filler, and interference co-vary: this is an operational stress curve, not a pure causal measurement of distance.

## Judge protocol and quality control

See `judge_rubric.md`. Use a single exact judge snapshot/provider/config for the full run, selected and recorded before test generation. Model availability and cost must be checked at execution; a specific judge has not been selected or called by this design. Include an initial engineering-only rubric check. No selecting the judge because it favors GT-MHA.

Submit one anonymous candidate per request, not eight competing outputs in the same context. Shuffle request order with a saved seed; remove model name, parameter count, checkpoint selector, generation timing, NLL, and architecture. The judge receives story prefix, continuation, evaluation mode, and necessary cut metadata, not hidden story-writing objectives or an expected reference ending. The model being evaluated receives only the narrative prefix.

For 20 preselected Track A prompts, also score original reference continuations as blinded calibration examples, never as mandatory answers. Insert fixed corrupted controls made from those references (one grammar corruption and one character-name contradiction), preserving their provenance. Check that the corresponding dimensions respond sensibly; publish failures rather than silently replacing the judge after unblinding. These are calibration controls, not model results.

Repeat 10% of main ratings with fresh request order and IDs; the first response stays the primary score and repeats quantify reliability. On 20 preselected prompts, two humans independently assess sample index 0 for each checkpoint, blinded and in separate shuffled order. Report discrepancies and agreement. If human resources are unavailable, label this planned validation incomplete.

## Aggregation and interpretation

Four primary endpoints: mean grammar, creativity, consistency, and plot coherence in Track A. Average five samples within each prompt, then prompts equally. Report 95% percentile bootstrap intervals using 10,000 shared prompt resamples, seed 731. For GT quadratic/residual versus MHA/GQA, use paired differences with the same resampled prompts. Preserve all samples, repeats, and checkpoints within their prompt cluster. Track C instead uses 60 scenario clusters.

Dimension-wise intervals are pointwise. To make confirmatory superiority claims across the 16 planned variant/baseline/dimension comparisons, also report conservative 99.6875% Bonferroni-adjusted bootstrap intervals using 100,000 resamples; otherwise describe comparisons as exploratory and avoid omnibus superiority language. An optional four-score mean is secondary and declared in advance. Failure to resolve a difference does not demonstrate equivalence or non-inferiority; no such margin is specified here.

Final-vs-best-validation sensitivity: evaluate Track A on the same 200 prompts and five seeds for both selectors. Reuse identical hashes rather than generating twice. Track B can reuse either selector's Track A outputs. Track C is final-checkpoint only unless an extension is frozen before outputs. Report both checkpoint selectors regardless of which performs better.

Report length distributions, delimiter-stop fraction, cap-hit fraction, narrative closure, repetition diagnostics, context-shift fraction, and invalid judging responses alongside quality. Length/context subgroups and excluding capped outputs are exploratory: they are model-outcome-dependent and can create selection bias. Do not use them to replace primary results. Full-story generation and excerpt scores are different estimands.

For transparency, report all frozen test prompts and model outputs. Choose printed examples using fixed prompt IDs and sample index 0 before unblinding, with any additional failure examples clearly labeled. The previous collaborator evaluation remains preliminary context; do not claim it never occurred or that the new protocol was chosen before observing it.

## Workload and release

Six primary checkpoints, one final selector each:

- Track A: 200 x 5 x 6 = 6,000 generations and primary judge calls.
- Track B: 6,000 excerpt judge calls, no extra generation.
- Track C: 60 x 3 x 5 x 6 = 5,400 short generations.
- Additional distinct best-validation checkpoints add 1,000 Track A generations/ratings each, plus 1,000 excerpt ratings if Track B sensitivity is run. Repeat ratings and calibration are extra. Request-token cost depends on judge choice; no monetary estimate is assumed.

Budget reductions should be fixed before results. Track A plus checkpoint sensitivity fulfills the central manuscript promise; Tracks B/C explain behavior. If needed, defer those diagnostics rather than omit the promised main quality evaluation or silently drop baselines.

Release layout: `protocol.md`, `judge_rubric.md`, dataset/checkpoint/code manifests, `prompts.jsonl`, `stress_scenarios.jsonl`, private mapping (released after judging), `generations.jsonl`, raw judge requests/responses, human adjudications, bootstrap results, and a report. Hash the frozen protocol, rubrics, and prompt files. Every record carries run/prompt/sample/checkpoint IDs. Full generation logs distinguish normal stop, cap, and infrastructure errors.

## Remaining inputs before execution

This design is complete enough to implement, but it is not a completed benchmark. Needed: checkpoint inventory and hashes, training-token/log provenance and best-validation snapshots, exact dataset/tokenizer files, constructed/reviewed stress prompts, exact judge snapshot and execution budget, and frozen manifests. Do not silently invent any of these. No retraining is required.
