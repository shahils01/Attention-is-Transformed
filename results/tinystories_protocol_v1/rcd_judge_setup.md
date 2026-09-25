# RCD judge setup recommendation

Inspected 2026-09-19. Recommendation only; no API requests have been made and the judge is not frozen.

## Primary judge

Use `gpt-4.1-2025-04-14` through `https://llm.rcd.clemson.edu/v1`, with the existing RLS key provided as `RCD_LLM_API_KEY`. Run the client on Palmetto (Clemson network); no GPU is needed for an API client. Use `/chat/completions`, temperature 0, strict JSON schema matching `judge_rubric.md`, and an initial 512 judge-output-token cap. This is separate from the generator's 3000-character-token limit. Verify schema support and truncation behavior in a development pilot before freezing. Temperature zero does not guarantee repeatability.

Rationale: a dated snapshot, structured output support, moderate cost, and continuity with GPT-based judging. This is not evidence that GPT-4.1 is the most accurate available judge. The live RCD catalog also lists GPT-5.6-sol and hosted GLM-5.3. GLM-5.3 could provide a secondary, separately reported agreement check; do not substitute a moving alias such as `tiger` or select judges by the resulting model ranking.

## Credentials and accounting

The portal lists active `Palmetto_IM_Key` under `Imitation_Learning_Shahil`. Existing key values cannot be recovered from this listing; do not rotate a working key. Use its existing secure location if supplied by the user, or create a dedicated key after authorization. Never put tokens in tracked files, command arguments, requests logs, or chat.

At inspection the project had 8.51 credits remaining; the shared pool showed 984.09 available. Pool availability is not a reserved allocation. Shared usage can be rate limited. The full evaluation likely exceeds the project's remaining allocation. Verify credit accounting with pilot usage before launching the full run.

The catalog quotes GPT-4.1 at $2 per million input tokens and $8 per million output tokens (cached input $0.50). Illustratively, 8,000 calls averaging 1,200 input and 250 output tokens cost $35.20 without caching; 160 development calls at the same averages cost $0.704. These are assumptions, not measured estimates or a spending authorization. Excerpt judging adds a separate workload; retries and reliability repeats add cost. RCD's own Batch API does not automatically receive OpenAI batch discounts. Do not assume GPT-4.1 supports Flex.

## Execution sequence

1. Verify authenticated model availability and one development request using the exact snapshot. Do not silently fall back to another model.
2. Use the 160 development generations for blinded calibration; check parse validity, refusals, output caps, score range, concrete evidence, and repeatability on a fixed subset. Include human review across the quality range. Judge selection/rubric changes must not depend on favorable architecture rankings.
3. Measure actual input/output usage and confirm an explicit run budget. Freeze prompt, schema, model ID, parameters, preprocessing, and missing-response handling with hashes before main judging.
4. Randomize anonymous candidates across all available checkpoint outputs. Start with low concurrency (e.g. four), honor Retry-After and bounded backoff, log every attempt and usage, and resume using stable candidate IDs. Never log Authorization headers. Store raw responses, returned model IDs and system fingerprints when available.
5. Judge each full continuation independently with its prefix; retain cap-hit stories. Keep optional first-320-character excerpt grading separate. Final paired comparisons require matched prompt/sample coverage across all models; bootstrap by prompt, retaining samples within each cluster.

## Sources

- RCD API routing, network prerequisites, data path, and accounting: https://docs.rcd.clemson.edu/llm/usage/openai_api/
- Live account credits: https://llm.rcd.clemson.edu/ui/openai
- Live catalog: https://llm.rcd.clemson.edu/ui/models?external=true
- API key metadata: https://llm.rcd.clemson.edu/ui/api-keys
- OpenAI snapshot, structured output support, and pricing: https://developers.openai.com/api/docs/models/gpt-4.1

The RCD gateway forwards GPT request content to OpenAI. Its documentation says it records accounting metadata rather than prompts or outputs; the local evaluation client must preserve the experimental record itself.
