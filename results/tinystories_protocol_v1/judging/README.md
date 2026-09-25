# TinyStories judging client

Remote root: `/scratch/shahils/tinystories_eval_20260919`.

The RCD credential is installed only at `/home/shahils/.config/tinystories-eval/rcd-token`, mode 0600. It is read at execution time and is never included in request artifacts or Slurm scripts. The authenticated models endpoint confirmed `gpt-4.1-2025-04-14` is available.

Use Python 3.7+ (the existing Palmetto `llava_video` environment). No additional packages or GPU are required. `judge.py prepare` snapshots the rubric, schema, input hashes, runner hash, and randomly ordered anonymous payloads. The private candidate-to-checkpoint mapping is never sent to the API.

`calibrate.slurm` runs the 160 development stories only, sequentially with automatic resume, under a conservative $5 reservation limit. Actual estimated cost is calculated separately from returned usage; this is not an RCD billing receipt. Reservations use UTF-8 bytes plus overhead as a conservative token bound, ignore input caching discounts, reserve maximum output tokens, and retain reservations for failed or uncertain attempts. Retries count toward this limit. The client is intentionally conservative and can stop before the actual cost reaches $5.

Every request and raw response is stored separately. Strict JSON schema and local semantic validation reject invalid scores, refusals, and truncated judge output. One schema-only retry is permitted; repeated invalid output becomes explicitly missing. Retryable HTTP failures have bounded retries with Retry-After handling. Unknown network outcomes stop for inspection to avoid silent duplicate billing. A process lock prevents simultaneous runners in the same directory. Valid completed results are not overwritten.

To resume or inspect from the remote root after activating the environment:

```sh
python judging/judge.py run judging/development_v1 --budget-usd 5 --limit 160
python judging/judge.py summary judging/development_v1
```

Before main judging: review anonymous development judgments, inspect score coverage and explanations, run a fixed repeatability subset, and freeze the protocol. No architecture ranking is a calibration criterion. A main run requires `frozen.json` containing the matching `config_sha256`; this is an experiment-integrity guard, not a claim that human review has occurred. Prepare main input from the completed generation files, retain cap-hit stories, and set an explicit full-run cost limit. Main and excerpt modes must use separate directories. All main-run candidates should be shuffled together once; do not rebuild a running manifest to add late records.

Offline checks passed for 160 unique anonymous payloads, rejecting boolean/out-of-range scores, inconsistent closure, and truncated output. A live smoke check and the development run verify RCD's actual schema support. The optional excerpt mode and full main run have not been launched.

Reference for the response format: https://developers.openai.com/api/docs/guides/structured-outputs

## Main submission

Job 16090585 was submitted with afterok dependencies on development job 16090566 and generation array 16086025. Development completed with 160 valid results, no failed attempts, and estimated usage $0.436558. The main job waits for all eight completed checkpoints and validates 8,000 paired candidates before preparing and freezing the input/configuration manifest. The rubric and judge parameters remain identical to development.

The primary run covers full stories only. The byte-based worst-case reservation limit is $200, intentionally much larger than the roughly $21.83 extrapolation from development usage; this is not a purchase or expected cost. Generation length differences, retries, and actual RCD accounting can change usage.

A blinded assistant spot-check of four development stories found a missed explicit fast/not-fast contradiction. Schema validity is not judgment accuracy. No human agreement or repeatability study has been completed; scores remain provisional pending a reliability audit. No checkpoint rankings informed the freeze.
