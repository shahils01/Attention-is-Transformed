# TinyStories bootstrap evaluation — job 3158800

Job completed successfully in 47 minutes. Evaluation uses 27,631 validation stories and 22,106,566 next-character targets. Story-isolated, nonoverlapping 512-character contexts; separators excluded. 2,000 story-cluster percentile bootstrap samples with common story resamples across models.

| Model | Checkpoint step (metadata) | NLL [95% CI] | PPL [95% CI] |
|---|---:|---:|---:|
| MHA | 266000 | 0.28930 [0.28826, 0.29032] | 1.33549 [1.33411, 1.33685] |
| GQA | 250000 | 0.29046 [0.28942, 0.29146] | 1.33704 [1.33565, 1.33838] |
| MQA | 249800 | 0.29572 [0.29466, 0.29675] | 1.34410 [1.34268, 1.34548] |
| Collaborative MHA | 250000 | 0.28959 [0.28855, 0.29063] | 1.33589 [1.33450, 1.33726] |
| GT-MHA residual | 250000 | 0.28753 [0.28651, 0.28855] | 1.33313 [1.33177, 1.33449] |

GT-MHA minus MHA: NLL difference -0.00177242, paired 95% CI [-0.00191014, -0.00162754]. This describes fixed-checkpoint evaluation uncertainty on validation data also used for selection, not training-seed uncertainty or an independent test-set result.

## Reporting constraints

- MHA checkpoint metadata says step 266000 although its source filename in the checkpoint manifest says 250000. MQA metadata says 249800; the other models say 250000. Resolve checkpoint provenance before making an equal-training-budget claim.
- These means use a different evaluation protocol from the training-log values; keep each mean with its corresponding confidence interval.
- This evaluation measures NLL/perplexity, not LLM-judge generation ratings.
- Evaluation throughput here is not the training throughput requested by Table 3(c).

## Verification

Recomputed all five NLLs from per-story loss sums and token counts; checked common story ordering and counts; reproduced marginal and paired NLL percentile intervals from the saved bootstrap replicates. All checks passed.
