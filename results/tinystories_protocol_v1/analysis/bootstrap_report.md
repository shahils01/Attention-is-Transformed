# TinyStories generation: paired prompt bootstrap

Paired prompt-cluster percentile bootstrap; five generations retained per sampled prompt; identical prompt draws across all methods.
20,000 resamples; seed 20260920; 200 prompts × five generations per model.

Pointwise 95% intervals, no multiplicity correction. Prompt-sampling uncertainty conditional on fixed checkpoints, sampled completions and recorded judge outputs. Not training-seed variability, judge repeatability, or a formal equivalence test.

| Model | Grammar | Consistency | Creativity | Plot | Average [95% CI] |
|---|---:|---:|---:|---:|---:|
| MHA | 8.743 | 8.568 | 6.950 | 8.000 | 8.065 [7.972, 8.157] |
| GT-MHA residual | 8.707 | 8.509 | 6.981 | 7.943 | 8.035 [7.947, 8.121] |
| GT-MHA quad | 8.651 | 8.498 | 6.957 | 7.952 | 8.014 [7.922, 8.102] |
| GQA | 8.663 | 8.434 | 6.959 | 7.924 | 7.995 [7.902, 8.087] |
| GT-MHA exact | 8.664 | 8.483 | 6.898 | 7.926 | 7.993 [7.898, 8.083] |
| Collaborative MHA | 8.524 | 8.307 | 6.957 | 7.851 | 7.910 [7.817, 8.000] |
| GT-MHA QK identity B4G8H16 | 8.488 | 8.303 | 6.934 | 7.800 | 7.881 [7.784, 7.976] |
| MQA | 8.230 | 8.007 | 6.880 | 7.549 | 7.667 [7.565, 7.767] |

## Paired average-score differences

Positive differences favor the first model.

| Comparison | Difference | 95% CI |
|---|---:|---:|
| Collaborative MHA minus MHA | -0.1555 | [-0.2392, -0.0702] |
| GQA minus MHA | -0.0703 | [-0.1538, +0.0123] |
| GT-MHA QK identity B4G8H16 minus MHA | -0.1840 | [-0.2643, -0.1045] |
| GT-MHA exact minus MHA | -0.0725 | [-0.1505, +0.0062] |
| GT-MHA quad minus MHA | -0.0508 | [-0.1267, +0.0243] |
| GT-MHA residual minus MHA | -0.0303 | [-0.1122, +0.0508] |
| MQA minus MHA | -0.3988 | [-0.4860, -0.3127] |
| GT-MHA residual minus Collaborative MHA | +0.1253 | [+0.0448, +0.2030] |
| GT-MHA residual minus GQA | +0.0400 | [-0.0470, +0.1273] |
| GT-MHA residual minus GT-MHA QK identity B4G8H16 | +0.1538 | [+0.0657, +0.2390] |
| GT-MHA residual minus GT-MHA exact | +0.0423 | [-0.0420, +0.1263] |
| GT-MHA residual minus GT-MHA quad | +0.0205 | [-0.0590, +0.1005] |
| GT-MHA residual minus MQA | +0.3685 | [+0.2760, +0.4600] |

Per-criterion intervals and paired differences are in `bootstrap_results.json`. All 8,000 unique records passed balance and score-range checks.
