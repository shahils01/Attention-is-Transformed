# C=4 GT-MHA: Table 3(b) GLUE development score

Use **73.99**. Recomputed from the 40 selected-learning-rate evaluation records in `summary.json`: eight tasks, each averaged over fine-tuning seeds 42–46. This matches the averaging convention used for the C=2 entry (73.19).

| Task | Mean ± sample SD | Selected LR |
|---|---:|---:|
| cola | 33.73 ± 1.15 | 2e-5 |
| sst2 | 87.59 ± 0.73 | 2e-5 |
| mrpc | 79.84 ± 0.36 | 2e-5 |
| stsb | 83.23 ± 0.57 | 5e-5 |
| qqp | 87.61 ± 0.09 | 5e-5 |
| mnli | 77.61 ± 0.12 | 5e-5 |
| qnli | 85.76 ± 0.40 | 3e-5 |
| rte | 56.53 ± 1.72 | 5e-5 |

Unrounded eight-task mean: 73.987677173005. All metrics are on the 0–100 scale. CoLA: Matthews correlation; MRPC/QQP: mean accuracy and F1; STS-B: mean Pearson and Spearman; MNLI: matched accuracy; remaining tasks: accuracy. WNLI and MNLI mismatched are excluded.

Learning rates were selected using seeds 42–44; seeds 45–46 were additional runs at those rates. These are five fine-tuning seeds conditional on one pretrained checkpoint, not three-pretraining-seed GLUE results.

Suggested table row (retaining the existing attention count and MLM loss):

```latex
4 & 14.98 & 2.12 & 73.99 \\
```

Suggested caption clarification: “GLUE dev is the unweighted mean over eight tasks, with each task averaged over five fine-tuning seeds from one pretrained checkpoint.”

The dev average of the eight individually selected submission checkpoints is 74.735269; it uses best-seed selection and is not the five-seed statistic above. The official GLUE test score is also a different statistic.
