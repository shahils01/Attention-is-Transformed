# C=2 QK identity GLUE validation

Five fine-tuning seeds per selected learning rate; one pretrained checkpoint (seed 44). Task SDs are sample SDs. Eight-task aggregate excludes WNLI.

| Task | QK identity mean ± SD | GT-MHA mean | Selected LR |
|---|---:|---:|---|
| cola | 33.21 ± 1.44 | 33.23 | 3e-5 |
| sst2 | 87.29 ± 0.75 | 87.48 | 5e-5 |
| mrpc | 82.11 ± 1.25 | 78.93 | 5e-5 |
| stsb | 81.96 ± 1.25 | 80.66 | 5e-5 |
| qqp | 87.22 ± 0.14 | 87.44 | 5e-5 |
| mnli | 75.55 ± 0.40 | 76.83 | 3e-5 |
| qnli | 85.05 ± 0.41 | 85.74 | 2e-5 |
| rte | 58.92 ± 1.15 | 55.23 | 2e-5 |

Eight-task dev mean: **73.914476**. GT-MHA: 73.193850.
GT-MHA minus QK identity: -0.720626.
Confirmation-only mean (seeds 45–46): 73.943170.

Metrics: CoLA Matthews; MRPC/QQP mean F1 and accuracy; STS-B mean Pearson and Spearman; MNLI matched accuracy; other tasks accuracy. Scores scaled to 0–100. Development results, not official test results.
