# GLUE validation across fine-tuning seeds

Scores are percentages. Mean ± sample standard deviation at the learning rate used for the official submission. Seeds 42–46 for each task except WNLI (seed 42 only). MRPC/QQP use mean accuracy and F1; STS-B uses mean Pearson and Spearman; MNLI uses the recorded matched accuracy. These are validation scores, not the official test scores.

| Method | COLA | SST2 | MRPC | STSB | QQP | MNLI | QNLI | RTE | WNLI |
|---|---|---|---|---|---|---|---|---|---|
| MHA | 32.09 ± 2.35 | 87.50 ± 0.23 | 82.40 ± 1.00 | 84.75 ± 0.52 | 87.66 ± 0.11 | 77.23 ± 0.23 | 86.12 ± 0.27 | 55.45 ± 1.29 | 56.34 |
| GQA | 32.01 ± 1.33 | 87.55 ± 0.22 | 83.66 ± 1.02 | 78.18 ± 3.20 | 87.23 ± 0.15 | 76.98 ± 0.18 | 85.48 ± 0.43 | 57.04 ± 2.77 | 56.34 |
| Collaborative MHA | 35.68 ± 2.19 | 88.10 ± 0.27 | 78.27 ± 0.75 | 79.93 ± 0.65 | 86.53 ± 0.04 | 75.22 ± 0.28 | 84.13 ± 1.12 | 54.58 ± 1.18 | 56.34 |
| GT-MHA | 33.73 ± 1.15 | 87.59 ± 0.73 | 79.84 ± 0.36 | 83.23 ± 0.57 | 87.61 ± 0.09 | 77.61 ± 0.12 | 85.76 ± 0.40 | 56.53 ± 1.72 | 56.34 |

## Interpretation and provenance

The learning rate is selected using validation means over seeds 42–44; seeds 45–46 are additional runs at that rate. Consequently, these validation summaries reflect model selection and are not independent estimates. They characterize fine-tuning variability conditional on the selected pretrained encoder, not pretraining variability. No nine-task aggregate uncertainty is calculated.

The official submission manifests and original metrics are retained alongside this report. summary.json records every included run path, metric, seed and learning rate. Historical runs and identity ablations are excluded from this four-method table.

## Audit findings

All 36 cells have the expected run counts; selected-checkpoint scores and best-seed scores agree with the manifests; all 32 three-learning-rate grids reproduce the selected learning rate.

## Collection scope

Retrieved on 2026-09-15 through the existing anayak2, arai3 and sshaik4 SSH control sockets. The direct lgma_runs folders yielded 201, 370 and 0 evaluation records respectively. The 164 runs required by the submission manifests were all found in the retrieved records; no additional sshaik4 files were needed for this main comparison. STS-B records stored on anayak2 were matched by run name to the packaged submission paths on arai3.

GT-MHA has a higher validation mean than MHA on CoLA, SST-2, MNLI and RTE; lower means on MRPC, STS-B, QQP and QNLI; WNLI is tied. These are descriptive comparisons, not significance tests.

This audit verifies result completeness, metric arithmetic and selection consistency. It does not independently establish identical training configurations across all runs. The LaTeX fragment has not been compiled against the current Overleaf document.
