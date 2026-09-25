# BERT MHA head diversity: completed run

- DeltaAI job: `3207532` on `anayak2`; `COMPLETED` with exit code `0:0` in 11 minutes 12 seconds.
- Model: BERT-base MHA, pretraining seed 42, checkpoint 100,000, SHA-256 `33bbc110d6525714ae9fb80469c6dfc0b551ffdd2834d5cc846d01c8e1162ef5`.
- Inputs: the same 1,000 frozen, deterministically masked WikiText-103 validation sequences and FP32 inference protocol used in the existing GT-MHA BERT diversity run. The validation file SHA-256 matches (`8e136ab7130c5fca602eb406ce2557967a974d901b53c4f2992f87b873e0fe5b`).
- Attention metric: normalized pairwise Jensen-Shannon divergence, averaged over unordered head pairs, examples, and layers.

## Descriptive result

| Measure | MHA seed 42 | Existing GT-MHA seed 43 |
| --- | ---: | ---: |
| Overall attention JSD | 0.4372 | 0.3722 |
| GT-MHA attention JSD across bases | — | 0.4549 |
| GT-MHA attention JSD within bases | — | approximately 0 |
| MHA nearest-neighbor attention JSD | 0.2478 | — |

MHA heads can have different attention distributions under this protocol. GT-MHA retains diversity across learned bases, while its BERT in-base attention distributions are essentially identical. Attention-map similarity is only a descriptive statistic: diversity does not establish functional necessity, and similarity does not establish dispensability.

## Single-head intervention

On 128 of the same frozen inputs (2,348 masked target tokens), zeroing one MHA head at a time before the attention output projection increased masked-token NLL for 125 of 144 heads. The median change was `+0.00198` nats per masked token. One head, layer 6 head 7, had a much larger change of `+0.30020`; this outlier should be confirmed on more examples before being featured in a paper claim. The baseline NLL was `1.91564`.

Single-head ablation measures the checkpoint's immediate sensitivity to an uncompensated head removal. It does not test whether a model trained with fewer independent projections can recover the same performance. Thus the positive ablation deltas do not show that MHA's head-specific parameters are irreducible. There is no matched GT-MHA ablation in these outputs. The MHA and GT-MHA diversity runs also use different pretraining seeds (42 and 43), so their numeric difference is descriptive rather than a paired seed estimate.

## Implication of the QK-identity comparison

The reported C=4 QK-identity aggregate score of 70.2 versus 70.3 for MHA is the more direct capacity result. With the QK transformations fixed to identity, heads within each base share an attention map, while the value-side paths remain head-specific. Near parity therefore supports the claim that *independent QK scoring maps are not necessary for this level of BERT transfer performance under the tested training setup*. It does not prove that whole MHA heads are individually removable, nor that every task or seed has the same tolerance. The diversity and single-head-ablation measurements can coexist with this result: a trained MHA may distribute useful information across many diverse heads, even though a constrained model can learn a similarly effective representation from fewer scoring maps.

For the paper, use the QK-identity ablation to motivate economical sharing of scoring structure. Present attention JSD as a characterization of what each architecture learns, not as a test of redundancy. The close C=4 QK-identity result also limits any claim that learned QK transformations are essential at this BERT operating point; their value must be argued through other regimes or advantages, not inferred from this comparison.

## Files

- `summary.json`: protocol, layer summaries, and maximum attention-context reconstruction errors (all below `1.6e-5`).
- `mean_pairwise_matrices.npz`: 12 layer-by-head matrices for attention JSD, raw output distance, and projected contribution distance.
- `per_example_layer.jsonl`: 12,000 per-example/per-layer rows.
- `single_head_ablation.json`: the baseline and 144 head interventions.

All four files were copied from DeltaAI and verified against the remote SHA-256 hashes.
