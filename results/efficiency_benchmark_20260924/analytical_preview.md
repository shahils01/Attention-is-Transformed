# Analytical TinyStories efficiency preview

These values use the published 12-layer TinyStories configurations at batch size 1, sequence length 512, and BF16. FLOPs count dominant multiply-add operations in one attention layer: Q/K/V and output projections, query--key and attention--value products, and dense GT-MHA transformation work. Softmax, masking, normalization, elementwise operations, and vocabulary/MLP computation are excluded. The GPU jobs independently record operator-profiled FLOPs as a secondary diagnostic.

| Method | Total parameters | Attention FLOPs/layer | KV bytes/token/layer | KV cache at 512 tokens, 12 layers |
|---|---:|---:|---:|---:|
| MHA | 152.10M | 5.369G | 4,096 | 24.00 MiB |
| GQA | 133.23M | 3.758G | 1,024 | 6.00 MiB |
| MQA | 128.51M | 3.355G | 256 | 1.50 MiB |
| Collaborative MHA | 128.52M | 3.355G | 2,176 | 12.75 MiB |
| GT-MHA | 124.58M | 3.089G | 1,024 | 6.00 MiB |

The cache values are architecture-level quantities and do not depend on GPU allocation. In particular, GT-MHA stores four base keys and four base values, giving the same persistent cache size as four-head GQA and a 75% reduction relative to 16-head MHA. The FLOP values are preliminary until checked against the staged benchmark output and should not yet be copied into the manuscript.
