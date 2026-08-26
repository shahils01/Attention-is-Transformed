# Controlled ImageNet-1K evaluation

This experiment asks whether GT-MHA transfers from language to vision while
holding the DeiT-B/16 macro-architecture and training recipe fixed. It uses the
non-distilled 12-layer DeiT-B classifier (86.57M parameters in this
implementation), 224 x 224 images, 16 x 16 patches, and training from scratch.

## Primary comparisons

| Attention | Role | Total params | Attention params |
|---|---|---:|---:|
| MHA | full reference | 86.57M | 28.35M |
| reduced-QK MHA (`D_k=384`) | parameter-reduction control | 79.48M | 21.26M |
| GQA (`4` KV heads) | widely used sharing baseline | 77.12M | 18.90M |
| Collaborative MHA (`D_k=384`) | closest prior method | 79.54M | 21.32M |
| GT-MHA residual (`C=4`, `P=8`) | proposed method | 73.18M | 14.96M |

All non-attention parameters are identical (58.22M). Reduced-QK MHA follows
the controlled comparison in Collaborative MHA: only query and key dimensions
are reduced; values and the output projection remain full-width.

Use `C=4` as the architecture-consistent primary GT-MHA setting. Also report a
predeclared `C=2` compression point: 69.64M total and 11.42M attention
parameters, reductions of 19.56% overall and 59.72% within attention versus
MHA. Shared-identity attention is an internal ablation, not a primary baseline.

SHViT and EfficientViT are useful related-work references, but they modify the
vision macro-architecture, convolutional blocks, or token mixer. Their published
numbers should not be presented as controlled attention-layer comparisons.

## Training protocol

The runner follows the standard DeiT-from-scratch recipe: 300 epochs, AdamW,
cosine decay with 5 warmup epochs, weight decay 0.05, RandAugment, Mixup 0.8,
CutMix 1.0, label smoothing 0.1, random erasing 0.25, stochastic depth 0.1, and
EMA evaluation. The base learning rate is `5e-4` at a global batch of 512 and
scales linearly with global batch size. RepeatAugment is disabled for all
reported models because timm's RepeatAugment sampler requires an indexable
dataset and is unsupported for iterable WebDataset shards. This fixed protocol
difference must be disclosed when comparing against published DeiT numbers.

Run at least three seeds for every model and report mean plus standard
deviation of ImageNet validation top-1 and top-5. Select the EMA checkpoint by
validation top-1 using the same rule for every model. Also report exact
parameters, peak memory, and images/second. Do not tune GT-MHA on validation
more extensively than the baselines.

## DeltaAI workflow

The authorized Kaggle archive remains on HDD. Conversion reads it directly and
writes only train/validation JPEGs and labels as resumable WebDataset shards on
NVMe:

```bash
sbatch deltaai/setup_vision_env_sshaik4.slurm
sbatch deltaai/prepare_imagenet_wds_sshaik4.slurm
```

On `sshaik4`, these launchers default to the isolated
`/u/sshaik4/Attention-is-Transformed-vision` worktree so the existing `bert`
checkout and its untracked files remain untouched. Vision dependencies are
installed in a separate `lgma-vision` environment that inherits DeltaAI's
CUDA-optimized PyTorch build; the setup job deliberately installs the small
vision packages with `--no-deps` to prevent replacing that Torch build.

After both jobs pass, first run one short smoke allocation by overriding
`EPOCHS=1` and then submit the controlled matrix:

```bash
sbatch --export=ALL,ATTENTION_TYPE=gt_mha_residual,EPOCHS=1,WANDB_MODE=offline \
  deltaai/train_deit_imagenet_gh200x4.slurm

bash deltaai/submit_deit_imagenet_matrix.sh
```

Jobs checkpoint after each epoch and automatically resume from
`checkpoint_last.pt` when resubmitted. Audit exact model sizes with:

```bash
python experiments/audit_vision_models.py --num-base-heads 4
python experiments/audit_vision_models.py --num-base-heads 2
```

## References

- Touvron et al., *Training data-efficient image transformers & distillation
  through attention* (DeiT), ICML 2021.
- Cordonnier et al., *Multi-Head Attention: Collaborate Instead of Concatenate*,
  2020.
- Yun and Ro, *SHViT: Single-Head Vision Transformer with Memory Efficient
  Macro Design*, CVPR 2024.
- Liu et al., *EfficientViT: Memory Efficient Vision Transformer with Cascaded
  Group Attention*, CVPR 2023.
