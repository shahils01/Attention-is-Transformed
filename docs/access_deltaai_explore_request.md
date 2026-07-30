# ACCESS Explore Request Draft — DeltaAI

## Recommended request pathway

- Project type: Explore ACCESS
- ACCESS Credits: 400,000 total, issued in two 200,000-credit increments
- Planned first exchange: up to 200,000 credits for approximately 1,400 NCSA DeltaAI GPU-hours
- Planned second exchange: up to 200,000 additional credits after the required progress report
- Project duration: 12 months

The current ACCESS exchange calculator converts 1,000 ACCESS Credits to approximately
7 NCSA DeltaAI GPU-hours. Exchange rates may change, so the conversion should be
rechecked immediately before exchanging credits.

## Project title

Evaluating Lie-Generated Metric Attention for Efficient Language and Vision-Language-Action Models

## Field of science

Artificial Intelligence and Intelligent Systems

## Keywords

attention mechanisms; efficient transformers; language models; vision-language-action
models; robot learning; distributed training; model compression; KV-cache efficiency

## Planned-work summary

This project will evaluate Lie-Generated Metric Attention (LGMA), a new transformer
attention architecture designed to reduce attention parameter and key-value cache costs
while retaining expressive, head-specific transformations. Conventional multi-head
attention learns independent query, key, and value projections for each attention head.
LGMA instead constructs head-specific attention metrics from a shared set of learned Lie
algebra generators. This structure may provide a more parameter-efficient and
memory-efficient alternative to standard attention.

The work will test LGMA in two complementary experimental settings. First, we will train
decoder-only language models and compare LGMA with conventional multi-head attention on
TinyStories and, if appropriate data access and preprocessing are available, a WikiHow
text corpus. These experiments will measure validation loss, perplexity, convergence,
training throughput, peak GPU memory, parameter efficiency, estimated KV-cache
requirements, and robustness across controlled configurations and random seeds. Second,
we will compare LGMA and conventional multi-head attention in vision-language-action
models trained on the LIBERO robot-learning benchmark. These experiments will measure
task success, sample efficiency, training throughput, peak GPU memory, convergence as a
function of samples and GPU-hours, and multi-GPU scaling.

NCSA DeltaAI is requested because the experiments require H100-class GPU memory and
mixed-precision throughput for repeated transformer training, vision encoding, and
multi-GPU vision-language-action experiments. Most runs will use one or two GH200/H100
GPUs. Selected experiments will use four GPUs on a
single DeltaAI node to evaluate distributed-data-parallel scaling. The project code
already supports CUDA, bf16 precision, checkpointing, and PyTorch distributed training.
An initial portion of the allocation will be used to validate the software environment
on DeltaAI's ARM-based NVIDIA Grace CPUs and to establish reproducible Apptainer or
Conda environments.

The project will produce open-source training code, configuration files, evaluation
tools, efficiency accounting, reproducible experiment summaries, and research results
comparing LGMA with standard multi-head attention. Results will be prepared for open
scientific dissemination.

## Preliminary DeltaAI GPU-hour budget

| Activity | GPU-hours |
|---|---:|
| ARM environment, dependency, and dataset validation | 80 |
| Single-, two-, and four-GPU scaling studies | 160 |
| TinyStories and WikiHow language-model baselines and LGMA experiments | 760 |
| VLA baseline and LGMA experiments across configurations and seeds | 700 |
| Evaluation, checkpoint recovery, and contingency | 400 |
| **Planned DeltaAI workload across both credit increments** | **2,100** |

## Software and technical readiness

- Python and PyTorch
- CUDA and bf16 mixed precision
- `torchrun` and PyTorch DistributedDataParallel
- Hugging Face Transformers and Datasets
- Apptainer/NGC or Conda environments
- Weights & Biases or offline experiment tracking
- Checkpointed Slurm workflows
- NumPy, SciPy, pandas, OpenCV, h5py, and Zarr
- MuJoCo/LIBERO evaluation tooling where supported

## Expected outputs

- Open-source LGMA implementation and experiment configurations
- Reproducible MHA-versus-LGMA language-model comparisons on TinyStories and WikiHow
- VLA benchmark results on LIBERO or related robot-learning datasets
- Parameter, KV-cache, memory, throughput, and GPU-hour accounting
- Public technical report, preprint, or peer-reviewed paper

## Supporting grant

To be completed. Only list a grant if it directly supports this work. If no grant
directly supports the work, state that the project is currently unfunded rather than
listing an adjacent award.

## Team

- Principal Investigator: Shahil Shaik
- Institution: Clemson University
- Faculty advisor: To be completed
- Co-PIs or additional users: To be completed

## Graduate-student advisor letter template

The final letter must be signed and placed on Clemson University letterhead.

> ACCESS Allocations:
>
> I am aware of Shahil Shaik's request to utilize ACCESS resources in support of
> research on Lie-Generated Metric Attention for efficient language and
> vision-language-action models. I intend to engage with and guide the computational
> and research activities conducted under this allocation. The proposed work is
> appropriate for NCSA DeltaAI and will support reproducible experiments evaluating
> transformer efficiency, distributed training, and robot-learning applications.
>
> Sincerely,
>
> [Faculty advisor name, title, department, and signature]

## Required materials and unresolved fields

- PI CV or résumé in PDF format
- Signed faculty-advisor letter in PDF format, if the PI is a graduate student
- Department
- Academic status
- Current country of residence
- Citizenship
- Highest completed degree and degree field, if requested
- Faculty advisor name
- Supporting grant information, if directly applicable
- Additional project members and their ACCESS IDs
