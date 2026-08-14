# ACCESS Explore Request Draft — Aayush Rai / GT-MHA

## Submission posture

This request should be presented as Aayush Rai's own, coordinated research workstream,
not as a duplicate of an existing allocation. It may share the GT-MHA/LGMA codebase and
faculty supervision with the earlier project, but the proposed experiments, accounting,
and deliverables should be distinct. Runs already funded or completed under another
ACCESS project must not be charged again here. Shahil Shaik should be listed under
**Other Collaborators** (and added as project personnel only if he will actually use this
allocation).

## Recommended live-form entries

- **Project type:** Explore ACCESS
- **PI:** Aayush Rai, Clemson University (ACCESS ID `arai3`)
- **Project title:** Geometry-Structured Attention for Robust Vision-Language Control
- **How the project will be used:** Dissertation or Thesis
- **Opportunity activities:** Machine learning; Software development
- **How Aayush heard about ACCESS:** PhD advisor or Word of mouth, whichever is accurate
- **Users outside the United States:** No
- **Primary field of science:** Artificial Intelligence and Intelligent Systems
- **Secondary field:** Robotics and/or Control Systems, if available in the form taxonomy
- **Supporting grant:** No
- **Resource:** ACCESS Credits (400,000-credit Explore allocation)

## Keywords

transformer attention, vision-language-action models, robot learning, control systems,
geometry-structured attention, efficient inference, robustness, distributed training

## Public overview — draft

This project will evaluate geometry-structured multi-head attention (GT-MHA), implemented
in our research codebase through Lie-generated metric attention, for efficient and robust
vision-language control. Standard multi-head attention learns largely independent
query, key, and value transformations for each head. GT-MHA instead generates
head-specific interaction geometries from a compact shared representation. This design
may reduce attention parameters and key-value-cache storage while preserving the
multiple relational views needed for multimodal control.

The central goal is to determine whether this structured attention mechanism can support
closed-loop decision making under changes in viewpoint, target orientation, scene
configuration, and task instruction. We will first run controlled architecture-screening
experiments to identify stable GT-MHA configurations and matched conventional-attention
baselines. We will then train and evaluate vision-language-action policies on the LIBERO
robot-learning benchmark, with emphasis on task success, generalization across scene and
instruction variations, convergence, and sensitivity to random seed. Where the benchmark
permits, evaluation will include perturbations relevant to perception-driven control,
such as camera-view changes or altered object configurations. These experiments connect
the proposed transformer work to the PI's background in control systems, mobile robotics,
and camera-based target orientation.

ACCESS resources are needed for repeated GPU training, multimodal vision processing,
robustness sweeps, and matched comparisons between GT-MHA and standard multi-head
attention. Most runs will use one or two modern NVIDIA GPUs, with a small number of
four-GPU jobs used to verify distributed-training behavior. The software stack uses
Python, PyTorch, CUDA/bfloat16, PyTorch DistributedDataParallel, Hugging Face tools,
MuJoCo/LIBERO, checkpointed Slurm jobs, and reproducible experiment tracking. An initial
environment-validation phase will establish a portable Apptainer or Conda environment
on the selected ACCESS system.

The requested allocation will support a robotics-centered workstream within a broader
collaboration on structured attention. It will not repeat or double-charge language-model
training and core architecture sweeps performed under another allocation. Expected
outputs include open-source experiment configurations, reproducible robustness and
efficiency reports, trained-model evaluation artifacts, and research results suitable
for a technical report, preprint, or peer-reviewed publication.

## Distinct experimental scope

1. **Controls-oriented benchmark design:** define evaluation perturbations and success
   criteria motivated by camera-based mobile-robot control and target orientation.
2. **VLA architecture screening:** compare selected GT-MHA variants against matched MHA
   baselines before committing to longer runs.
3. **Robustness and generalization:** measure task success across scene, viewpoint,
   object-configuration, and instruction changes supported by the benchmark.
4. **Efficiency accounting:** report parameter count, peak memory, throughput,
   convergence per GPU-hour, and estimated inference-cache requirements.
5. **Reproducibility:** run selected seeds, retain checkpoints and configurations, and
   publish an auditable summary separating this allocation's runs from related projects.

## Preliminary GPU-hour plan

The exact GPU-hour equivalent should be rechecked in the ACCESS exchange calculator at
the time credits are exchanged.

| Activity | GPU-hours |
|---|---:|
| Environment, dataset, and benchmark validation | 80 |
| Matched GT-MHA/MHA architecture screening | 260 |
| VLA training across selected configurations and seeds | 620 |
| Controls-oriented robustness and generalization evaluation | 260 |
| Multi-GPU verification, checkpoint recovery, and contingency | 180 |
| **Planned workload for the initial 200,000-credit exchange** | **1,400** |

If ACCESS issues the Explore award in two credit increments, the second increment should
be exchanged only after reviewing first-stage results and defining additional,
non-duplicative experiments in the required progress report.

## Personnel and collaborator entries

- **Principal Investigator:** Aayush Rai, Clemson University
- **Faculty advisor:** Yue Wang, Ph.D., Department of Mechanical Engineering,
  Clemson University
- **Other collaborator:** Shahil Shaik, Clemson University — disclose here if Shahil has
  a significant connection to the research activity, even if he will not use or manage
  this ACCESS allocation
- **Additional project personnel:** Add only people who will use or manage this ACCESS
  project; confirm their ACCESS IDs and roles before entry

## Evidence relevant to PI fit

Aayush Rai's public Google Scholar profile identifies him as a Clemson Ph.D. student in
control systems and dynamics. It lists the 2023 publication "3D Coverage Control and
Target Orientation Alignment Using Unmanned Ground Vehicle with Onboard Camera Sensor"
(A. Rai and Y. Wang). The CV should add complete bibliographic details from Aayush's
résumé or the publication record once verified.

## Advisor-letter adaptation — draft text

The existing Shahil letter can be used as a formatting and content template with the
advisor's permission, but a letter naming Aayush must be reviewed and signed by the
advisor. Do not alter a previously signed letter or reuse its signature on changed text.

> ACCESS Allocations Review Committee:
>
> I am writing in support of Aayush Rai's Explore ACCESS allocation request. Aayush is a
> Ph.D. student in the Department of Mechanical Engineering at Clemson University, and I
> serve as his faculty advisor.
>
> I am aware of and support the proposed project, "Geometry-Structured Attention for
> Robust Vision-Language Control." The work will evaluate GT-MHA/Lie-generated metric
> attention in vision-language-action models, with emphasis on closed-loop task success,
> robustness to scene and viewpoint changes, computational efficiency, and reproducible
> comparison with conventional multi-head attention.
>
> I intend to engage with and guide the computational and research activities conducted
> under this allocation, including experimental design, responsible resource use,
> interpretation of results, reproducibility, and dissemination. I will remain involved
> throughout the allocation period and will support Aayush in meeting applicable ACCESS
> reporting and resource-use requirements.
>
> Sincerely,
>
> Yue Wang, Ph.D.  
> Faculty Advisor  
> Department of Mechanical Engineering  
> Clemson University

## Items still needed before form entry

- Aayush's old résumé, for creation of a PI CV/resume PDF no longer than three pages
- Confirmation of the field-of-science labels available in ACCESS
- Confirmation whether Shahil has a significant connection requiring disclosure under
  **Other Collaborators**; he need not be added as ACCESS project personnel if he will
  not use or manage the allocation
- A newly reviewed and signed advisor letter on Clemson letterhead naming Aayush
