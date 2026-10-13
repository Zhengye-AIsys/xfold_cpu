## ASC25 AlphaFold 3 Code and Individual Contributions

During ASC25, I worked on the AlphaFold 3 challenge, which I subsequently extended into my undergraduate dissertation.

For my dissertation, I independently implemented three operators using Intel AMX instructions. My work was informed by publicly available optimization strategies from other university teams.

After the competition, I also learned through technical discussions that Intel had previously explored related operator optimizations for AlphaFold 2. https://github.com/IntelLabs/open-omics-alphafold?utm_source=chatgpt.com

To clarify, the three AMX operators were developed as part of my undergraduate dissertation and were not used in our ASC25 competition implementation. The GPU and CPU parallelization optimizations during ASC25 were implemented by other team members.

I would be happy to share the source code of my undergraduate dissertation. The source code only includes CPU version. The GPU technical detail can be shown on the 'asc25-proposal.pdf'.

| Optimization                                                 | Implementation                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **CPU BF16 Mixed-Precision Inference**                       | Enables CPU BF16 autocast while retaining FP32 model parameters and floating-point input storage. The FP32 baseline remains selectable. |
| **Tiled GLU Fusion[Me]**                                     | Uses LIBXSMM for the two GLU projections and performs SiLU/multiplication in C++, avoiding a full concatenated projection buffer. Applied to Transition and DiffusionTransition. |
| **LIBXSMM Triangle Multiplication[Me]**                      | Moves projections and channel-wise contractions to tiled native GEMM, supporting both incoming and outgoing directions. |
| **Query-Blocked Attention[Me]**                              | Computes full-key softmax in query blocks with per-thread workspace. Native projections and attention are integrated into GridSelfAttention and diffusion SelfAttention. |
| **Persistent BF16 Weight Packing**                           | Caches VNNI2-packed weights for native GLU, Linear and Triangle operations. Handles odd inner dimensions through padding and invalidates cached copies when weights change. |
| **Static Feature Reuse[Me]**                                 | Caches reusable recycle embeddings, relative/bond features, diffusion conditioning, pair logits and atom-encoder layout data within each model call. |
| **Optional IPEX LayerNorm[Me]**                              | Adds a selectable IPEX functional LayerNorm backend while retaining existing parameters and the PyTorch implementation. |
| **CPU Thread and NUMA Configuration[Other Team members]**    | Provides configurable intra-op/inter-op threads, OpenMP/MKL settings, CPU affinity and NUMA memory placement. |
| **Configurable Sample and Task Parallelism [Other Team members]** | Distributes complete diffusion trajectories or input/seed jobs using cyclic, block or weighted scheduling. Supports idle ranks, MPI launchers, Gloo and optional oneCCL communication. |

## ASC25 Team Size and Individual Responsibilities

Our ASC25 team consisted of five students and two coaches.

I was responsible for two competition tasks:

- **AlphaFold 3:** Both the preliminary and final rounds.
- **DeepSeek CPU Inference:** The final round.

See the 'asc25-proposal.pdf'

The preliminary rounds was online and final round was in the Qinghai University.

## Performance Improvements and Hardware Configuration

| Platform         | Hardware Configuration      |
| ---------------- | --------------------------- |
| CPU              | Intel Xeon Gold 6459C       |
| CPU Cores        | 32 vCPU                     |
| CPU Memory       | 64 GB                       |
| GPU              | 2 × NVIDIA A800 80GB PCIe   |
| GPU Platform GPU | 2 × Hygon C86 7380（32 核） |

| Sample | Baseline（original JAX） | Optimized（GPU） | Speedup |
| ------ | ------------------------ | ---------------- | ------- |
| 37aa   | 111.06 s                 | 36.29 s          | 3.06×   |
| 107aa  | 111.42 s                 | 36.29 s          | 3.07×   |
| 301aa  | 123.69 s                 | 36.57 s          | 3.38×   |
| 436aa  | 124.26 s                 | 41.73 s          | 2.98×   |
| 583aa  | 157.61 s                 | 56.81 s          | 2.77×   |
| 1491aa | 399.32 s                 | 331.44 s         | 1.20×   |

| Sample | Baseline（PyTorch FP32 CPU ） | Optimized（CPU） | Speedup |
| ------ | ----------------------------- | ---------------- | ------- |
| 37aa   | 579.17 s                      | 16.65 s          | 34.78×  |
| 107aa  | 600.45 s                      | 24.92 s          | 24.10×  |
| 301aa  | 2093.25 s                     | 97.55 s          | 21.46×  |
| 436aa  | 2092.30 s                     | 189.04 s         | 11.07×  |
| 583aa  | 4842.42 s                     | 319.10 s         | 15.17×  |
| 1491aa | ≥35000 s                      | 1703.39 s        | ≥20.55× |

End-to-end speedup ratio; speedup ratio tests for individual components were not conducted.

### GPU Kernel Programming Experience

I have not independently written CUDA or Triton kernels
entirely from scratch.

However, during ASC26, I explored GPU operator fusion
optimizations with the assistance of Codex.

In this project, I was responsible for analysing performance
bottlenecks and proposing operator fusion strategies.

I then used Codex as an AI coding assistant to help translate
these optimization ideas into GPU kernel implementations.

Therefore, while my hands-on experience with writing
low-level GPU kernels is still limited, I have gained
practical experience in identifying performance bottlenecks
and designing GPU optimization strategies.

If you are interested in, please see the 'asc26.pdf'

### Fujian Apex Internship

During my internship at Fujian Apex, my primary task was
deploying the DeepSeek 671B model on a server equipped
with eight NVIDIA A800 80GB GPUs.

The deployment used AWQ 4-bit quantization (download from Hugging face) to reduce
the model's memory requirements and fit the model
within the available GPU memory.

I successfully deployed the model and made it
operational on the target hardware.

To clarify, my work was primarily focused on model
deployment and configuration. I did not develop custom
GPU kernels or modify the underlying inference framework.

### C/C++ Systems Programming Experience

My experience with C/C++ systems programming is currently
limited, primarily involving the implementation of three
Intel AMX-accelerated operators for my undergraduate dissertation.

Beyond these operators, I have not yet gained extensive
hands-on experience in areas such as concurrency, networking,
or low-level memory management.

However, I am very interested in low-level systems programming
and performance optimization, and I am keen to strengthen
these skills through practical projects and further study.