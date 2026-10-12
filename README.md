# xfold: Democratize AlphaFold3

xfold is an open-source, PyTorch-based reimplementation of AlphaFold3, designed to accelerate protein structure prediction research and make cutting-edge AI technology more accessible to the scientific community.

Future developments for xfold will focus on integrating cutting-edge performance optimization techniques and advanced parallelization strategies. Our ultimate goal is to democratize AlphaFold3, empowering a broader researcher to contribute to and benefit from this transformative technology.

<div align="center">
    <img src="./assets/comparison.gif" width="400">
    <p><em>Visualization result comparison of 2pv7</em></p>
</div>

## Recent Developments 🚀

* **December 2024**: Successful migration to PyTorch, with validation confirming alignment with the original implementation

## Getting Started

### Step 1: Prepare the Environment

Follow the setup instructions provided in the [AlphaFold3 README](https://github.com/google-deepmind/alphafold3) to ensure dependencies are correctly installed and the AlphaFold 3 model parameters are downloaded.

### Step 2: Install xfold

Install this checkout, including its CPU adaptation, using pip:

```bash
python -m pip install -e .
```

### Step 3: Running Predictions

Execute protein structure predictions with the following command:

```bash
python run_alphafold.py \
    --db_dir=$PATH_TO_AF3_DATASET \
    --json_path=./fold_input.json \
    --model_dir=$PATH_TO_AF3_MODEL \
    --output_dir=./output
```

### CPU FP32 inference (stage 1)

`--device=auto` (the default) selects CUDA when available, otherwise CPU.
Use `--device=cpu` to select the CPU explicitly, including on a CUDA host.
CPU inference defaults to `--cpu_precision=fp32`, using FP32 model weights and
floating-point features with ordinary PyTorch operators and CPU autocast disabled.
It does not enable LIBXSMM, IPEX, or distributed inference.
Integer indices and boolean masks retain their dtypes.

For an input JSON that already includes MSA and template data:

```bash
python run_alphafold.py \
    --device=cpu \
    --cpu_precision=fp32 \
    --run_data_pipeline=False \
    --json_path=./fold_input_data.json \
    --model_dir=$PATH_TO_AF3_MODEL \
    --output_dir=./output_cpu_fp32
```

AlphaFold3 dependencies, chemical component data, and model parameters are still
required. The current weight loader expects `af3.bin.zst`, `fourier_weight.npy`,
and `fourier_bias.npy` inside `--model_dir`.
Triton is loaded only when its CUDA operators are selected; it is not
required by xfold's PyTorch CPU operator path. CUDA inference retains BF16 autocast
and uses Triton when `--fastnn=True`; use `--fastnn=False` for PyTorch CUDA operators.

The CPU adaptation can be checked without importing PyTorch, AlphaFold3, or Triton:

```bash
python -m unittest discover -s tests -p 'test_cpu_static.py' -v
```

These are source-level checks only. Numerical accuracy, complete CPU inference,
and performance must be validated separately on the target environment.

### CPU BF16 mixed-precision baseline (stage 2)

Set `--device=cpu --cpu_precision=bf16` to enable PyTorch CPU BF16 autocast for
the model forward pass. Model weights and floating-point input features remain
stored in FP32; integer indices and boolean masks are unchanged. PyTorch selects
the execution dtype for eligible operations. This follows the
[PyTorch CPU autocast approach](https://docs.pytorch.org/docs/2.6/amp.html).
It is a mixed-precision baseline, not the explicit all-BF16 conversion in the
reference `xfold_cpu` project.

```bash
python run_alphafold.py \
    --device=cpu \
    --cpu_precision=bf16 \
    --run_data_pipeline=False \
    --json_path=./fold_input_data.json \
    --model_dir=$PATH_TO_AF3_MODEL \
    --output_dir=./output_cpu_bf16
```

Compare with the FP32 command above using the same input, model parameters, seeds,
sample count, and CPU/thread settings. Both modes use ordinary PyTorch operators
and the same model code and padding strategy. FP32 mode explicitly disables CPU
autocast, even when invoked inside an existing CPU autocast context.

The noise schedule and initial diffusion positions retain their FP32 construction;
this stage adds no explicit BF16 casts to model features, coordinates, or weights.
Intermediate tensors follow autocast's operation-specific rules, including dtype
promotion and in-place updates. BF16 result tensors are converted back to FP32
before NumPy conversion and structure extraction. The startup log reports the
selected CPU precision. CUDA retains its existing BF16 autocast; requesting
`--cpu_precision=bf16` on a resolved CUDA device raises an error.

Use this command to run both stages' static checks without importing model dependencies:

```bash
python -m unittest discover -s tests -p 'test_*static.py' -v
```

Static checks do not establish BF16 numerical accuracy, hardware acceleration, or
speedup. Those require the target cluster and its PyTorch/AlphaFold3 environment.

### Optional C++ / LIBXSMM infrastructure (stage 3)

The separate [af3_kernels package](af3_kernels/README.md) adds native CPU tools,
a pinned LIBXSMM source snapshot, and an explicit Linux x86_64 build path.
It is not a dependency of the main xfold package. No model operator or inference
configuration was changed in stage 3. The default FP32 and BF16 baselines still
use PyTorch; stage 4 adds an explicit GLU opt-in described below.

On the target cluster, after preparing its CPU PyTorch environment:

```bash
MAX_JOBS=8 CC=gcc CXX=g++ \
  python -m pip install -v --no-build-isolation --no-deps ./af3_kernels
```

Importing `af3_kernels` does not build or load native code. Native loading happens
only when an explicit tool/probe is called. This stage has only been checked
statically on the development Mac; native compilation and execution remain
unverified. See the package README for build, tools, licenses, and validation steps.

### C++ / LIBXSMM GLU in Transition (stage 4)

`--glu_backend=cpp` selects the native GLU in both `Transition` and
`DiffusionTransition`. Two LIBXSMM projections and tile-local C++ SiLU/multiply
replace `matmul -> chunk -> silu -> multiply`. Normalization, the final projection,
and diffusion conditioning continue using their existing PyTorch implementations.
FP32 and BF16 are supported; this does not convert the whole Transition to native
code or prove that AMX executes.

The default `--glu_backend=auto` preserves previous behavior: PyTorch on CPU,
Triton on CUDA with `--fastnn=True`. `--glu_backend=torch` explicitly selects the
original PyTorch GLU. LayerNorm and Attention backend choices are independent.
The C++ path is CPU inference-only and requires rebuilding the optional package
on Linux x86_64. Missing packages, an older stage 3 binary, or unsupported native
shapes raise errors; there is no silent fallback.

On the target cluster, rebuild the optional package using the stage 3 command
above, then compare these four settings with the same input and inference flags:

| Comparison | Arguments |
| --- | --- |
| PyTorch FP32 baseline | `--device=cpu --cpu_precision=fp32 --glu_backend=torch` |
| C++ FP32 GLU | `--device=cpu --cpu_precision=fp32 --glu_backend=cpp` |
| PyTorch BF16 baseline | `--device=cpu --cpu_precision=bf16 --glu_backend=torch` |
| C++ BF16 GLU | `--device=cpu --cpu_precision=bf16 --glu_backend=cpp` |

For example, after rebuilding on the cluster:

```bash
python run_alphafold.py \
    --device=cpu --cpu_precision=bf16 --glu_backend=cpp \
    --run_data_pipeline=False \
    --json_path=./fold_input_data.json \
    --model_dir=$PATH_TO_AF3_MODEL \
    --output_dir=./output_cpu_bf16_cpp_glu
```

Stored model weights remain FP32. The bridge reads CPU autocast state, prepares
FP32/BF16 contiguous inputs, and caches one transposed/converted weight per
Transition. The cache refreshes when the weight's mutation counter, storage,
shape, stride, device, or dtype changes; tensors without a mutation counter are
not cached. The cache is not part of `state_dict`. BF16 VNNI2 packing is still
performed per native call; persistent packed-weight caching is future work.

The development Mac runs only the source checks listed above. Target validation
must cover native compilation, GLU/Transition numerical comparisons, edge shapes,
cache invalidation, full inference, and timing with matched CPU/thread settings.
See [native operator details and cluster validation](af3_kernels/README.md).

### Triangle Multiplication and Attention (stages 5–6)

Stage 5 adds `--triangle_backend=cpp` for both outgoing and incoming Triangle
Multiplication. Its four Linear operations and channel-wise contraction use tiled
LIBXSMM GEMMs. LayerNorm and sigmoid run through ATen in C++; this is not a new
LIBXSMM LayerNorm implementation. It preserves the original adjacent-channel
pairing, mask multiplication order, and return of an update without adding the
residual. Pairformer still owns the residual addition.

Stage 6 adds `--attention_backend=cpp` for `GridSelfAttention` and diffusion
`SelfAttention`. Q/K/V, query gating, Grid bias/output projections, and the two
attention GEMMs use native kernels. The SelfAttention Q bias is added to FP32
accumulation before output rounding. Attention computes full-key softmax in query
blocks, with scratch storage per OpenMP thread instead of a full extra logits
tensor. This is query-blocked attention, not an online/Flash Attention algorithm.
Grid normalization/transposition and diffusion adaptive normalization/output
conditioning retain their existing implementations. MSA and CrossAttention
remain unchanged in this stage.

Both flags accept `auto`, `torch`, and `cpp`. `auto` preserves the earlier choices;
`torch` selects the preserved PyTorch path. Each flag is independent of
`--glu_backend`, `--cpu_precision`, and the LayerNorm backend. All native operators
are CPU inference-only, require the rebuilt Linux x86_64 optional package, and
raise explicit errors rather than silently falling back. Parameter names,
initialization order, and checkpoint layouts are unchanged.

After rebuilding `./af3_kernels` on the cluster, compare these settings under
both `--cpu_precision=fp32` and `bf16`, using otherwise identical inputs, seeds,
sample counts, and CPU/thread settings:

| Comparison | Backend flags |
| --- | --- |
| Full PyTorch baseline | `--glu_backend=torch --triangle_backend=torch --attention_backend=torch` |
| Triangle only | `--glu_backend=torch --triangle_backend=cpp --attention_backend=torch` |
| Attention only | `--glu_backend=torch --triangle_backend=torch --attention_backend=cpp` |
| Combined stages 4–6 | `--glu_backend=cpp --triangle_backend=cpp --attention_backend=cpp` |

```bash
python run_alphafold.py \
    --device=cpu --cpu_precision=bf16 \
    --glu_backend=cpp --triangle_backend=cpp --attention_backend=cpp \
    --run_data_pipeline=False \
    --json_path=./fold_input_data.json \
    --model_dir=$PATH_TO_AF3_MODEL \
    --output_dir=./output_cpu_bf16_cpp_all
```

The original PyTorch functions are retained for comparisons, including their
finite `-1e9` attention mask semantics: fully masked rows produce a uniform
softmax over the real keys. Native attention preserves that behavior, supports
broadcastable bias/masks and unequal query/key lengths, and does not replace NaN
results with zeros. BF16 padding adds zero probabilities only for internal PV
packing. Stored model parameters remain FP32; converted contiguous weight copies
reuse the stage 4 mutation-aware cache. Packed weights still refresh per call.

Only source checks run on the Mac. Native compilation, dtype/layout correctness,
operator/module numerical errors, full inference, memory consumption, and AMX
execution/performance still require validation on the target cluster. See the
[expanded cluster checklist](af3_kernels/README.md).

### Repeated-computation caches (stage 7)

Stage 7 adds independently selectable packed-weight and input caches. The default
`--cache_mode=off` preserves prior behavior, including the earlier contiguous
weight copies. Precision and PyTorch/native backend flags remain independent.

| Mode | New caching behavior |
| --- | --- |
| `off` | Earlier stages' behavior |
| `weights` | Persist BF16 VNNI2 weights for native GLU/Linear/Triangle |
| `inputs` | Reuse immutable recycle/diffusion features within one model call |
| `all` | Enable both |

Packed weights refresh on source storage/version/layout/dtype/device changes.
Odd K is padded before full-width packing, and native calls bypass repacking.
Unversioned weights bypass reuse. FP32 and PyTorch operators do not use VNNI2.
Packed copies use additional storage beyond parameters and contiguous copies.

Input caches cover Evoformer initial pair/single embeddings, masks, relative
encoding and bonds; diffusion static single/pair conditioning; transformer pair
logits; and the denoising atom encoder's static conditioning, layouts and masks.
Noise coordinates, noise-level embeddings, noise-conditioned single transitions,
recycled activations, template processing and random MSA sampling keep executing.
Mutable initial activations are cloned before in-place updates; immutable pair
conditioning retains its identity for downstream caches.

Every `AlphaFold3.forward` starts a new scope and clears input caches before and
after inference, including exceptions. Recycle caches are released before
diffusion, and diffusion caches before confidence calculation. Weight caches can
persist. `model.reset_input_cache()` is also explicit. Keys use source identity,
mutation counters and layout, parameter versions and LayerNorm settings, head
layout, autocast and backend selection. Equal-shaped inputs do not share a cache
merely because of shape. Training, gradient-enabled and unscoped calls bypass
input reuse.

Features must be immutable within an example scope. PyTorch inference tensors
have no mutation counters, so these keys rely on identity plus scope boundaries
for those features; call `reset_input_cache()` before changing them in place.
Unversioned parameters are not cached. `.data`/external-storage weight writes
remain unsupported by version-based keys. See the
[PyTorch 2.6 inference-mode contract](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/autograd/grad_mode.py).

`--input_cache_max_mb=512` sets the example-wide retained input-storage budget in
MiB; `0` disables retention. Full backing storage is charged, including views.
Entries that do not fit are recomputed. This budget excludes packed weights,
existing model storage and temporary allocations, and does not bound process RSS.

On the cluster, rebuild optional package 0.4.0 before using native packed weights,
then compare the following flags with otherwise identical inference commands:

```text
--cache_mode=off
--cache_mode=weights
--cache_mode=inputs --input_cache_max_mb=512
--cache_mode=all --input_cache_max_mb=512
```

Use PyTorch with `inputs` to isolate repeated-computation savings, then native
BF16 with `weights`, then combine them. Hold input, seed, samples and threads
fixed. Validate A → B → A on one reused model against fresh models, multiple seeds,
changed conditioning, precision/backend switches, weight reloads, odd sizes,
capacity limits and exception cleanup. Use fresh processes for memory/timing
comparisons. Mac static checks do not establish accuracy or performance; see the
[stage 7 cluster checklist](af3_kernels/README.md).

### IPEX LayerNorm and CPU runtime configuration (stage 8)

`--layer_norm_backend=auto|torch|ipex` selects LayerNorm independently of
GLU/Triangle/Attention, precision and caching. `auto` preserves PyTorch CPU and
the existing fastnn CUDA selection; `torch` selects the original PyTorch
implementation. `ipex` explicitly loads the optional Linux x86_64 CPU dependency
and checks `ipex.llm.functional.fast_layer_norm` before model construction.
Missing/incompatible dependencies fail clearly. Stored parameters, checkpoint
keys and FP32/BF16 autocast behavior remain unchanged. IPEX is not a required
dependency, and no whole-model optimization, tracing or compilation is enabled.
Native Triangle's internal normalizations still call ATen directly.

`--cpu_threads` and `--cpu_interop_threads` are optional positive integer CPU
settings. Unset values preserve runtime defaults. Settings are applied before
model/weight construction; an initialized inter-op pool requires a fresh process
for a differing value. Startup logs record PyTorch counts, allowed CPUs (when
available), and OpenMP/MKL affinity environment settings.

Use `scripts/run_cpu.sh` on the cluster to apply OpenMP/MKL thread counts, optional
core/NUMA binding and explicit shared-library preloads before Python starts:

```bash
# Inspect the command only. 8 is an illustration, not a cluster tuning result.
XFOLD_CPU_THREADS=8 XFOLD_CPU_INTEROP_THREADS=1 \
  bash scripts/run_cpu.sh --dry-run \
    --cpu_precision=bf16 --layer_norm_backend=torch \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./output_stage8_torch
```

Run the same command on the target cluster without `--dry-run`; change
`--layer_norm_backend=ipex` and the output directory for the IPEX comparison.
Strict backend comparisons use fresh processes because importing IPEX registers
process-wide PyTorch operators. Numerical accuracy and speed are unvalidated on
this Mac. Configuration examples, API sources, topology/runtime considerations,
and the later cluster checklist are in [the stage 8 guide](docs/stage8_cpu.md).

## Acknowledgments

The xfold source code is licensed under the Apache License, Version 2.0.

This project may include code adapted from or inspired by AlphaFold 3 source code, which is licensed under Apache-2.0.

AlphaFold 3 model parameters/weights are not included in this repository and are subject to Google DeepMind's separate AlphaFold 3 Model Parameters Terms of Use. Users are kindly requested to review and comply with the AlphaFold3 license, available at https://github.com/google-deepmind/alphafold3?tab=readme-ov-file#licence-and-disclaimer.

## Contributing

We welcome contributions from the research community! Open an [issue](https://github.com/shenggan/xfold/issues) or send a [pull request](https://github.com/shenggan/xfold/pulls).
