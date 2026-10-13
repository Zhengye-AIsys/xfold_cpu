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

## CPU adaptation

The CPU implementation adds FP32/BF16 execution, optional C++/LIBXSMM operators,
feature and packed-weight caches, CPU runtime configuration, and sample/task
scheduling. PyTorch remains the default CPU backend. The parameter names and
checkpoint layout are retained; 

Python >=3.11 and PyTorch >=2.6 are required. Structure prediction also needs the
upstream AlphaFold3 environment, chemical component data, and model files
`af3.bin.zst`, `fourier_weight.npy`, and `fourier_bias.npy` in `--model_dir`.

| Area | Configuration | Implementation |
| --- | --- | --- |
| Precision | `--cpu_precision=fp32|bf16` | CPU autocast; parameters and floating-point inputs stored in FP32 |
| GLU | `--glu_backend=auto|torch|cpp` | GLU in ordinary and diffusion Transition |
| Triangle | `--triangle_backend=auto|torch|cpp` | Incoming/outgoing Triangle Multiplication |
| Attention | `--attention_backend=auto|torch|cpp` | GridSelfAttention and diffusion SelfAttention |
| LayerNorm | `--layer_norm_backend=auto|torch|ipex` | PyTorch or optional IPEX functional interface |
| Caches | `--cache_mode=off|weights|inputs|all` | Packed BF16 weights and per-call static features |
| Parallel work | `--distributed_mode=off|samples|tasks` | Complete diffusion trajectories or input/seed jobs |

CPU defaults are FP32, PyTorch operators, cache mode off, and serial execution.
`auto` backend selection keeps the previous CPU/CUDA choices. Triton is loaded
when a CUDA operator is selected. Explicit native choices check their dependencies
and report errors if unavailable.

### PyTorch baseline

For an input with completed MSA/template data:

```bash
python run_alphafold.py \
    --device=cpu --cpu_precision=fp32 \
    --run_data_pipeline=False \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./output_cpu_fp32
```

Change `--cpu_precision=bf16` and the output directory for the mixed-precision
baseline, keeping input, seeds, samples and thread settings fixed. Autocast selects
eligible computation dtypes; noise schedule and initial coordinates retain FP32
construction. BF16 outputs are converted to FP32 before NumPy conversion.

### Native operators and caching

Build optional package 0.4.0 on the target Linux x86_64 CPU:

```bash
MAX_JOBS=8 CC=gcc CXX=g++ \
  python -m pip install -v --no-build-isolation --no-deps ./af3_kernels
```

GLU uses two tiled LIBXSMM projections with C++ SiLU/multiply. Triangle uses
LIBXSMM projections and contraction with ATen normalization/sigmoid. Attention
computes full-key softmax in query blocks; Grid and SelfAttention projections also
use the native path. MSA/CrossAttention, adaptive normalization and conditioning
retain their existing implementations.

Enable each backend separately before combining them:

```bash
python run_alphafold.py \
    --device=cpu --cpu_precision=bf16 \
    --glu_backend=cpp --triangle_backend=cpp --attention_backend=cpp \
    --run_data_pipeline=False --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./output_cpp
```

`weights` caches BF16 VNNI2 parameters in native GLU/Linear/Triangle. `inputs`
caches static recycle/diffusion features within one model call; `all` enables both.
`off` retains earlier contiguous-weight caching while disabling the added caches.
Parameter mutation/version, layout, precision and backend changes invalidate
cached copies. Versionless parameters are not reused.

Input scopes are cleared at model entry/exit and phase boundaries. Dynamic
coordinates and noise-conditioned computations continue to run.
`--input_cache_max_mb=512` limits retained input-feature storage; oversized entries
are recomputed. The quota excludes weight copies, model storage and scratch.

Build details, API shapes, cache lifecycle and numerical checks are in the
[native operator guide](af3_kernels/README.md).

### CPU threads and NUMA

`--cpu_threads` and `--cpu_interop_threads` configure PyTorch thread counts.
`scripts/run_cpu.sh` also applies OpenMP/MKL environment, CPU binding and memory
policy before model Python startup:

```bash
XFOLD_CPU_THREADS=8 XFOLD_CPU_INTEROP_THREADS=1 \
  bash scripts/run_cpu.sh --dry-run \
    --cpu_precision=bf16 --layer_norm_backend=torch \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./output_cpu
```



### Sample and task scheduling

`samples` assigns complete diffusion trajectories for one input/seed. Rank 0
computes the trunk and output heads. `tasks` assigns full inputs or input/seed
pairs, with configurable `auto|input|seed` granularity. Each rank holds a complete
model; computation inside an operator remains local to that rank.

Scheduling supports cyclic, contiguous block, and weighted assignment, including
remainders and idle ranks. Counts are independent of physical nodes; sample and
step defaults remain 5 and 200. Sample mode uses indexed CPU RNG streams. Compare
with serial `--sample_rng=indexed`; default legacy sampling uses a different stream.

```bash
XFOLD_DIST_PPN=8 XFOLD_DIST_MODE=samples XFOLD_CPU_THREADS=4 \
  bash scripts/run_distributed.sh --dry-run \
    --cpu_precision=bf16 --num_diffusion_samples=7 \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./output_samples
```

## Acknowledgments

The xfold source code is licensed under the Apache License, Version 2.0.

This project may include code adapted from or inspired by AlphaFold 3 source code, which is licensed under Apache-2.0.

AlphaFold 3 model parameters/weights are not included in this repository and are subject to Google DeepMind's separate AlphaFold 3 Model Parameters Terms of Use. Users are kindly requested to review and comply with the AlphaFold3 license, available at https://github.com/google-deepmind/alphafold3?tab=readme-ov-file#licence-and-disclaimer.

## Contributing

We welcome contributions from the research community! Open an [issue](https://github.com/shenggan/xfold/issues) or send a [pull request](https://github.com/shenggan/xfold/pulls).
