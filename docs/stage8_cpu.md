# Stage 8: optional IPEX LayerNorm and CPU runtime configuration

Only xfold is changed. xfold_cpu remains a read-only implementation reference.
The development Mac runs source and shell-syntax checks only; there is no Torch,
IPEX, model, binding, or native-kernel execution during those checks.

## LayerNorm selection

| Flag | Behavior |
| --- | --- |
| `--layer_norm_backend=auto` | Existing selection: PyTorch CPU; Triton CUDA when fastnn is enabled |
| `--layer_norm_backend=torch` | Original `F.layer_norm` implementation |
| `--layer_norm_backend=ipex` | Explicit CPU IPEX public functional API |

IPEX selection is independent of `--cpu_precision`, the three native operator
backend flags and `--cache_mode`. It does not imply BF16, LIBXSMM or AMX.
The wrapper retains the original LayerNorm module, parameter names, weight/bias
objects, optional affine parameters, normalized shape and epsilon. It passes
the current input and autocast context to IPEX without dtype conversion,
detachment or added contiguous copies. FP32 stored parameters remain FP32.
The opt-in path supports eval FP32/BF16 CPU inference; training, gradients that
require backpropagation, other devices/layouts/dtypes are rejected explicitly.

Only the selected IPEX path imports `intel_extension_for_pytorch`. The runner
preflights Linux x86_64, successful import and a callable
`ipex.llm.functional.fast_layer_norm` before allocating/loading the model.
Version/import errors, including IPEX's import-time exits, become an actionable
error; there is no silent backend fallback. Direct LayerNorm use has the same
lazy dependency checks. Auto/PyTorch require no IPEX installation.

On the cluster, install a **CPU IPEX release compatible with the existing
PyTorch environment**, following the target release's requirements. Do not
install a latest unpinned IPEX wheel over the validated native build environment.
If PyTorch changes, rebuild `af3_kernels` in that environment before native use.
The main xfold dependency list is unchanged; IPEX is installed separately.
The selected path logs IPEX and PyTorch versions and its public API name.

The interface follows the [IPEX CPU API documentation](https://intel.github.io/intel-extension-for-pytorch/cpu/2.6.0%2Bcpu/tutorials/api_doc.html)
and [public functional implementation](https://github.com/intel/intel-extension-for-pytorch/blob/main/intel_extension_for_pytorch/llm/functional/fusions.py).
IPEX versions may delegate FastLayerNorm to PyTorch's LayerNorm; the
[CPU implementation](https://github.com/intel/intel-extension-for-pytorch/blob/main/intel_extension_for_pytorch/transformers/models/cpu/fusions/mha_fusion.py)
currently does so. A successful call does not establish a faster kernel.
No `ipex.optimize`, whole-model replacement, JIT tracing or `torch.compile` is
introduced. Native Triangle's two normalizations retain their ATen calls.

Switching the LayerNorm configuration invalidates stage 7 input-cache keys.
Importing IPEX can register process-wide operators; selecting `torch` afterwards
does not unload them. Use separate fresh processes for strict PyTorch/IPEX
accuracy, timing and memory comparisons, including native combinations.

## Thread settings

`--cpu_threads` sets PyTorch intra-op parallelism; `--cpu_interop_threads` sets
inter-op parallelism. Both default to unset. Explicit values must be positive
32-bit integers and require a resolved CPU device. Configuration runs before
optional operator loaders and model/weight construction.

An already matching inter-op count is left alone, allowing repeated runners
with the same setting. A differing request that PyTorch cannot apply raises an
error requesting a fresh process. Inter-op is configured first so such a failure
does not partially change intra-op settings. There is no per-input thread reset.
See [PyTorch intra-op guidance](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)
and the [inter-op one-time-setting contract](https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html).

Python flags configure PyTorch; they do not guarantee the thread count of a
separately linked OpenMP runtime in LIBXSMM. The launcher exports consistent
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` values **before Python starts**. Direct Python
use logs conflicting environment values rather than changing environment
variables after library initialization. Logs also show PyTorch counts, allowed
CPU IDs when the platform exposes them, and selected affinity variables.

## Single-process cluster launcher

Use `bash scripts/run_cpu.sh [--dry-run] <run_alphafold.py flags>` from the checkout.
`--dry-run` must be the first argument; it prints environment/escaped command
without launching Python or numactl or checking target hardware. The actual
launcher requires Linux. It locates the runner relative to itself, while input,
model and output paths retain the caller's working-directory semantics.
It selects CPU and defaults to `--run_data_pipeline=False`; an explicit pipeline
flag can override that default. Thread and device flags are launcher-owned:
use the environment settings below rather than passing those three flags again.
Both single- and double-dash spellings of the three managed flags are rejected.
`--flagfile`/`-flagfile` is also rejected because nested file contents could
override the launcher settings; pass the runner flags directly. Other operator,
cache, precision and input flags are forwarded as arguments. This follows
[Abseil's flag parsing and flagfile precedence](https://github.com/abseil/abseil-py/blob/main/absl/flags/_flagvalues.py).

| Environment variable | Unset behavior / accepted values |
| --- | --- |
| `XFOLD_CPU_PYTHON` | `python`; executable name or path |
| `XFOLD_CPU_ENV_FILE` | No config file; explicitly selected readable shell file |
| `XFOLD_CPU_THREADS` | Inherit; positive integer sets PyTorch, OpenMP and MKL |
| `XFOLD_CPU_INTEROP_THREADS` | Inherit; positive integer sets PyTorch inter-op |
| `XFOLD_CPU_CPUS` | Inherit scheduler cpuset; CPU list such as `0-7,16-23` or `+0-7` |
| `XFOLD_CPU_NODES` | Inherit; NUMA node CPU list; mutually exclusive with `XFOLD_CPU_CPUS` |
| `XFOLD_CPU_MEM_POLICY` | `inherit`; optional `bind`, `interleave`, `local` |
| `XFOLD_CPU_MEM_NODES` | Empty; required for bind/interleave, forbidden for inherit/local |
| `XFOLD_CPU_AFFINITY` | `inherit`; optional `compact`, `spread`, `none` |
| `XFOLD_CPU_KMP_BLOCKTIME` | Inherit; optional nonnegative Intel OpenMP blocktime |
| `XFOLD_CPU_OMP_LIBRARY` | No preload; optional readable absolute library path |
| `XFOLD_CPU_ALLOCATOR_LIBRARY` | No preload; optional readable absolute library path |

Copy/edit `scripts/cpu_env.sh.example` outside tracked files, then select it
explicitly. It is sourced shell configuration, so use a file you maintain.
Its assignments override identically named values supplied in the environment.
The example has empty counts and inherited policies; no machine topology or
cluster paths are embedded.

Compact/spread use `OMP_PLACES=cores` and `OMP_PROC_BIND=close/spread`; none uses
`OMP_PROC_BIND=false`. Explicit affinity rejects conflicting `KMP_AFFINITY` or
`GOMP_CPU_AFFINITY` settings; unset those variables before using the portable
policy. Inherit preserves the existing runtime settings. KMP blocktime is
meaningful only with a runtime that supports it.

Exact CPU IDs use `numactl --physcpubind`; node CPU selection uses `--cpunodebind`.
Memory bind/interleave/local map to `--membind`/`--interleave`/`--localalloc`.
No binding request means no numactl dependency. The launcher uses arrays, checks
mutually exclusive/required settings and does not use eval or bypass scheduler
restrictions with `--all`. Numactl validates actual IDs, topology and permissions
on the target. See [the numactl source](https://github.com/numactl/numactl/blob/master/numactl.c).

Choose IDs and thread counts from the scheduler allocation and actual topology;
do not assume a socket equals one NUMA node or every logical CPU is a separate
physical core. Dry-run does not validate whether the target grants those CPUs.

Preloads are explicit absolute readable files without whitespace/colons;
existing `LD_PRELOAD` is preserved. No library is preloaded by default. Match
the OpenMP runtime used by PyTorch, IPEX and the native extension; do not preload
Intel OpenMP over a conflicting GNU OpenMP build or assume a copied lib path is
compatible. A preload path check establishes only file availability. Allocator
and runtime changes need separate cluster validation and memory measurements.

## Later cluster comparison

For an already featurized input, first compare LayerNorm with fixed threads:

```bash
XFOLD_CPU_THREADS=8 XFOLD_CPU_INTEROP_THREADS=1 \
  bash scripts/run_cpu.sh --dry-run \
    --cpu_precision=bf16 --layer_norm_backend=torch \
    --glu_backend=torch --triangle_backend=torch --attention_backend=torch \
    --cache_mode=off --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./stage8_torch
```

On the cluster remove `--dry-run`, then run a fresh process changing only
LayerNorm to `ipex` and the output directory. Repeat FP32/BF16. Then vary thread
counts, followed by affinity, then NUMA placement. An illustrative binding is:

```bash
XFOLD_CPU_THREADS=8 XFOLD_CPU_INTEROP_THREADS=1 \
XFOLD_CPU_CPUS=+0-7 XFOLD_CPU_MEM_POLICY=local XFOLD_CPU_AFFINITY=compact \
  bash scripts/run_cpu.sh --dry-run \
    --cpu_precision=bf16 --layer_norm_backend=ipex \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./stage8_bound
```

These numbers/policies are examples for command construction, not performance
recommendations. Keep input, seed, samples, weights and operator/cache choices
fixed during each comparison. Validate:

1. LayerNorm FP32/BF16, BF16 activation with FP32 parameters, bias/no bias,
   no affine parameters, rank/shape variations, noncontiguous and empty inputs;
   preserve output dtype and measure numerical error against PyTorch.
2. Missing/wrong IPEX dependency or API, invalid thread values, non-CPU/training/
   gradient calls, same/differing inter-op requests in a reused process, and
   stage 7 backend/cache invalidation.
3. Actual allowed CPU set, NUMA allocation and active OpenMP runtime; inspect
   `torch.__config__.parallel_info()` and native `runtime_info()`'s
   `openmp_max_threads` when using LIBXSMM. Measure affinity rather than assuming
   environment settings guarantee placement.
4. End-to-end accuracy and fresh-process cold/warm timing/peak memory; enable
   earlier native operators and caches individually before combining them.
   Validate optional runtime/allocator preloads separately.

## Static validation on this Mac

```bash
python3 -m unittest discover -s tests -p 'test_*static.py' -v
bash -n scripts/run_cpu.sh
bash -n scripts/cpu_env.sh.example
git diff --check
```

Checks cover source syntax, optional loading/preflight, preserved PyTorch/default
behavior, parameter/precision contracts, cache keys, thread validation/timing,
and launcher argument/environment/binding guards. Shell checks parse only; they
do not run even the dry-run launcher. No dependency installation, IPEX import,
Tensor/model execution, native build, or accuracy/performance test ran here.
MPI/oneCCL remains reserved for stage 9.
