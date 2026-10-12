# Optional C++ / LIBXSMM tools and GLU (stages 3–4)

This is a separate, optional package. Installing or importing xfold does not build,
load, or require it. Stage 3 added tools only. Stage 4 adds a GLU operator selected
explicitly by `--glu_backend=cpp`; default FP32/BF16 inference still uses PyTorch.

## Source and build

The LIBXSMM build-source snapshot is pinned to
`0d1f61ed425aae8318fffafeea9834971a33aec7`, matching the reference af3_kernels
commit `40d3b7dfb8d2328c96e2f05863e3e0dfcd654301`.
`libxsmm.lock.json` records the upstream URL, revision, and file SHA-256 values.
Only build inputs are vendored; upstream sample datasets, tests, and documentation
are omitted. No Git metadata, compiled library, or path to the reference checkout
is included. Retained upstream source and licenses are unchanged.

Build on Linux x86_64 in the target CPU PyTorch environment. The reference
environment uses Python 3.11, PyTorch 2.6.0+cpu, and GCC/G++ 13.3. GNU Make,
setuptools, wheel, and an OpenMP-capable compiler are required. From xfold's root:

```bash
MAX_JOBS=8 CC=gcc CXX=g++ \
  python -m pip install -v --no-build-isolation --no-deps ./af3_kernels
```

`--no-build-isolation` makes the extension use the active PyTorch headers and ABI.
`--no-deps` avoids replacing that PyTorch installation. The build validates the
source checksums, copies only manifest-listed files into a fresh build directory,
rechecks the copies, builds static PIC
`libxsmm.a` and `libxsmmnoblas.a`, and links them into `af3_kernels._C`.
OpenMP is enabled for both compiling and linking the extension. `-march=native`
requires building on the intended target architecture; rebuild when moving to
another CPU or changing PyTorch.

`LIBXSMM_ROOT` may select another source checkout only if all pinned build files
match the manifest. It is read as an input, copied into a separate build workspace,
and never built in place. Normal builds use the vendored snapshot and need no
network access. Importing the Python package never invokes a compiler.

## Infrastructure tools

```python
import af3_kernels

print(af3_kernels.is_available())  # Explicit probe; does not compile anything.
print(af3_kernels.runtime_info())  # Requires the built native extension.
```

`runtime_info` reports the compiled revision, detected/selected LIBXSMM target,
PyTorch version, and OpenMP thread limit. A target name does not prove that AMX
instructions executed; that requires inspecting generated kernels on the target.

The native tensor tools are inference-only and accept CPU, strided FP32/BF16
tensors without gradients:

Conjugate/negative-bit views must be resolved before passing raw storage to these
tools. Aligned allocations use shared ownership so storage is also released if
tensor construction raises an exception.

- `pad_and_align_tensor(x, shape, need_init=True, init=0.0, alignment=64)`:
  preserves rank and dtype, pads without shrinking, and uses aligned storage.
  An already contiguous/aligned tensor of the requested shape may be returned
  unchanged. With `need_init=False`, padding contents are unspecified.
- `vnni_repack_tensor(x, type=2, BLK_SZ=0)`: contiguous, nonempty 2D BF16 input
  with an even row count. Type 2 packs pairs of rows in VNNI2 order; a positive
  block size must divide the columns. The output is opaque packed storage with
  the same shape, not an ordinary matrix for PyTorch operations.
- `gemm(a, b)`: a small-matrix infrastructure probe using LIBXSMM JIT, not a
  replacement for model Linear/Attention. Inputs are contiguous, nonempty 2D
  matrices with matching FP32/BF16 dtype and compatible dimensions. BF16 requires
  an even inner dimension and packs B before dispatch. Output and accumulation
  are FP32. Unsupported JIT shapes raise an error; there is no silent fallback.

## GLU operator and Transition integration (stage 4)

`gated_linear_unit(x, weight, M_BLK=128, N_BLK=128)` accepts contiguous CPU
FP32/BF16 tensors without gradients, with matching dtypes and shapes `[..., K]`
and `[K, 2*N]`. The weight is the transposed `nn.Linear.weight`; the first N
columns are the SiLU gate and the second N columns are the value projection.
The result has shape `[..., N]` and the input dtype. Arbitrary leading dimensions,
empty leading dimensions, partial M/N tiles, and odd K in BF16 are handled.
K and 2*N must be positive and fit LIBXSMM integers; block sizes must also be
positive LIBXSMM integers. Unsupported dtype, layout, shape, or JIT kernel fails
explicitly. Inputs are never modified.

The implementation uses two LIBXSMM GEMMs per tile, with FP32 accumulation.
BF16 weights are packed as VNNI2 over pairs of K rows; odd K is zero-padded first.
Per-thread scratch holds the two projection tiles. C++ SiLU and multiply write
directly into the final output, avoiding a full `[M, 2*N]` intermediate tensor.
GEMM descriptors, edge descriptors, and scratch allocation are prepared before
the OpenMP region. GEMM output, SiLU output, and final multiplication are stored
in the selected dtype to follow the PyTorch GLU's rounding boundaries. Floating
point accumulation and transcendental implementation can still produce numeric
differences; this has not been checked on the Mac.

The raw native API does not implement autocast or autograd. xfold's lazy bridge
in `xfold/fastnn/gated_linear_unit.py` reads CPU autocast before disabling it around
native dispatch, converts local inputs, and leaves the stored FP32 parameters
unchanged. Both ordinary and diffusion Transitions call this bridge. Each owns
one contiguous transposed/converted weight cache, which refreshes for versioned
mutations (including `load_state_dict`/the JAX loader), storage/shape/stride changes,
device/dtype changes, and FP32/BF16 mode changes. Unversioned inference tensors
bypass reuse. Mutating parameters through `.data` or external raw storage bypasses
PyTorch's mutation tracking and is not supported with this cache.

Only the contiguous converted weight is cached in this stage; VNNI packing runs
on each BF16 call. LayerNorm, the second Linear, adaptive conditioning, and other
model operators keep their existing implementations. `--glu_backend=auto` retains
the old backend selection, while `torch` and `cpp` select GLU explicitly. A `cpp`
request preflights loading before model/weight construction and rejects CUDA or
an old extension without the GLU symbol. There is no automatic fallback.

## Validation status

On the development Mac, validation is source-only:

```bash
python -m unittest discover -s tests -p 'test_*static.py' -v
```

The checks cover Python syntax, optional loading, build configuration, packaging,
source checksums, native interface/API consistency, backend selection, autocast
bridging, cache lifecycle, both Transition call sites, and edge-shape guards.
They do not compile the native extension or test GLU/GEMM/VNNI correctness.

On the target cluster, validation should proceed in this order:

1. Rebuild this package in the intended CPU PyTorch environment (including after
   a stage 3 installation), then inspect `runtime_info()` and the GLU binding.
2. Under `torch.inference_mode()`, compare native GLU with
   `torch.nn.functional.silu((x @ w)[..., :N]) * (x @ w)[..., N:]`.
   Cover FP32, explicit BF16, and the xfold CPU BF16 autocast bridge with FP32
   stored weights. Record absolute and relative errors before choosing tolerances.
3. Cover rank 1/2/3/4, empty leading dimensions, M/N below/equal/above 128,
   nonmultiples of 128, and odd/even K. Test noncontiguous bridge inputs and
   validation failures, not just the model's usual power-of-two dimensions.
4. Compare ordinary and diffusion Transitions under both backends using the same
   weights/inputs. Reload weights, mutate them with version-tracked operations,
   change dtype/device, and switch FP32/BF16 to validate cache refreshes.
5. Compare full inference with `--glu_backend=torch` and `cpp`, first FP32 then
   BF16, holding seeds, sample count, input padding, parameters, and thread settings
   fixed. Use separate output directories, warm up kernels, and distinguish cold
   loading/packing time from steady-state latency.
6. Inspect the generated GEMM kernels/target to establish AMX execution and then
   measure performance. Neither a successful import nor a target name proves AMX.

These cluster steps are documented for later execution; none ran on the Mac.

## Attribution

The stage 3 setup script, minimal bindings, and Python loader are newly implemented.
The padding and packing tools are rewritten using the reference interfaces and
LIBXSMM APIs. The stage 4 GLU is a new implementation of the reference's tiled
projection idea, using the pinned public LIBXSMM API rather than copying its
functor/timing framework. Reference Triangle Multiplication, Attention, and timing
implementations were not copied. Newly written source files carry no Xflops brand
or copyright claim. Third-party licenses and required notices are retained.

Padding and VNNI interfaces follow the BSD-licensed af3_kernels reference.
See `LICENSE` for Xflops/Intel notices. LIBXSMM retains its upstream BSD-3-Clause
license in `third_party/libxsmm/LICENSE.md`; a copy is packaged in
`src/af3_kernels/licenses/LIBXSMM.txt` for binary distributions.
