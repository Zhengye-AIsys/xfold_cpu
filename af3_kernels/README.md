# Optional C++ / LIBXSMM infrastructure (stage 3)

This is a separate, optional package. Installing or importing xfold does not build,
load, or require it. No model operator is replaced in this stage. The stage 1 FP32
and stage 2 BF16 autocast paths keep using ordinary PyTorch operators.

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

## Explicit tools (not used by the model)

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

Future operator migration must explicitly handle stage 2's FP32 stored weights
and BF16 autocast state, including weight packing/cache lifecycle. This stage
does not change the baseline to explicit all-BF16 storage.

## Validation status

On the development Mac, validation is source-only:

```bash
python -m unittest discover -s tests -p 'test_*static.py' -v
```

The checks cover Python syntax, optional loading, build configuration, packaging,
source checksums, native interface/API consistency, and preservation of model
operators. They do not compile the native extension or test GEMM/VNNI correctness.
Native build, alignment/layout tests, FP32/BF16 comparisons, and AMX verification
remain required on the target cluster.

## Attribution

The stage 3 setup script, minimal bindings, and Python loader are newly implemented.
The padding and packing tools are rewritten using the reference interfaces and
LIBXSMM APIs; the reference GLU, Triangle Multiplication, Attention, and timing
implementations were not copied. Newly written source files carry no Xflops brand
or copyright claim. Third-party licenses and required notices are retained.

Padding and VNNI interfaces follow the BSD-licensed af3_kernels reference.
See `LICENSE` for Xflops/Intel notices. LIBXSMM retains its upstream BSD-3-Clause
license in `third_party/libxsmm/LICENSE.md`; a copy is packaged in
`src/af3_kernels/licenses/LIBXSMM.txt` for binary distributions.
