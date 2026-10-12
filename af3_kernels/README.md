# Optional C++ / LIBXSMM CPU operators (stages 3–7)

This is a separate, optional package. Installing or importing xfold does not build,
load, or require it. Stage 3 added tools; stage 4 added opt-in GLU. Stages 5/6
add opt-in Triangle Multiplication and Grid/SelfAttention. Default FP32/BF16 CPU
inference still uses PyTorch. Package version 0.4.0 must be rebuilt on the cluster
to include the new bindings, even if an earlier stage is already installed.

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

Stage 7 optionally extends these caches to retain VNNI2 weights; the default
`off` mode preserves stage 4/5/6's per-call packing behavior.

## Triangle and Attention operators (stages 5–6)

All new raw APIs require CPU inference tensors with resolved views and no
gradients. They do not infer CPU autocast state. The xfold bridges handle autocast,
detaching inference parameters, contiguous copies, and cache invalidation.

- `linear(x, weight, bias=None)` accepts `[..., K]`, transposed weight `[K, N]`,
  and optional `[N]` bias, all contiguous with matching FP32/BF16 dtype. Output
  is `[..., N]` in that dtype. It flattens leading dimensions, tiles the matrix
  operation, adds the selected-dtype bias to FP32 accumulation, and only then
  rounds/stores the result. Batched tiled GEMM is also used internally by Triangle.
- `triangle_multiplication(pair, mask, outgoing, left_weight, left_bias,
  projection, gate, center_weight, center_bias, output_weight, gating_weight,
  left_eps=1e-5, center_eps=1e-5, block_size=128)` returns only the Triangle update
  `[N, N, C]`. Pair input is contiguous FP32/BF16; mask may be absent, scalar,
  `[N]`, or broadcastable to `[N, N]` and is used multiplicatively. Linear weights
  are contiguous `[C, 2*C]`, `[C, 2*C]`, `[C, C]`, `[C, C]`, all in the selected
  compute dtype. Normalization vectors keep their own compatible input dtype,
  with optional biases. ATen implements LayerNorm/sigmoid in C++; LIBXSMM handles
  projections and contraction. Adjacent projected channels form A/B. Outgoing
  computes `A @ B.T`; incoming computes `B.T @ A`. Projection/gate/contraction
  inputs are released before output projections to limit overlapping allocations.
- `dot_product_attention(q, k, v, mask=None, bias=None, query_block=128,
  q_pre_scaled=False)` accepts contiguous matching-dtype FP32/BF16 Q/K/V of
  shapes `[B, H, Q, D]`, `[B, H, K, D]`, `[B, H, K, Dv]`. Q/K may have different
  lengths, and D/Dv need not match. Bias is FP32/BF16 and broadcastable to
  `[B, H, Q, K]`. Mask rank 1 means `[K]`, rank 2 means `[B, K]`; those ranks are
  converted to bool. Other mask ranks up to 4 must already be boolean and
  broadcastable. True means a valid key. Broadcast views retain their strides
  rather than materializing full mask/bias tensors. Zero-length keys return a
  zero output, and empty batches/heads/queries return empty outputs.

Native attention scales Q once, uses LIBXSMM for QK/PV, and uses FP32 max/exp/sum
inside a query tile for softmax. Projection, bias addition, softmax, and PV outputs
follow the selected dtype boundaries. The xfold bridge scales Q before casting
for the first matmul, then passes `q_pre_scaled=True` to avoid scaling twice.
The finite `-1e9` mask is applied after bias, matching the original PyTorch code;
fully masked real rows retain its uniform softmax. NaNs/infinities are not hidden
with `nan_to_num`. Odd BF16 head/K dimensions are padded for VNNI2; internal PV
padding receives zero probability. This computes full-key softmax per query block,
not online softmax. Scratch is proportional to `threads * query_block * K`, plus
one FP32 exponential row per thread; packing buffers and original model bias
storage are additional allocations. It does not prove reduced total peak memory
or a speedup without cluster measurements.

`GridSelfAttention` uses native bias/Q/K/V/gating/output projections and native
attention, while its normalization and transpose flow remain in Python/PyTorch.
Diffusion `SelfAttention` uses native Q (including bias)/K/V/gating and attention,
while adaptive normalization/output conditioning remain PyTorch. MSA and
CrossAttention are unchanged. Each module keeps its existing parameter names
and owns stage 4 cache objects; no replacement class is selected at import time.

`--triangle_backend=auto|torch|cpp` and `--attention_backend=auto|torch|cpp` select
the two families independently. `cpp` preflights the required native symbols
before model/weight construction. PyTorch/CUDA defaults remain unchanged.

## Cache lifecycle and packed weight ABI (stage 7)

`--cache_mode=off|weights|inputs|all` selects caches independently of native
operators. `off` is the default. `weights` reuses BF16 packed parameters in native
GLU, Linear (including Grid/SelfAttention projections), and the four Triangle
projections. Changing Q/K/V activations and contraction operands are not retained.
FP32 stays on the ordinary path. Source version/storage/layout/dtype/device keys
invalidate contiguous and packed copies; unversioned weights are packed without
retention. Stored FP32 parameters and checkpoint keys remain unchanged.

The 0.4.0 binding exposes `_C.packed_weight_api == 1`. GLU/Linear accept
`weight_packed=False`; Triangle accepts `weights_packed=False`. True means BF16
full-width VNNI2 (`BLK_SZ=0`) with original K zero-padded to the next even value.
The packed tensor is opaque storage of shape `[K_padded, N]` (or `[K_padded, 2*N]`
for GLU). Original input K is validated independently; inputs are padded and
prepacked weights bypass conversion. Triangle contraction operands stay normal.
The bridge requests True only for selected BF16 packed caches. Ordinary calls
preserve earlier argument lists. Missing stage 7 markers are rejected before
model construction when native packed caching is requested, and on direct packed
cache use. Rebuild in the active target PyTorch environment; imports never build.

Input caches use `xfold.nn.cache.InputCache` inside `input_cache_scope()`. Every
AlphaFold3 call starts fresh and clears input data on exit, including exceptions.
Recycle and diffusion phase boundaries release their feature caches too.
`AlphaFold3.reset_input_cache()` clears feature caches while retaining weights.
For standalone module comparisons, explicitly enter `input_cache_scope()` under
`torch.inference_mode()` with eval modules; unscoped/gradient/training calls bypass
input reuse.

Cached components are initial recycle embeddings/masks/relative/bond features,
diffusion static conditioning, diffusion/atom transformer pair logits, and atom
encoder static conditioning/layout conversions. Dynamic positions, Fourier noise
embedding, noise-conditioned single transitions, queries/key activations, MSA
sampling and recycled updates remain uncached. Clone-before-update protects
mutable initial activations and diffusion single conditioning; pair conditioning
stays read-only. The trunk's one-shot atom encoder bypasses retention, preserving
capacity for repeated denoising.

Keys include scope, recursive dependency identities/versions/layouts, versioned
parameters, LayerNorm eps/shape, head layout, precision and backend settings.
Sources are retained to prevent allocator/id reuse. Versionless inference feature
tensors are immutable within an explicit scope; reset caches before mutating
them. Versionless parameters bypass reuse, and `.data`/external-storage writes
bypass mutation tracking. Multiple input/seed model calls use fresh scopes.

The example-wide quota defaults to 512 MiB (`--input_cache_max_mb`). Entries charge
full backing storage, deduplicating aliases within one entry and conservatively
charging aliases across entries. Replacement/reset releases quota. Entries that
do not fit execute normally without retention. The limit excludes weight copies,
existing inputs, scratch and transient storage. Packed caches retain contiguous
and packed copies until invalidated; compare memory in fresh processes.

## Validation status

On the development Mac, validation is source-only:

```bash
python -m unittest discover -s tests -p 'test_*static.py' -v
```

The checks cover Python syntax, optional loading, build configuration, packaging,
source checksums, native interface/API consistency, backend selection, autocast
bridging, cache lifecycle, both Transition call sites, and edge-shape guards.
Stages 5/6 also check explicit dispatch, preserved parameter modules, Triangle
directions/interleaving/residual ownership, attention broadcast/scaling/masking,
bias rounding, and bounded scratch allocation before parallel execution.
Stage 7 adds scope/cleanup, training/gradient bypass, dependency/weight keys,
capacity accounting, immutable/mutable boundaries and packed-weight guards.
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

Additional stages 5/6 cluster validation:

1. Rebuild the 0.3.0 package and check the `linear`, `triangle_multiplication`, and
   `dot_product_attention` symbols before model inference.
2. Compare Linear with/without bias and Triangle in both directions against
   PyTorch, first FP32, then explicit BF16, then BF16 autocast with FP32 stored
   parameters. Include fractional masks, empty dimensions, odd channel/token
   sizes, and tiles below/equal/above the block size. Confirm residual addition
   happens exactly once and weights/checkpoint keys remain unchanged.
3. Compare attention with D and Dv equal/unequal, odd/even D/K, unequal Q/K
   lengths, query-block remainders, absent bias/mask, broadcasted and noncontiguous
   bias/mask, rank-1/rank-2 key masks, rank-3/rank-4 boolean masks, and empty keys.
   Cover partially/fully masked rows, extreme logits, and NaN/infinity behavior.
   Measure absolute/relative errors and choose tolerances from recorded results.
4. Compare both Grid transpose directions and conditioned/unconditioned diffusion
   SelfAttention, including Q bias. Reload/mutate/move weights and switch precision
   to check cache refreshes. Run end-to-end comparisons with Triangle and Attention
   individually enabled before combining them with stage 4 GLU.
5. Hold input/seeds/samples/threads fixed; record cold/warm timings and peak memory
   as well as output differences. Inspect generated kernels to verify AMX use.

Additional stage 7 cluster validation:

1. Rebuild 0.4.0 and verify the ABI marker. Compare cached/unpacked BF16 GLU,
   Linear with bias, Triangle in both directions, and Grid/SelfAttention,
   including odd K/output widths and block remainders. Confirm no double packing
   or parameter/input mutation. Record numeric errors rather than assuming
   bitwise equality from static checks.
2. Compare `off`, `weights`, `inputs`, `all` at matching precision/backends. Reuse
   one model for A → B → A and multiple seeds, including equal-length different
   proteins; compare each with fresh models and the uncached path. Change trunk
   embeddings and conditioning-enabled/disabled modes too.
3. Reload/mutate tracked weights, move/change dtype, switch backend/LayerNorm eps,
   call `reset_input_cache`, and force exceptions. Confirm feature data is released
   while valid packed weights can persist across examples.
4. Repeat with quotas 0/small/large and oversized pair logits. Recomputed paths
   must preserve results. Training/gradient-enabled and standalone unscoped calls
   must not retain input features.
5. Record cold/warm timings and peak memory in separate processes. Include
   signature checks, clones, packing and capacity effects before attributing
   a performance or memory improvement.

## Attribution

The stage 3 setup script, minimal bindings, and Python loader are newly implemented.
The padding and packing tools are rewritten using the reference interfaces and
LIBXSMM APIs. The stage 4 GLU is a new implementation of the reference's tiled
projection idea, using the pinned public LIBXSMM API rather than copying its
functor/timing framework. Stages 5/6 implement the Triangle and blocked Attention
ideas with the pinned public LIBXSMM API and the original xfold semantics; the
reference implementations and timing framework were not copied. Newly written
source files carry no Xflops brand
or copyright claim. Third-party licenses and required notices are retained.

Padding and VNNI interfaces follow the BSD-licensed af3_kernels reference.
See `LICENSE` for Xflops/Intel notices. LIBXSMM retains its upstream BSD-3-Clause
license in `third_party/libxsmm/LICENSE.md`; a copy is packaged in
`src/af3_kernels/licenses/LIBXSMM.txt` for binary distributions.
