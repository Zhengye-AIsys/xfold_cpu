# Stage 9: flexible CPU process and work scheduling

The reference xfold_cpu diffusion code fixes five samples, 200 steps and at most
five **ranks**, with a handwritten table of step segments and broadcasts. A rank
is a worker process, not necessarily a physical node. This xfold implementation
uses generic independent-work scheduling and leaves xfold_cpu unchanged.

## Modes and ownership

`--distributed_mode=off` preserves serial inference, its default RNG sequence,
operator choices, parameter names and FP32/BF16 storage/precision handling.
It never loads torch.distributed or oneCCL through the new coordinator.

`samples` processes inputs/seeds in their existing order. Rank 0 loads inputs,
featurizes and computes the Evoformer trunk once; it broadcasts embeddings with
their existing dtype. Each rank runs complete assigned diffusion trajectories,
then all ranks participate in result broadcasts in the same owner/sample order.
Rank 0 computes confidence/distogram, extracts structures and writes outputs.
Empty assignments still participate in coordination and communication.

`tasks` schedules complete inputs or input/seed pairs. Each worker uses its local
model for full serial inference on assigned jobs. Results are gathered to rank 0,
which writes each completed input once and ranks its samples in original seed
order. Workers never race over JSON, sample directories or ranking_scores.csv.
Granularity is selected with `--distributed_task_granularity=auto|input|seed`:
auto chooses input when there are multiple inputs, otherwise seed. Seed mode
flattens all input/seed pairs, allowing multiple inputs and seeds in the same wave.
Input mode avoids re-featurizing each seed as a separate input job.

Each rank has a full model replica, even if its assignment is empty. Model/weights
must fit per rank. This is task/sample parallelism; it does not implement the
reference's operator-level distributed Triangle/Attention kernels. Trunk and
output heads remain rank-0 work in sample mode. There is no fixed node/rank cap,
but increasing ranks beyond independent work does not create additional compute
parallelism. Five samples expose at most five concurrent diffusion trajectories.
Use more samples, seeds, inputs or fewer processes as appropriate.

## Scheduling and counts

`--num_diffusion_samples` and `--diffusion_steps` must be positive integers;
their defaults remain 5 and 200. There is no fixed-step segmentation table.

| `--distributed_schedule` | Assignment |
| --- | --- |
| `cyclic` | Rank r receives indices r, r+world_size, ... |
| `block` | Contiguous balanced ranges, with remainder assigned to early ranks |
| `weighted` | Contiguous ranges from cumulative integer capacities |

Weighted mode uses `--distributed_rank_weights=...`, with one nonnegative integer
per actual global rank and a positive sum. Zero gives no sample/task work to that
rank; rank 0 still performs its coordinating/trunk/output roles. For total work T,
range boundaries are `floor(T * prefix_weight / total_weight)`. Adjacent ranges
share boundaries, so they cover all work once, including nondivisible totals.
These are user-specified capacities, not measured throughput or automatic
load balancing. Task mode executes synchronized waves, so one slower job can
hold up the next wave; use the granularity and weights accordingly.

No rank count is inferred from the number of samples or channels. World size
comes from the launched process group and can be less than, equal to or greater
than work count. Physical host count, processes per host, threads per process and
task/sample count are separate settings.

## RNG and serial comparisons

`--sample_rng=auto|legacy|indexed` defaults to auto: legacy in off/task mode,
indexed in sample mode. Explicit legacy is rejected in sample mode.
Indexed mode uses stable BLAKE2b stream seeds from the original seed and sample
index, without rank, world size, Python hash or scheduling policy. Each sample
isolates/restores CPU RNG state and uses the original denoising step function.
It does not reseed CUDA inside the added sample stream. Initial coordinates,
noise schedule and original denoising arithmetic are preserved in their CPU
precision path; this stage does not fix/change the existing denoising formula.

For numerical comparisons, use serial **indexed** CPU sampling versus distributed
indexed sampling, holding input, samples, steps, model, backend and precision
fixed. Legacy serial and indexed sampling intentionally consume different RNG
streams and can produce different structures. Task mode with auto/legacy uses
the existing per-seed random reset; compare input/seed granularity with ordinary
serial mode. Actual numerical parity across CPU hardware remains unvalidated.

## Communication and lifecycle

`--distributed_backend=gloo|ccl` defaults to CPU Gloo. Multi-rank ccl explicitly
imports `oneccl_bindings_for_pytorch`; use CPU bindings matching the target
PyTorch environment. The [oneCCL binding repository](https://github.com/intel/torch-ccl)
is marked deprecated, so it remains an optional reference-compatible choice.
Gloo provides the ordinary PyTorch CPU communication path. A one-rank distributed
mode needs no process group or oneCCL import and reports that its backend is unused.

Runtime rank mapping prefers a complete RANK/WORLD_SIZE pair from torchrun, then
PMI (Intel MPI), Open MPI, then Slurm. Partial families and invalid values fail
clearly. Stale canonical RANK/WORLD_SIZE/LOCAL_RANK variables should be removed
before a fresh MPI launch. Multiple ranks require an explicit reachable
MASTER_ADDR and MASTER_PORT. Initialization is explicit env:// with configurable
`--distributed_timeout_seconds=3600`. This must cover the longest computation
phase, including trunk, a whole trajectory/job and output handling.

The [PyTorch distributed API](https://docs.pytorch.org/docs/stable/distributed)
defines rank/group setup and collective ordering. Controls use explicit CPU int64;
tensor data is broadcast as bounded uint8 views, preserving FP32/BF16 bits.
Variable objects use protocol-5 serialization over 16 MiB chunks, including
unequal/empty task results. Full objects are retained/decoded only where needed;
discarding ranks reuse one chunk buffer during root-only gathering. This avoids
requiring backend-specific object-gather APIs. The [CCL datatype/reduction mapping](https://github.com/intel/torch-ccl/blob/master/src/utils.cpp)
includes the control/data types used here.

`--distributed_max_payload_mb=4096` limits each serialized object, not total RSS
or tensor/model memory. Large features, serialized source data, decoded objects
and pending results need additional memory; raise the explicit budget only after
checking capacity. All ranks require compatible Python/PyTorch/AlphaFold3 classes
for object serialization. Verify input/result transport on the actual installed
AlphaFold3 release; the [Structure serialization implementation](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/structure/structure.py)
removes cached properties when pickling.

Managed phases agree on Python computation failures before the next collective,
including feature/model allocation, trunk, diffusion, conversion, structure
extraction and output. Errors report the failed rank and phase. Serialization,
receiving allocations and decoding have collective guards.
Tensor layout/contiguity and byte-view preparation also agree before broadcasting.
Process groups are destroyed in finally; no exit barrier is added on errors. A killed process,
transport failure, import failure before coordinator entry, or exhausted system
memory can still require launcher termination/collective timeout. There is no
checkpoint/resume or live rank join/leave support in this stage.

Stage 7 feature scopes/reset boundaries remain intact for every model call;
each process has private feature/weight caches. Communication, serialized data
and model replicas are outside the input-cache quota. Stage 8 precision/operator/
cache flags remain independent. Model/collective/input/output settings must match
across ranks; CPU thread counts and worker-local binding may differ.

Distributed modes currently require CPU, inference enabled, and completed
MSA/template data (`--run_data_pipeline=False`). Run the data pipeline separately.
Input output names must be distinct after sanitization and seeds unique within an
input, preventing accidental output collisions. Serial mode retains its existing
pipeline behavior.

## Launch options

Use `bash scripts/run_distributed.sh [--dry-run] <runner flags>` in a shared
checkout/environment. Dry-run must come first and prints only configuration and
escaped command; static checks on this Mac do not execute even that mode.

Common launcher environment:

| Variable | Default / role |
| --- | --- |
| XFOLD_DIST_LAUNCHER | torchrun / openmpi / intelmpi / external |
| XFOLD_DIST_MODE | samples / tasks |
| XFOLD_DIST_BACKEND | gloo / ccl |
| XFOLD_DIST_SCHEDULE | cyclic / block / weighted |
| XFOLD_DIST_RANK_WEIGHTS | Empty; comma-separated capacities for weighted mode |
| XFOLD_DIST_TASK_GRANULARITY | auto / input / seed |
| XFOLD_DIST_NNODES | 1; homogeneous launch host count |
| XFOLD_DIST_PPN | 1 for torchrun/count derivation; MPI placement is forced only when explicitly set |
| XFOLD_DIST_NODE_RANK | 0 on one host; required per host for multi-host torchrun |
| XFOLD_DIST_WORLD_SIZE | MPI explicit total; otherwise NNODES * PPN |
| XFOLD_DIST_HOSTFILE | Optional readable implementation-specific MPI hostfile |
| XFOLD_DIST_MPIRUN | mpirun; choose the intended MPI implementation explicitly |
| XFOLD_DIST_MPI_BINDING | none / inherit; avoids MPI single-core binding by default |
| XFOLD_DIST_TIMEOUT_SECONDS | 3600; longest-phase timeout |
| XFOLD_DIST_MAX_PAYLOAD_MB | 4096 per serialized object |
| XFOLD_DIST_ENV_FILE | Explicit readable shell configuration file |

MASTER_ADDR/MASTER_PORT are the PyTorch rendezvous endpoint. The one-host default
is 127.0.0.1:29500; multi-host launches/hostfiles require MASTER_ADDR. Choose a
free port for each job. Config file assignments override identically named
environment settings. Copy/edit `scripts/distributed_env.sh.example` for the job;
distributed settings use XFOLD_DIST_* rather than forwarded runner flags, and
flagfiles are rejected to prevent overriding launcher-owned choices.

### One host / homogeneous torchrun

```bash
XFOLD_DIST_PPN=8 XFOLD_DIST_MODE=samples XFOLD_CPU_THREADS=4 \
  bash scripts/run_distributed.sh --dry-run \
    --num_diffusion_samples=7 --cpu_precision=bf16 \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./samples_8ranks
```

For multiple homogeneous hosts, start one launcher on every allocated host with
the same NNODES/PPN/MASTER_ADDR/MASTER_PORT and a distinct NODE_RANK. Torchrun
requires homogeneous local worker counts; see its [launch documentation](https://docs.pytorch.org/docs/stable/elastic/run).
The script disables automatic restarts to avoid repeating partially written jobs.
It uses no-python workers so CPU/NUMA environment settings apply before the
worker's model Python process starts. Torchrun itself may default OMP_NUM_THREADS
to 1; explicitly select XFOLD_CPU_THREADS for the intended compute allocation.

### Heterogeneous MPI slots / arbitrary total ranks

An Open MPI hostfile may give different slot counts per host. For example,
two hosts with 4 and 2 slots form six ranks. Choose a reachable master host,
unset XFOLD_DIST_PPN to preserve hostfile placement, and set the explicit total:

```bash
# The hostfile and endpoint must match your real allocation.
MASTER_ADDR=node01 MASTER_PORT=29500 XFOLD_DIST_LAUNCHER=openmpi \
XFOLD_DIST_WORLD_SIZE=6 XFOLD_DIST_HOSTFILE=/shared/hosts.openmpi \
XFOLD_DIST_MODE=tasks XFOLD_DIST_TASK_GRANULARITY=seed XFOLD_CPU_THREADS=8 \
  bash scripts/run_distributed.sh --dry-run \
    --input_dir=/shared/inputs --model_dir=/shared/models --output_dir=/shared/results
```

Open MPI uses --hostfile and its slot syntax; see [MPI host scheduling](https://docs.open-mpi.org/en/main/launching-apps/scheduling.html).
Intel MPI uses -f and optional -ppn; use the [Intel MPI launch syntax](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-15/running-an-mpi-program.html)
and that implementation's hostfile format. Physical host count is not forced to
equal world size. MPI binding defaults to none, leaving stage 8/scheduler CPU
sets in control; choose inherit if the MPI allocation already pins workers.
No oversubscription, root privilege bypass or scheduler restriction bypass is
enabled by the script.

### Existing MPI/Slurm launch

Set XFOLD_DIST_LAUNCHER=external and invoke the wrapper once per rank using the
cluster's own mpirun/srun command, with exported MASTER_ADDR/MASTER_PORT. The
runtime accepts its PMI/Open MPI/Slurm environment. This accommodates scheduler
launch options and node layouts beyond the supplied wrapper; it does not start a
second launcher in external mode. Programs, model/input paths and output location
must be accessible with matching paths across ranks. Only rank 0 writes outputs.

## CPU affinity on each worker

The wrapper reuses `scripts/run_cpu.sh` and all stage 8 XFOLD_CPU_* settings.
Set XFOLD_CPU_ENV_FILE to a shared shell file that assigns worker-local resources
after examining the launcher's rank variables and host. For example, obtain the
local rank with:

```bash
worker_local_rank=${LOCAL_RANK:-${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}}
# Map worker_local_rank/hostname to the actual allocated CPU/node lists here.
```

Do not give all workers on a host the same CPU list unless intentional. Reserve
communication-worker cores when using oneCCL, and match runtime/library versions
before optional preloads. See [stage 8 configuration](stage8_cpu.md).
Rank weights describe the global rank order assigned by the chosen launcher,
not simply hostname order. Inspect actual mapping before selecting capacities.

## Validation boundary and cluster checklist

On this Mac only Python source compilation/AST contracts, shell parsing and
whitespace/reference audits run. No planner runtime tests, process groups,
MPI/CCL/IPEX loading, tensor inference, native builds or timings run here.

```bash
python3 -m unittest discover -s tests -p 'test_*static.py' -v
bash -n scripts/run_distributed.sh
bash -n scripts/distributed_env.sh.example
git diff --check
```

On the cluster, validate in this order:

1. Pure scheduling matrices for 1/2/3/5/6/8+ ranks, zero work, fewer/more jobs,
   nondivisible totals, weighted zero capacities, and invalid configurations.
   Prove every work index occurs once and no index is lost or duplicated.
2. One-rank indexed CPU serial versus sample mode, then 2/6/8+ ranks with
   1/5/7/13 samples and different step counts. Hold precision/backends fixed.
   Check order, masks, numerical error and deterministic streams across schedules.
3. Input and seed task modes versus legacy serial for multiple inputs/seeds,
   idle workers, different capacities and nondivisible totals. Verify global
   ranking order/ties, best output, JSON and sample directories.
4. Gloo first; then matching CPU oneCCL bindings. Test object serialization,
   unequal result sizes, payload budgets, FP32/BF16 byte transport, reused-model
   A→B→A, and optional native operators/IPEX/caches individually.
5. Missing/bad inputs, duplicate names/seeds, invalid flags, rank disagreement,
   compute/serialization/output exceptions, a killed rank and timeout cleanup.
   Confirm no exit-barrier hang or duplicate writer; do not claim recovery.
6. Actual per-host CPU/NUMA placement, communication cores, model replica memory,
   root pending-result memory and fresh-process cold/warm throughput/latency.
   More ranks are a supported configuration, not proof of a speedup.
