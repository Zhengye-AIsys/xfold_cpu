#!/usr/bin/env bash
# Fixed-size CPU process groups. Physical hosts, ranks and work are independent.
set -euo pipefail
fail() { printf 'Distributed launcher: %s\n' "$*" >&2; exit 2; }
positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ && ${#2} -le 10 ]] || fail "$1 must be a positive 32-bit integer."
    (( 10#$2 <= 2147483647 )) || fail "$1 exceeds the 32-bit count limit."
}

dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if [[ -n "${XFOLD_DIST_ENV_FILE:-}" ]]; then
    [[ -r "$XFOLD_DIST_ENV_FILE" ]] || fail 'XFOLD_DIST_ENV_FILE is not readable.'
    source "$XFOLD_DIST_ENV_FILE"
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
launcher=${XFOLD_DIST_LAUNCHER:-torchrun}
mode=${XFOLD_DIST_MODE:-samples}
backend=${XFOLD_DIST_BACKEND:-gloo}
schedule=${XFOLD_DIST_SCHEDULE:-cyclic}
granularity=${XFOLD_DIST_TASK_GRANULARITY:-auto}
nnodes=${XFOLD_DIST_NNODES:-1}
ppn=${XFOLD_DIST_PPN:-1}
timeout=${XFOLD_DIST_TIMEOUT_SECONDS:-3600}
budget=${XFOLD_DIST_MAX_PAYLOAD_MB:-4096}
weights=${XFOLD_DIST_RANK_WEIGHTS:-}
hostfile=${XFOLD_DIST_HOSTFILE:-}
mpi_binding=${XFOLD_DIST_MPI_BINDING:-none}
case "$mode" in samples|tasks) ;; *) fail 'XFOLD_DIST_MODE must be samples or tasks.' ;; esac
case "$backend" in gloo|ccl) ;; *) fail 'XFOLD_DIST_BACKEND must be gloo or ccl.' ;; esac
case "$schedule" in cyclic|block|weighted) ;; *) fail 'Invalid work schedule.' ;; esac
case "$granularity" in auto|input|seed) ;; *) fail 'Invalid task granularity.' ;; esac
case "$mpi_binding" in none|inherit) ;; *) fail 'MPI binding must be none or inherit.' ;; esac
positive_integer XFOLD_DIST_NNODES "$nnodes"
positive_integer XFOLD_DIST_PPN "$ppn"
positive_integer XFOLD_DIST_TIMEOUT_SECONDS "$timeout"
positive_integer XFOLD_DIST_MAX_PAYLOAD_MB "$budget"
if [[ "$schedule" == weighted ]]; then
    [[ "$weights" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'Weighted scheduling needs comma-separated rank weights.'
else
    [[ -z "$weights" ]] || fail 'Rank weights require weighted scheduling.'
fi
for argument in "$@"; do
    flag_name=${argument%%=*}
    case "$flag_name" in
        --distributed_*|-distributed_*|--flagfile|-flagfile)
            fail 'Set distributed settings through XFOLD_DIST_*; pass other runner flags directly.' ;;
    esac
done

worker=(bash "$repo_dir/scripts/run_cpu.sh"
    "--distributed_mode=$mode" "--distributed_backend=$backend"
    "--distributed_schedule=$schedule" "--distributed_rank_weights=$weights"
    "--distributed_task_granularity=$granularity"
    "--distributed_timeout_seconds=$timeout" "--distributed_max_payload_mb=$budget" "$@")
mpi_exports=()
# Worker-side CPU configuration can use local-rank/hostname to select distinct
# CPU sets and thread counts. Do not source that file on the launcher host.
for name in XFOLD_CPU_ENV_FILE XFOLD_CPU_PYTHON XFOLD_CPU_THREADS XFOLD_CPU_INTEROP_THREADS \
    XFOLD_CPU_CPUS XFOLD_CPU_NODES XFOLD_CPU_MEM_POLICY XFOLD_CPU_MEM_NODES XFOLD_CPU_AFFINITY \
    XFOLD_CPU_KMP_BLOCKTIME XFOLD_CPU_OMP_LIBRARY XFOLD_CPU_ALLOCATOR_LIBRARY \
    OMP_NUM_THREADS MKL_NUM_THREADS OMP_PLACES OMP_PROC_BIND KMP_AFFINITY KMP_BLOCKTIME \
    GOMP_CPU_AFFINITY LD_PRELOAD GLOO_SOCKET_IFNAME CCL_ATL_TRANSPORT CCL_WORKER_COUNT \
    CCL_WORKER_AFFINITY CCL_LOG_LEVEL;
do
    if [[ -n "${!name+x}" ]]; then
        export "$name"
        mpi_exports+=(-x "$name")
    fi
done

case "$launcher" in
    torchrun)
        [[ -z "$hostfile" && -z "${XFOLD_DIST_WORLD_SIZE:-}" ]] || fail 'torchrun uses NNODES/PPN, not hostfile/world size.'
        if (( nnodes > 1 )); then
            [[ -n "${XFOLD_DIST_NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]] || fail 'Multi-node torchrun needs NODE_RANK and MASTER_ADDR.'
        fi
        node_rank=${XFOLD_DIST_NODE_RANK:-0}
        [[ "$node_rank" =~ ^[0-9]+$ && ${#node_rank} -le 10 ]] || fail 'Invalid node rank.'
        (( 10#$node_rank < nnodes )) || fail 'Node rank must be less than NNODES.'
        export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
        export MASTER_PORT=${MASTER_PORT:-29500}
        command=("${XFOLD_CPU_PYTHON:-python}" -m torch.distributed.run
            "--nnodes=$nnodes" "--nproc-per-node=$ppn" "--node-rank=$node_rank"
            "--master-addr=$MASTER_ADDR" "--master-port=$MASTER_PORT"
            --max-restarts=0 --no-python "${worker[@]}") ;;
    openmpi|intelmpi)
        world_size=${XFOLD_DIST_WORLD_SIZE:-$((nnodes * ppn))}
        positive_integer XFOLD_DIST_WORLD_SIZE "$world_size"
        if [[ -n "$hostfile" ]]; then
            [[ -r "$hostfile" ]] || fail 'Hostfile must be readable.'
        fi
        if (( nnodes > 1 )) || [[ -n "$hostfile" ]]; then
            [[ -n "${MASTER_ADDR:-}" ]] || fail 'Multi-host MPI requires an explicit MASTER_ADDR.'
        fi
        export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
        export MASTER_PORT=${MASTER_PORT:-29500}
        command=("${XFOLD_DIST_MPIRUN:-mpirun}" -n "$world_size")
        if [[ "$launcher" == openmpi ]]; then
            command+=(-x MASTER_ADDR -x MASTER_PORT ${mpi_exports[@]+"${mpi_exports[@]}"})
            if [[ "$mpi_binding" == none ]]; then command+=(--bind-to none); fi
            if [[ -n "${XFOLD_DIST_PPN:-}" ]]; then command+=(--map-by "ppr:$ppn:node"); fi
            if [[ -n "$hostfile" ]]; then command+=(--hostfile "$hostfile"); fi
        else
            if [[ "$mpi_binding" == none ]]; then command+=(-genv I_MPI_PIN 0); fi
            if [[ -n "${XFOLD_DIST_PPN:-}" ]]; then command+=(-ppn "$ppn"); fi
            if [[ -n "$hostfile" ]]; then command+=(-f "$hostfile"); fi
        fi
        command+=("${worker[@]}") ;;
    external)
        # Execute once per rank under an existing MPI/Slurm allocation.
        [[ -n "${MASTER_ADDR:-}" && -n "${MASTER_PORT:-}" ]] || fail 'External launch needs MASTER_ADDR/MASTER_PORT.'
        command=("${worker[@]}") ;;
    *) fail 'Launcher must be torchrun, openmpi, intelmpi, or external.' ;;
esac
positive_integer MASTER_PORT "$MASTER_PORT"
(( 10#$MASTER_PORT <= 65535 )) || fail 'MASTER_PORT must be within 1..65535.'
export MASTER_ADDR MASTER_PORT
printf 'Launcher=%s mode=%s backend=%s schedule=%s\n' "$launcher" "$mode" "$backend" "$schedule"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$dry_run" == 1 ]]; then exit 0; fi
[[ "$(uname -s)" == Linux ]] || fail 'Distributed CPU launcher requires Linux.'
command -v "${command[0]}" >/dev/null || fail 'Requested launcher executable was not found.'
exec "${command[@]}"
