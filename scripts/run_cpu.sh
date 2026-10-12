#!/usr/bin/env bash
# Single-process CPU inference; environment/binding is applied before Python.
set -euo pipefail

fail() { printf 'CPU launcher: %s\n' "$*" >&2; exit 2; }
positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer."
    [[ ${#2} -le 10 ]] || fail "$1 exceeds the 32-bit thread count limit."
    (( 10#$2 <= 2147483647 )) || fail "$1 exceeds the 32-bit thread count limit."
}
index_list() {
    [[ "$2" =~ ^\+?[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] ||
        fail "$1 must be a CPU/node list, for example 0-7,16-23 or +0-7."
}

dry_run=0
if [[ "${1:-}" == '--dry-run' ]]; then
    dry_run=1
    shift
fi
if [[ -n "${XFOLD_CPU_ENV_FILE:-}" ]]; then
    [[ -r "$XFOLD_CPU_ENV_FILE" ]] || fail 'XFOLD_CPU_ENV_FILE is not readable.'
    source "$XFOLD_CPU_ENV_FILE"
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_binary=${XFOLD_CPU_PYTHON:-python}
threads=${XFOLD_CPU_THREADS:-}
interop_threads=${XFOLD_CPU_INTEROP_THREADS:-}
cpus=${XFOLD_CPU_CPUS:-}
cpu_nodes=${XFOLD_CPU_NODES:-}
memory_nodes=${XFOLD_CPU_MEM_NODES:-}
memory_policy=${XFOLD_CPU_MEM_POLICY:-inherit}
affinity=${XFOLD_CPU_AFFINITY:-inherit}
runner_flags=()
binding=()

# These launcher-owned flags have one source of truth, avoiding last-flag-wins
# device/thread overrides while OpenMP is configured for another value.
for argument in "$@"; do
    # Abseil accepts one or two leading dashes and flag=value / flag value.
    flag_name=${argument%%=*}
    case "$flag_name" in
        --device|-device|--cpu_threads|-cpu_threads|--cpu_interop_threads|-cpu_interop_threads)
            fail 'Device is CPU; set thread counts with XFOLD_CPU_THREADS / XFOLD_CPU_INTEROP_THREADS.' ;;
        --flagfile|-flagfile)
            fail 'Flagfiles can override launcher-owned settings; pass runner flags directly.' ;;
    esac
done
if [[ -n "$threads" ]]; then
    positive_integer XFOLD_CPU_THREADS "$threads"
    export OMP_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    runner_flags+=("--cpu_threads=$threads")
fi
if [[ -n "$interop_threads" ]]; then
    positive_integer XFOLD_CPU_INTEROP_THREADS "$interop_threads"
    runner_flags+=("--cpu_interop_threads=$interop_threads")
fi

case "$affinity" in
    inherit) ;;
    compact|spread|none)
        [[ -z "${KMP_AFFINITY:-}" && -z "${GOMP_CPU_AFFINITY:-}" ]] ||
            fail 'Unset KMP_AFFINITY/GOMP_CPU_AFFINITY before selecting XFOLD_CPU_AFFINITY.'
        if [[ "$affinity" == none ]]; then
            export OMP_PROC_BIND=false
        else
            export OMP_PLACES=cores
            if [[ "$affinity" == compact ]]; then
                export OMP_PROC_BIND=close
            else
                export OMP_PROC_BIND=spread
            fi
        fi ;;
    *) fail 'XFOLD_CPU_AFFINITY must be inherit, compact, spread, or none.' ;;
esac
if [[ -n "${XFOLD_CPU_KMP_BLOCKTIME:-}" ]]; then
    [[ "$XFOLD_CPU_KMP_BLOCKTIME" =~ ^[0-9]+$ ]] || fail 'XFOLD_CPU_KMP_BLOCKTIME must be nonnegative.'
    export KMP_BLOCKTIME="$XFOLD_CPU_KMP_BLOCKTIME"
fi

[[ -z "$cpus" || -z "$cpu_nodes" ]] || fail 'Select XFOLD_CPU_CPUS or XFOLD_CPU_NODES, not both.'
if [[ -n "$cpus" ]]; then
    index_list XFOLD_CPU_CPUS "$cpus"
    binding+=("--physcpubind=$cpus")
fi
if [[ -n "$cpu_nodes" ]]; then
    index_list XFOLD_CPU_NODES "$cpu_nodes"
    binding+=("--cpunodebind=$cpu_nodes")
fi
case "$memory_policy" in
    inherit)
        [[ -z "$memory_nodes" ]] || fail 'Memory nodes require bind or interleave policy.' ;;
    bind|interleave)
        [[ -n "$memory_nodes" ]] || fail 'Set XFOLD_CPU_MEM_NODES for bind/interleave.'
        index_list XFOLD_CPU_MEM_NODES "$memory_nodes"
        if [[ "$memory_policy" == bind ]]; then
            binding+=("--membind=$memory_nodes")
        else
            binding+=("--interleave=$memory_nodes")
        fi ;;
    local)
        [[ -z "$memory_nodes" ]] || fail 'Local memory policy does not accept a node list.'
        binding+=(--localalloc) ;;
    *) fail 'XFOLD_CPU_MEM_POLICY must be inherit, bind, interleave, or local.' ;;
esac

# Shared libraries are opt-in absolute paths; no CONDA paths or automatic
# replacement of the OpenMP runtime. Existing preloads remain visible/preserved.
for library in "${XFOLD_CPU_OMP_LIBRARY:-}" "${XFOLD_CPU_ALLOCATOR_LIBRARY:-}"; do
    if [[ -n "$library" ]]; then
        [[ "$library" == /* && -f "$library" && -r "$library" ]] || fail 'Preload library must be a readable absolute file path.'
        [[ "$library" != *:* && "$library" != *[[:space:]]* ]] || fail 'Preload paths cannot contain colons or whitespace.'
        export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$library"
    fi
done

command=("$python_binary" "$repo_dir/run_alphafold.py" --device=cpu --run_data_pipeline=False
         ${runner_flags[@]+"${runner_flags[@]}"} "$@")
if [[ -n "${binding[*]-}" ]]; then
    command=(numactl "${binding[@]}" -- "${command[@]}")
fi
for name in OMP_NUM_THREADS MKL_NUM_THREADS OMP_PLACES OMP_PROC_BIND KMP_AFFINITY KMP_BLOCKTIME GOMP_CPU_AFFINITY LD_PRELOAD; do
    printf '%s=%q\n' "$name" "${!name-<unset>}"
done
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$dry_run" == 1 ]]; then
    exit 0
fi
[[ "$(uname -s)" == Linux ]] || fail 'This cluster launcher requires Linux; --dry-run only prints configuration.'
command -v "$python_binary" >/dev/null || fail 'XFOLD_CPU_PYTHON executable was not found.'
if [[ -n "${binding[*]-}" ]]; then
    command -v numactl >/dev/null || fail 'numactl is required for the requested binding.'
fi
exec "${command[@]}"
