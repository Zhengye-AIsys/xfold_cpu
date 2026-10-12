"""Explicit process-wide CPU thread settings, applied before model construction."""

import os
import warnings

import torch


def configure_cpu_threads(num_threads=None, num_interop_threads=None):
    """None preserves runtime defaults; a used inter-op pool cannot be resized."""
    for name, value in (('cpu_threads', num_threads), ('cpu_interop_threads', num_interop_threads)):
        if value is not None and (type(value) is not int or not 0 < value <= 2147483647):
            raise ValueError(f'{name} must be a positive 32-bit integer or None.')
    # Skip an already matching setting so repeated runners can share one process.
    # A differing, previously initialized/set pool must fail instead of silently
    # ignoring the request. Set inter-op first to avoid partial intra-op changes.
    if num_interop_threads is not None and num_interop_threads != torch.get_num_interop_threads():
        try:
            torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                'Cannot change CPU inter-op threads in this process. Start a '
                'fresh process and set --cpu_interop_threads before inference.'
            ) from error
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            value = os.environ.get(name)
            if value is not None and value != str(num_threads):
                warnings.warn(
                    f'{name}={value} differs from --cpu_threads={num_threads}. '
                    'Use scripts/run_cpu.sh with XFOLD_CPU_THREADS to configure '
                    'PyTorch and OpenMP before Python starts.', RuntimeWarning,
                )
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    if affinity is not None and torch.get_num_threads() > len(affinity):
        warnings.warn('PyTorch CPU threads exceed the allowed CPU affinity count.', RuntimeWarning)
    print(f'CPU threads: intra-op={torch.get_num_threads()}, '
          f'inter-op={torch.get_num_interop_threads()}; allowed CPUs={affinity}')
    print('CPU environment: ' + ', '.join(
        f'{name}={os.environ.get(name, "<unset>")}' for name in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_PLACES', 'OMP_PROC_BIND',
         'KMP_AFFINITY', 'KMP_BLOCKTIME', 'GOMP_CPU_AFFINITY')))
