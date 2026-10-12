"""Optional native tools. Importing this module neither loads nor builds native code."""

from functools import lru_cache
from importlib import import_module
import platform


__version__ = '0.2.0'
__all__ = [
    'is_available', 'load_extension', 'runtime_info',
    'pad_and_align_tensor', 'vnni_repack_tensor', 'gemm', 'gated_linear_unit',
]


@lru_cache(maxsize=1)
def load_extension():
    """Explicitly load a prebuilt extension; never compile or install at runtime."""
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('af3_kernels native tools require Linux x86_64.')
    try:
        import_module('torch')  # Load PyTorch libraries before loading the extension.
        return import_module('af3_kernels._C')
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'The optional af3_kernels extension is unavailable. Build it in the '
            'target PyTorch environment with --no-build-isolation --no-deps. '
            'Ordinary xfold FP32/BF16 inference does not require this extension.'
        ) from error


def is_available():
    """Probe native loading on explicit request; does not claim AMX execution."""
    try:
        load_extension()
    except RuntimeError:
        return False
    return True


def runtime_info():
    return load_extension().runtime_info()


def pad_and_align_tensor(x, shape, need_init=True, init=0.0, alignment=64):
    """Pad CPU tensors without shrinking; an already aligned tensor may be returned as-is."""
    return load_extension().pad_and_align_tensor(x, tuple(shape), need_init, init, alignment)


def vnni_repack_tensor(x, type=2, BLK_SZ=0):
    """Pack contiguous BF16 weights in VNNI2 layout; this is an opaque storage layout."""
    return load_extension().vnni_repack_tensor(x, type, BLK_SZ)


def gemm(a, b):
    """Infrastructure probe: CPU 2D FP32/BF16 inputs, FP32 output; inference only."""
    return load_extension().gemm(a, b)


def gated_linear_unit(x, weight, M_BLK=128, N_BLK=128):
    """CPU inference GLU: [..., K] @ [K, 2*N], FP32/BF16 matching inputs.

    Returns [..., N] in the input dtype. This raw API does not implement autocast;
    the xfold bridge handles autocast, contiguous copies, and weight caching.
    """
    return load_extension().gated_linear_unit(x, weight, M_BLK, N_BLK)
