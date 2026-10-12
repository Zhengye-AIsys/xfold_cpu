"""Optional native tools. Importing this module neither loads nor builds native code."""

from functools import lru_cache
from importlib import import_module
import platform


__version__ = '0.4.0'
__all__ = [
    'is_available', 'load_extension', 'runtime_info',
    'pad_and_align_tensor', 'vnni_repack_tensor', 'gemm', 'gated_linear_unit',
    'linear', 'triangle_multiplication', 'dot_product_attention',
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


def gated_linear_unit(x, weight, M_BLK=128, N_BLK=128, weight_packed=False):
    """CPU inference GLU: [..., K] @ [K, 2*N], FP32/BF16 matching inputs.

    Returns [..., N] in the input dtype. This raw API does not implement autocast;
    the xfold bridge handles autocast, contiguous copies, and weight caching.
    weight_packed=True requires full-width VNNI2 storage with K rounded up to even.
    """
    if weight_packed:
        return load_extension().gated_linear_unit(x, weight, M_BLK, N_BLK, True)
    return load_extension().gated_linear_unit(x, weight, M_BLK, N_BLK)


def linear(x, weight, bias=None, weight_packed=False):
    """Raw CPU inference Linear, matching FP32/BF16; packed weights are full-width VNNI2."""
    if weight_packed:
        return load_extension().linear(x, weight, bias, True)
    return load_extension().linear(x, weight, bias)


def triangle_multiplication(
    pair, mask, outgoing, left_weight, left_bias, projection, gate,
    center_weight, center_bias, output_weight, gating_weight,
    left_eps=1e-5, center_eps=1e-5, block_size=128, weights_packed=False,
):
    """Raw Triangle update; weights_packed applies to all four Linear weights."""
    arguments = (
        pair, mask, outgoing, left_weight, left_bias, projection, gate,
        center_weight, center_bias, output_weight, gating_weight,
        left_eps, center_eps, block_size,
    )
    if weights_packed:
        return load_extension().triangle_multiplication(*arguments, True)
    return load_extension().triangle_multiplication(*arguments)


def dot_product_attention(q, k, v, mask=None, bias=None, query_block=128, q_pre_scaled=False):
    """Raw CPU inference attention [B, H, Q/K, D], matching FP32/BF16.

    Broadcastable mask/bias, finite -1e9 masking. No implicit autocast or NaN cleanup.
    q_pre_scaled is reserved for the bridge that scales before the autocast cast.
    """
    return load_extension().dot_product_attention(q, k, v, mask, bias, query_block, q_pre_scaled)
