"""Optional IPEX LayerNorm bridge; importing this file never imports IPEX."""

from functools import lru_cache
from importlib import import_module
import platform

import torch


@lru_cache(maxsize=1)
def load_ipex_layer_norm():
    """Check the explicitly selected CPU dependency before model construction."""
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('IPEX LayerNorm requires the target Linux x86_64 environment.')
    try:
        ipex = import_module('intel_extension_for_pytorch')
    except (ImportError, OSError, RuntimeError, SystemExit) as error:
        raise RuntimeError(
            'IPEX LayerNorm is unavailable. Install a CPU IPEX release matching '
            'the target PyTorch environment, or select --layer_norm_backend=torch.'
        ) from error
    functional = getattr(getattr(ipex, 'llm', None), 'functional', None)
    operation = getattr(functional, 'fast_layer_norm', None)
    if not callable(operation):
        raise RuntimeError(
            'IPEX lacks llm.functional.fast_layer_norm; use a compatible CPU '
            'IPEX release or select --layer_norm_backend=torch.'
        )
    print(f'IPEX LayerNorm: IPEX={getattr(ipex, "__version__", "unknown")}, '
          f'PyTorch={torch.__version__}; llm.functional.fast_layer_norm')
    return operation


def layer_norm_ipex(input, normalized_shape, weight, bias, eps):
    """Forward existing parameters/precision to the public IPEX functional API."""
    tensors = (input,) + tuple(value for value in (weight, bias) if value is not None)
    for tensor in tensors:
        if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
            raise ValueError('IPEX LayerNorm requires strided CPU tensors.')
        if tensor.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('IPEX LayerNorm supports FP32/BF16 inference only.')
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError('IPEX LayerNorm supports inference only; disable gradients.')
    # Preserve the CPU autocast context and optional affine parameters. In
    # particular, BF16 activations may have FP32 stored parameters, as in PyTorch.
    operation = load_ipex_layer_norm()
    return operation(input, normalized_shape, weight, bias, eps)
