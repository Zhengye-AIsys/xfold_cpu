"""Shared inference bridges; optional native code loads only on explicit use."""

import torch


def load_native_operator(name):
    try:
        from af3_kernels import load_extension
    except ImportError as error:
        raise RuntimeError(
            'This CPU backend requires ./af3_kernels, built on Linux x86_64 '
            'in the target PyTorch environment with --no-build-isolation --no-deps.'
        ) from error
    extension = load_extension()
    if not hasattr(extension, name):
        raise RuntimeError(f'Rebuild af3_kernels: this binary does not contain {name}.')
    return extension


def check_cpu_inference(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
            raise ValueError('Native operators require strided CPU tensors.')
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError('Native operators support inference only; disable gradients.')


def compute_dtype(x, *weights):
    for tensor in (x, *weights):
        if tensor.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('Native computation supports FP32/BF16 only.')
    if torch.is_autocast_enabled('cpu'):
        dtype = torch.get_autocast_dtype('cpu')
    else:
        dtype = x.dtype
        if any(weight.dtype != dtype for weight in weights):
            raise ValueError('Input/weight dtypes must match outside CPU autocast.')
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Native operators support FP32/BF16 CPU autocast only.')
    return dtype


def prepare_tensor(tensor, dtype=None):
    if tensor is None:
        return None
    tensor = tensor.detach().resolve_conj().resolve_neg()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()


def linear_cpp(x, layer, weight_cache):
    """Keep nn.Linear parameters/checkpoints; round bias and output like autocast."""
    check_cpu_inference(x, layer.weight, layer.bias)
    dtype = compute_dtype(x, layer.weight, *(() if layer.bias is None else (layer.bias,)))
    extension = load_native_operator('linear')
    with torch.amp.autocast(device_type='cpu', enabled=False):
        weight = layer.weight.T.resolve_conj().resolve_neg()
        return extension.linear(
            prepare_tensor(x, dtype), weight_cache.prepare(weight, dtype),
            prepare_tensor(layer.bias, dtype),
        )
