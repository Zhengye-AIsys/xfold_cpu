import torch

from xfold.fastnn import config as fastnn_config


class GLUWeightCache:
    """One inference-only contiguous weight copy per Transition, outside state_dict."""

    def __init__(self):
        self._key = None
        self._source = None
        self._weight = None

    def prepare(self, weight, dtype):
        try:
            version = weight._version
        except RuntimeError:
            # Inference tensors have no mutation counter; do not reuse their copies.
            self._key = self._source = self._weight = None
            return weight.detach().to(dtype=dtype).contiguous()
        key = (weight.data_ptr(), version, tuple(weight.shape), tuple(weight.stride()),
               weight.dtype, weight.device, dtype)
        if self._key != key:
            self._weight = weight.detach().to(dtype=dtype).contiguous()
            # Retain the source so allocator address reuse cannot produce a cache hit.
            self._source = weight
            self._key = key
        return self._weight


def load_cpp_glu():
    """Load only on explicit selection; detect an older stage 3 binary early."""
    try:
        from af3_kernels import load_extension
    except ImportError as error:
        raise RuntimeError(
            'C++ GLU requires the optional af3_kernels package. Build ./af3_kernels '
            'on Linux x86_64 with --no-build-isolation --no-deps.'
        ) from error
    extension = load_extension()
    if not hasattr(extension, 'gated_linear_unit'):
        raise RuntimeError('Rebuild af3_kernels: this binary does not contain the stage 4 GLU.')
    return extension


def gated_linear_unit_cpp(x, weight, weight_cache=None):
    """CPU inference bridge honoring autocast without changing stored weights."""
    if x.device.type != 'cpu' or weight.device.type != 'cpu':
        raise ValueError('C++ GLU requires CPU tensors.')
    if x.dtype not in (torch.float32, torch.bfloat16) or weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('C++ GLU supports FP32 and BF16 only.')
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError('C++ GLU supports inference only; use torch.no_grad() or torch.inference_mode().')
    dtype = torch.get_autocast_dtype('cpu') if torch.is_autocast_enabled('cpu') else x.dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('C++ GLU supports FP32/BF16 CPU autocast only.')
    if not torch.is_autocast_enabled('cpu') and weight.dtype != x.dtype:
        raise ValueError('C++ GLU input and weight dtypes must match outside CPU autocast.')
    extension = load_cpp_glu()
    with torch.amp.autocast(device_type='cpu', enabled=False):
        native_x = x.detach().resolve_conj().resolve_neg().to(dtype=dtype).contiguous()
        weight = weight.resolve_conj().resolve_neg()
        native_weight = (
            weight_cache.prepare(weight, dtype) if weight_cache is not None
            else weight.detach().to(dtype=dtype).contiguous()
        )
        return extension.gated_linear_unit(native_x, native_weight)


def gated_linear_unit_triton(x, weight):
    if x.device.type != 'cuda':
        raise ValueError('Triton GLU requires CUDA tensors.')
    from ._triton_gated_linear_unit import gated_linear_unit_triton as implementation

    return implementation(x, weight)


def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out


def gated_linear_unit(x, weight, weight_cache=None):
    if fastnn_config.gated_linear_unit_implementation == "torch":
        out = gated_linear_unit_torch(x, weight)
    elif fastnn_config.gated_linear_unit_implementation == "triton":
        out = gated_linear_unit_triton(x, weight)
    elif fastnn_config.gated_linear_unit_implementation == "cpp":
        out = gated_linear_unit_cpp(x, weight, weight_cache)
    else:
        raise ValueError(
            "Unknown gated_linear_unit_implementation: "
            f"{fastnn_config.gated_linear_unit_implementation}"
        )
    return out
