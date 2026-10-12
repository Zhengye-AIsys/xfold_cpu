import torch

from xfold.fastnn import config as fastnn_config


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


def gated_linear_unit(x, weight):
    if fastnn_config.gated_linear_unit_implementation == "torch":
        out = gated_linear_unit_torch(x, weight)
    elif fastnn_config.gated_linear_unit_implementation == "triton":
        out = gated_linear_unit_triton(x, weight)
    else:
        raise ValueError(
            "Unknown gated_linear_unit_implementation: "
            f"{fastnn_config.gated_linear_unit_implementation}"
        )
    return out
