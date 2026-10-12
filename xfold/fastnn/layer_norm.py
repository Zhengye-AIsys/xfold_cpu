import numbers
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Size

from xfold.fastnn import config as fastnn_config

_shape_t = Union[int, List[int], Size]


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(
            normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if fastnn_config.layer_norm_implementation == "torch":
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif fastnn_config.layer_norm_implementation == "ipex":
            if self.training:
                raise RuntimeError('IPEX LayerNorm supports eval inference only.')
            from ._ipex_layer_norm import layer_norm_ipex

            return layer_norm_ipex(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif fastnn_config.layer_norm_implementation == "triton":
            if input.device.type != 'cuda':
                raise ValueError('Triton LayerNorm requires CUDA tensors.')
            from ._triton_layer_norm import LayerNormTritonFunc

            fast_out = LayerNormTritonFunc.apply(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
            return fast_out
        else:
            raise ValueError(
                "layer_norm_implementation must be 'torch', 'triton' or 'ipex', "
                f"got {fastnn_config.layer_norm_implementation}"
            )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
