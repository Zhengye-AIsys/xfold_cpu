from typing import Optional

import torch

from xfold.fastnn import config as fastnn_config


def dot_product_attention_triton(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 bias: Optional[torch.Tensor] = None):
    if q.device.type != 'cuda':
        raise ValueError('Triton attention requires CUDA tensors.')
    from ._triton_attention import dot_product_attention_triton as implementation

    return implementation(q, k, v, mask, bias)


def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2))

    if bias is not None:
        logits += bias

    if mask is not None:
        if mask.dim() == 1:
            mask = mask[None, None, None, :].to(dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask[:, None, None, :].to(dtype=torch.bool)
        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)


def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None):
    # if q,k,v is 3-dimensional, add a batch dimension
    qkv_dims = q.dim()
    if qkv_dims == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    if fastnn_config.dot_product_attention_implementation == "torch":
        out = dot_product_attention_torch(q, k, v, mask, bias)
    elif fastnn_config.dot_product_attention_implementation == "triton":
        if k.shape[-1] not in {16, 32, 64, 128}:
            out = dot_product_attention_torch(q, k, v, mask, bias)
        else:
            out = dot_product_attention_triton(q, k, v, mask, bias)
    else:
        raise ValueError(
            f"Unknown dot_product_attention_implementation: {fastnn_config.dot_product_attention_implementation}"
        )

    if qkv_dims == 3:
        out = out.squeeze(0)

    return out
