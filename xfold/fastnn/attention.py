from typing import Optional

import torch

from xfold.fastnn import config as fastnn_config


def dot_product_attention_cpp(q, k, v, mask=None, bias=None):
    from .native import check_cpu_inference, compute_dtype, load_native_operator, prepare_tensor

    check_cpu_inference(q, k, v, mask, bias)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or q.size(-1) == 0:
        raise ValueError('C++ attention expects [B, H, Q/K, D] with positive head dimensions.')
    dtype = compute_dtype(q, k, v)
    if bias is not None and bias.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('C++ attention bias must be FP32/BF16.')
    if mask is not None and mask.dim() not in (1, 2) and mask.dtype != torch.bool:
        raise ValueError('Attention masks of rank 0/3/4 must be boolean, as in the PyTorch path.')
    extension = load_native_operator('dot_product_attention')
    with torch.amp.autocast(device_type='cpu', enabled=False):
        # The baseline scales Q before matmul/autocast; preserve that order.
        scaled_q = q.detach() * (q.size(-1) ** -0.5)
        native_mask = prepare_tensor(mask)
        if native_mask is not None and native_mask.dim() in (1, 2):
            native_mask = native_mask.to(dtype=torch.bool)
        return extension.dot_product_attention(
            prepare_tensor(scaled_q, dtype), prepare_tensor(k, dtype),
            prepare_tensor(v, dtype), native_mask, prepare_tensor(bias, torch.float32),
            q_pre_scaled=True,
        )


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
    elif fastnn_config.dot_product_attention_implementation == "cpp":
        out = dot_product_attention_cpp(q, k, v, mask, bias)
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
