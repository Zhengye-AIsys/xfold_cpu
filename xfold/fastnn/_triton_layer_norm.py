"""CUDA LayerNorm kernels, imported only by the Triton backend."""

import torch
import triton
import triton.language as tl


# modified from triton/python/tutorials/05-layer-norm.py
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    USE_WEIGHTS: tl.constexpr = True,
    USE_BIAS: tl.constexpr = True,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    X += row * N
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if USE_WEIGHTS is True:
            w = tl.load(W + cols, mask=mask)
            if USE_BIAS is True:
                b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        if USE_WEIGHTS is True:
            y = x_hat * w
            if USE_BIAS is True:
                y += b
        else:
            y = x_hat
        # Write output
        tl.store(Y + cols, y, mask=mask)


class LayerNormTritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        x = x.contiguous()
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        N = x.shape[-1]
        M = x.numel() // N
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        use_weight = weight is not None
        use_bias = bias is not None
        _layer_norm_fwd_fused[(M, )](  #
            x, y, weight, bias,    #
            N, eps,   #
            BLOCK_SIZE=BLOCK_SIZE, USE_WEIGHTS=use_weight, USE_BIAS=use_bias, #
            num_warps=num_warps, num_ctas=1)
        return y
