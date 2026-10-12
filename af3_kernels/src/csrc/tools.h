// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <torch/extension.h>
#include <libxsmm.h>

namespace af3_detail {
void check_tensor(const at::Tensor& tensor);
libxsmm_blasint xsmm_size(int64_t size);
at::Tensor aligned_empty(const at::Tensor& like, at::IntArrayRef shape, int64_t alignment);
void check_auxiliary(const at::Tensor& tensor);
at::Tensor tiled_matmul(const at::Tensor& a, const at::Tensor& b,
                       const c10::optional<at::Tensor>& bias = c10::nullopt,
                       int64_t m_block = 128, int64_t n_block = 128);
}

at::Tensor pad_and_align_tensor(const at::Tensor& tensor, at::IntArrayRef shape,
                               bool need_init, double init, int64_t alignment);
at::Tensor vnni_repack_tensor(const at::Tensor& tensor, int64_t type, int64_t block_size);
at::Tensor xsmm_gemm(const at::Tensor& a, const at::Tensor& b);
at::Tensor xsmm_gated_linear_unit(const at::Tensor& x, const at::Tensor& weight,
                                 int64_t m_block, int64_t n_block);
at::Tensor xsmm_linear(const at::Tensor& x, const at::Tensor& weight,
                       const c10::optional<at::Tensor>& bias);
at::Tensor xsmm_triangle_multiplication(
    const at::Tensor& pair, const c10::optional<at::Tensor>& mask, bool outgoing,
    const at::Tensor& left_weight, const c10::optional<at::Tensor>& left_bias,
    const at::Tensor& projection, const at::Tensor& gate,
    const at::Tensor& center_weight, const c10::optional<at::Tensor>& center_bias,
    const at::Tensor& output_weight, const at::Tensor& gating_weight,
    double left_eps, double center_eps, int64_t block_size);
at::Tensor xsmm_dot_product_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& bias,
    int64_t query_block, bool q_pre_scaled);
