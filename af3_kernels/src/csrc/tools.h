// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <torch/extension.h>
#include <libxsmm.h>

namespace af3_detail {
void check_tensor(const at::Tensor& tensor);
libxsmm_blasint xsmm_size(int64_t size);
at::Tensor aligned_empty(const at::Tensor& like, at::IntArrayRef shape, int64_t alignment);
}

at::Tensor pad_and_align_tensor(const at::Tensor& tensor, at::IntArrayRef shape,
                               bool need_init, double init, int64_t alignment);
at::Tensor vnni_repack_tensor(const at::Tensor& tensor, int64_t type, int64_t block_size);
at::Tensor xsmm_gemm(const at::Tensor& a, const at::Tensor& b);
at::Tensor xsmm_gated_linear_unit(const at::Tensor& x, const at::Tensor& weight,
                                 int64_t m_block, int64_t n_block);
