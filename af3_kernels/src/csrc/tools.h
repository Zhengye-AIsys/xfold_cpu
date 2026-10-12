// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <torch/extension.h>

at::Tensor pad_and_align_tensor(const at::Tensor& tensor, at::IntArrayRef shape,
                               bool need_init, double init, int64_t alignment);
at::Tensor vnni_repack_tensor(const at::Tensor& tensor, int64_t type, int64_t block_size);
at::Tensor xsmm_gemm(const at::Tensor& a, const at::Tensor& b);
