// SPDX-License-Identifier: BSD-3-Clause
#include "tools.h"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace {

template <typename T>
void run_glu(const at::Tensor& input, const at::Tensor& right, at::Tensor& output,
             int64_t m, libxsmm_blasint k, libxsmm_blasint n,
             libxsmm_blasint mb, libxsmm_blasint nb, libxsmm_blasint weight_stride,
             libxsmm_datatype dtype, libxsmm_bitfield flags) {
    const int64_t m_tiles = (m - 1) / mb + 1;
    const int64_t n_tiles = (static_cast<int64_t>(n) - 1) / nb + 1;
    const libxsmm_blasint heights[2] = {mb, static_cast<libxsmm_blasint>((m - 1) % mb + 1)};
    const libxsmm_blasint widths[2] = {nb, static_cast<libxsmm_blasint>((n - 1) % nb + 1)};
    libxsmm_gemmfunction kernels[2][2];
    // Dispatch before OpenMP so unsupported shapes throw on the calling thread.
    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
            // Row-major projection is C^T = W^T X^T. BF16 W^T uses VNNI2.
            const auto shape = libxsmm_create_gemm_shape(
                widths[w], heights[h], k, weight_stride, k, nb,
                dtype, dtype, dtype, LIBXSMM_DATATYPE_F32);
            kernels[h][w] = libxsmm_dispatch_gemm(shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
            TORCH_CHECK(kernels[h][w] != nullptr,
                        "LIBXSMM could not generate the GLU projection for this CPU/shape.");
        }
    }
    const int threads = std::max(1, omp_get_max_threads());
    auto scratch = af3_detail::aligned_empty(input, {threads, 2, mb, nb}, 64);
    const int64_t tile_elements = static_cast<int64_t>(mb) * nb;
    const auto* x = input.data_ptr<T>();
    const auto* weight = right.data_ptr<T>();
    auto* out = output.data_ptr<T>();
    auto* temporary = scratch.data_ptr<T>();
    const int64_t pack = dtype == LIBXSMM_DATATYPE_BF16 ? 2 : 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t mi = 0; mi < m_tiles; ++mi) {
        for (int64_t ni = 0; ni < n_tiles; ++ni) {
            const int hi = mi == m_tiles - 1 ? 1 : 0;
            const int wi = ni == n_tiles - 1 ? 1 : 0;
            const int64_t row = mi * mb;
            const int64_t col = ni * nb;
            auto* gate = temporary + static_cast<int64_t>(omp_get_thread_num()) * 2 * tile_elements;
            auto* value = gate + tile_elements;
            libxsmm_gemm_param params = {};
            params.a.primary = const_cast<T*>(weight + col * pack);
            params.b.primary = const_cast<T*>(x + row * k);
            params.c.primary = gate;
            kernels[hi][wi](&params);
            params.a.primary = const_cast<T*>(weight + (n + col) * pack);
            params.c.primary = value;
            kernels[hi][wi](&params);

            // Match PyTorch's projection -> SiLU -> multiply dtype boundaries.
            // BF16 GEMMs accumulate in FP32 and round when storing gate/value.
            for (int64_t r = 0; r < heights[hi]; ++r) {
                for (int64_t c = 0; c < widths[wi]; ++c) {
                    const int64_t index = r * nb + c;
                    const float a = static_cast<float>(gate[index]);
                    const float sigmoid = a >= 0.0f ? 1.0f / (1.0f + std::exp(-a))
                                                    : std::exp(a) / (1.0f + std::exp(a));
                    const T activated = T(a * sigmoid);
                    out[(row + r) * n + col + c] =
                        T(static_cast<float>(activated) * static_cast<float>(value[index]));
                }
            }
        }
    }
}

}  // namespace

at::Tensor xsmm_gated_linear_unit(const at::Tensor& x, const at::Tensor& weight,
                                 int64_t m_block, int64_t n_block) {
    af3_detail::check_tensor(x);
    af3_detail::check_tensor(weight);
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 2, "GLU expects [..., K] and [K, 2*N].");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "GLU inputs must be contiguous.");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "GLU input dtypes must match.");
    TORCH_CHECK(x.size(-1) == weight.size(0), "GLU inner dimensions must match.");
    const auto k = af3_detail::xsmm_size(weight.size(0));
    const auto weight_stride = af3_detail::xsmm_size(weight.size(1));
    TORCH_CHECK(weight_stride % 2 == 0, "GLU weight width must be positive and even.");
    const auto n = weight_stride / 2;
    const auto requested_mb = af3_detail::xsmm_size(m_block);
    const auto requested_nb = af3_detail::xsmm_size(n_block);
    const int64_t m = x.numel() / k;
    auto shape = x.sizes().vec();
    shape.back() = n;
    auto output = af3_detail::aligned_empty(x, shape, 64);
    if (m == 0) return output;
    const auto mb = static_cast<libxsmm_blasint>(std::min<int64_t>(requested_mb, m));
    const auto nb = std::min(requested_nb, n);
    auto input = x.view({m, k});
    auto right = weight;
    auto dtype = LIBXSMM_DATATYPE_F32;
    auto inner = k;
    libxsmm_bitfield flags = LIBXSMM_GEMM_FLAG_BETA_0;
    if (x.scalar_type() == at::kBFloat16) {
        // VNNI2 consumes pairs along K. Pad an odd K without changing results.
        inner = af3_detail::xsmm_size(static_cast<int64_t>(k) + k % 2);
        input = pad_and_align_tensor(input, {m, inner}, true, 0.0, 64);
        right = pad_and_align_tensor(weight, {inner, weight_stride}, true, 0.0, 64);
        right = vnni_repack_tensor(right, 2, 0);
        dtype = LIBXSMM_DATATYPE_BF16;
        flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
        run_glu<at::BFloat16>(input, right, output, m, inner, n, mb, nb, weight_stride, dtype, flags);
    } else {
        run_glu<float>(input, right, output, m, inner, n, mb, nb, weight_stride, dtype, flags);
    }
    return output;
}
