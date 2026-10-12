// SPDX-License-Identifier: BSD-3-Clause
#include "tools.h"

#include <algorithm>
#include <omp.h>

namespace {

template <typename T>
void multiply(const at::Tensor& a, const at::Tensor& right, at::Tensor& output,
              const c10::optional<at::Tensor>& bias, int64_t batches, int64_t m,
              libxsmm_blasint k, libxsmm_blasint n, libxsmm_blasint mb, libxsmm_blasint nb,
              libxsmm_datatype dtype, libxsmm_bitfield flags) {
    const int64_t mt = (m - 1) / mb + 1;
    const int64_t nt = (static_cast<int64_t>(n) - 1) / nb + 1;
    const libxsmm_blasint heights[2] = {mb, static_cast<libxsmm_blasint>((m - 1) % mb + 1)};
    const libxsmm_blasint widths[2] = {nb, static_cast<libxsmm_blasint>((n - 1) % nb + 1)};
    libxsmm_gemmfunction kernels[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const auto shape = libxsmm_create_gemm_shape(
                widths[j], heights[i], k, n, k, nb,
                dtype, dtype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
            kernels[i][j] = libxsmm_dispatch_gemm(shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
            TORCH_CHECK(kernels[i][j] != nullptr, "LIBXSMM could not generate a tiled matrix kernel.");
        }
    }
    auto scratch = af3_detail::aligned_empty(
        at::empty({0}, a.options().dtype(at::kFloat)), {std::max(1, omp_get_max_threads()), mb, nb}, 64);
    const auto* x = a.data_ptr<T>();
    const auto* w = right.data_ptr<T>();
    const auto* bias_data = bias.has_value() ? bias->data_ptr<T>() : nullptr;
    auto* y = output.data_ptr<T>();
    auto* temporary = scratch.data_ptr<float>();
    const int64_t pack = dtype == LIBXSMM_DATATYPE_BF16 ? 2 : 1;
    const int64_t tile_elements = static_cast<int64_t>(mb) * nb;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t mi = 0; mi < mt; ++mi) {
            for (int64_t ni = 0; ni < nt; ++ni) {
                const int hi = mi == mt - 1 ? 1 : 0;
                const int wi = ni == nt - 1 ? 1 : 0;
                const int64_t row = mi * mb, col = ni * nb;
                auto* tile = temporary + static_cast<int64_t>(omp_get_thread_num()) * tile_elements;
                libxsmm_gemm_param params = {};
                params.a.primary = const_cast<T*>(w + batch * k * n + col * pack);
                params.b.primary = const_cast<T*>(x + batch * m * k + row * k);
                params.c.primary = tile;
                kernels[hi][wi](&params);
                for (int64_t r = 0; r < heights[hi]; ++r) {
                    for (int64_t c = 0; c < widths[wi]; ++c) {
                        float value = tile[r * nb + c];
                        if (bias_data != nullptr) value += static_cast<float>(bias_data[col + c]);
                        y[batch * m * n + (row + r) * n + col + c] = T(value);
                    }
                }
            }
        }
    }
}

}  // namespace

namespace af3_detail {

at::Tensor tiled_matmul(const at::Tensor& a, const at::Tensor& b,
                       const c10::optional<at::Tensor>& bias, int64_t m_block, int64_t n_block, bool b_packed) {
    check_tensor(a);
    check_tensor(b);
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && a.dim() == b.dim(), "Matrix operands must be rank 2 or 3.");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Matrix operands must be contiguous.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Matrix dtypes must match.");
    const auto k = xsmm_size(a.size(-1));
    TORCH_CHECK(!b_packed || a.scalar_type() == at::kBFloat16, "Packed matrices require BF16.");
    TORCH_CHECK(b.size(-2) == static_cast<int64_t>(k) + (b_packed ? k % 2 : 0),
                "Matrix inner dimensions must match the normal/padded packed weight.");
    TORCH_CHECK(a.dim() == 2 || a.size(0) == b.size(0), "Matrix batch dimensions must match.");
    const int64_t batches = a.dim() == 3 ? a.size(0) : 1;
    const int64_t m = a.size(-2);
    const auto n = xsmm_size(b.size(-1));
    const auto requested_mb = xsmm_size(m_block), requested_nb = xsmm_size(n_block);
    if (bias.has_value()) {
        check_tensor(*bias);
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n && bias->is_contiguous() &&
                    bias->scalar_type() == a.scalar_type(), "Bias must be a matching contiguous [N] vector.");
    }
    auto shape = a.sizes().vec();
    shape.back() = n;
    auto output = aligned_empty(a, shape, 64);
    if (batches == 0 || m == 0) return output;
    auto input = a.view({batches, m, k});
    auto right = b.view({batches, b.size(-2), n});
    auto inner = k;
    auto dtype = LIBXSMM_DATATYPE_F32;
    libxsmm_bitfield flags = LIBXSMM_GEMM_FLAG_BETA_0;
    if (a.scalar_type() == at::kBFloat16) {
        inner = xsmm_size(static_cast<int64_t>(k) + k % 2);
        input = pad_and_align_tensor(input, {batches, m, inner}, true, 0.0, 64);
        if (!b_packed) {
            right = pad_and_align_tensor(right, {batches, inner, n}, true, 0.0, 64);
            right = vnni_repack_tensor(right.view({batches * inner, n}), 2, 0).view({batches, inner, n});
        }
        dtype = LIBXSMM_DATATYPE_BF16;
        flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
    }
    const auto mb = static_cast<libxsmm_blasint>(std::min<int64_t>(requested_mb, m));
    const auto nb = std::min(requested_nb, n);
    if (dtype == LIBXSMM_DATATYPE_BF16) {
        multiply<at::BFloat16>(input, right, output, bias, batches, m, inner, n, mb, nb, dtype, flags);
    } else {
        multiply<float>(input, right, output, bias, batches, m, inner, n, mb, nb, dtype, flags);
    }
    return output;
}

}  // namespace af3_detail

at::Tensor xsmm_linear(const at::Tensor& x, const at::Tensor& weight,
                       const c10::optional<at::Tensor>& bias, bool weight_packed) {
    af3_detail::check_tensor(x);
    TORCH_CHECK(x.dim() >= 1 && x.is_contiguous(), "Linear input must be contiguous [..., K].");
    TORCH_CHECK(weight.dim() == 2, "Linear weight must be [K, N] or its padded packed layout.");
    const auto k = af3_detail::xsmm_size(x.size(-1));
    auto output = af3_detail::tiled_matmul(x.view({x.numel() / k, k}), weight, bias, 128, 128, weight_packed);
    auto shape = x.sizes().vec();
    shape.back() = weight.size(1);
    return output.view(shape);
}
